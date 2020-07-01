import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import utils
from .net import MultiscaleEpe, Upsample, MaskFlownet_S, MaskFlownet,EpeLossWithMask
from utils.flow_utils import centralize
import pdb
class MaskFlowNetModel(object):
    
    def __init__(self, config):
        
        
        if config.model['class'] == 'MaskFlownet_S':
            self.net = MaskFlownet_S(config)
        else:
            self.net = MaskFlownet(config)
              
        utils.init_weights(self.net, init_type='xavier')
        self.net.cuda()
        self.strides = config.model['strides'] or [64, 32, 16, 8, 4]
        self.scale = self.strides[-1]
        
        self.optim = torch.optim.SGD(self.net.parameters(), lr=config.model['lr'],
            momentum=config.model['momentum'])      
        cudnn.benchmark = True
        
        multiscale_weights = config.model['multiscale_weights']
        self.multiscale_epe = MultiscaleEpe(
			scales = self.strides, weights = multiscale_weights, match = 'upsampling',
			eps = 1e-8, q= 0.4)
        self.epeloss = EpeLossWithMask(eps=1e-8, q=0.4)         
           
    def set_input(self, image0, image1, label, mask=None):
        if isinstance(mask, list):
            mask = torch.ones((label.shape[0], label.shape[1], label.shape[2], 1), dtype=label.dtype, device=label.device)
        self.image0 = image0.cuda().permute(0,3, 1, 2)
        self.image1 = image1.cuda().permute(0,3, 1, 2)
        self.label = label.cuda().permute(0, 3, 1, 2)
        self.mask = mask.cuda().permute(0,3, 1, 2)
       
        self.image0, self.image1, _= centralize(self.image0, self.image1)
        shape = self.image0.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            im0 = F.interpolate(im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
            im1 = F.interpolate(im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
                          
    def forward_only(self, image0, image1, label=None, mask= None):
        self.image0 = image0.cuda().permute(0,3, 1, 2)
        self.image1 = image1.cuda().permute(0,3, 1, 2)
       
        self.image0, self.image1, _= centralize(self.image0, self.image1)
        shape = self.image0.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            im0 = F.interpolate(im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
            im1 = F.interpolate(im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
         
    def criterion(self,label, mask, predictions):
        loss = self.multiscale_epe(label, mask, *predictions).mean()
        
        return loss
     
    # def adjust_learning_rate(self, lr):
    #     for param_group in self.optim.param_groups:
    #         param_group['lr'] = lr
           
    def step(self):  
        preds, flows, warpeds = self.net(self.image0, self.image1)
        # up_flow = Upsample(pred[-1], 4)
        # up_occ_mask = Upsample(flows[0], 4)
        self.label = self.label.flip(1)
        loss = self.criterion(self.label, self.mask, preds)
        # pdb.set_trace()
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.net)
        self.optim.step()
        return {'loss': loss}
         
    def load_state(self, path, Iter, resume=False):
        model_path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))
        if resume:
            utils.load_state(model_path, self.net, self.optim)
        else:
            utils.load_state(model_path, self.net)
    
    def save_state(self, path, Iter):
        model_path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optim.state_dict()}, model_path)
    
    def switch_to(self, phase):
        if phase == 'train':
            self.net.train()          
        else:
            self.net.eval()
            
    def eval(self):
        pass                 
                   
