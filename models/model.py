import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import utils
from .net import MultiscaleEpe
from .net import Upsample
from utils.flow_utils import centralize
class MaskFlownetModel(object):
    
    def __init__(self, config):
        self.net = config.model['class'](config)
        utils.init_weights(self.net, init_type='xavier')
        self.net.cuda()
        self.world_size = dist.get_world_size()
        self.strides = config.model['strides'] or [64, 32, 16, 8, 4]
        self.scale = self.strides[-1]
        
        multiscale_weights = config.model['multiscale_weights']
        self.criterion = MultiscaleEpe(
			scales = self.strides, weights = multiscale_weights, match = 'upsampling',
			eps = 1e-8, q= None)
        
        if config['optim'] == 'SGD':
            self.optim = torch.optim.SGD(
                self.net.parameters(), lr=config['lr'],
                momentum=config['momentum'], weight_decay=config['weight_decay'])
        elif config['optim'] == 'Adam':
            self.optim = torch.optim.Adam(
                self.net.parameters(), lr=config['lr'],
                betas=(config['beta1'], 0.999))
        else:
            raise Exception("No such optimizer: {}".format(config['optim']))
        
        cudnn.benchmark = True
    
    def set_input(self, image0, image1, label=None, mask=None):
        self.image0 = image0.cuda().permute(0,3, 1, 2)
        self.image1 = image1.cuda().permute(0,3, 1, 2)
        self.label = label.permute(0,3, 1, 2)
        self.mask = mask.permute(0,3, 1, 2)
       
        self.image0, self.image1, _= centralize(self.image0, self.image1)
        shape = self.image0.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            im0 = F.interpolate(im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
            im1 = F.interpolate(im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
                          
    def eval(self):
        pass
    
    def forward_only(self):
        pass
    def step(self):
        pred, flows, warpeds = self.net(self.image0, self.image1)
        
        loss = self.criterion(pred, flows, self.label, self.mask) / self.world_size
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
            
                    
                