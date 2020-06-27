import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class DENSODataset(Dataset):
    def __init__(self, denso_root, split_file, resize=None, samples = None):
        self.denso_root = denso_root
        self.split_file = split_file
        self.resize = resize
        denso_image = os.path.join(denso_root)
        num_files = len(os.listdir(denso_image))-1
        self.image_list = []
        self.image_lines = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                self.image_lines.append(sp)
                
        
    
    def __len__(self):
        return len(self.image_lines)
    
    def __getitem__(self,idx):
        img_path= self.image_lines[idx]
        
        flow = []
        occ = []
        
        
        img0_path = img_path[0]
        img1_path = img_path[1]
        img0_path =os.path.join(self.denso_root, img0_path)
        img1_path = os.path.join(self.denso_root, img1_path)
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        
        if self.resize is not None:
            img0 = cv2.resize(img0, self.resize)
            img1 = cv2.resize(img1, self.resize)
        
        img0 = torch.tensor(img0/255.).float()
        img1 = torch.tensor(img1/255.).float()
    
        return img0, img1, flow, occ, img0_path              