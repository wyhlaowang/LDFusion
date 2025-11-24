import os
import random
import cv2
import torch
from einops import rearrange
from torchvision import transforms
from torchvision.io.image import read_image, ImageReadMode
from torch.utils.data import Dataset
import numpy as np


class TrainData(Dataset):
    def __init__(self,
                prompt_feature,
                prompt_dir,
                im_h=224, 
                im_w=224, 
                if_transform=True,
                if_pair=True):
        self.prompt_feature = prompt_feature
        self.prompt_dir = prompt_dir
        self.prompt_len = len(prompt_feature)
        self.im_h = im_h
        self.im_w = im_w
        self.if_transform = if_transform
        self.if_pair = if_pair

        vis_dir = './YOUR_DATA_DIR/vi/'
        ir_dir = './YOUR_DATA_DIR/ir/'

        self.vis_file_list = []
        self.ir_file_list = []

        file_ls = os.listdir(vis_dir)
        vis_file = [vis_dir + i for i in file_ls]
        ir_file = [ir_dir + i for i in file_ls]

        # Repeat to ensure multiple patches can be sampled from each image
        self.vis_file_list.extend(vis_file*10) 
        self.ir_file_list.extend(ir_file*10)  

    def __len__(self):
        return len(self.vis_file_list)
    
    def _to_y(self, im):
        im_ra = rearrange(im, 'c h w -> h w c').numpy()
        im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
        im_y = torch.from_numpy(im_ycrcb[:,:,0]).unsqueeze(0)       

        return im_y 

    def _to_rgb(self, im_rgb, im_y):
        im_rgb_ra = rearrange(im_rgb, 'c h w -> h w c').numpy()
        im_y_ra = rearrange(im_y, 'c h w -> h w c').numpy()
        y = np.expand_dims(im_y_ra[:,:,0], -1)
        crcb = cv2.cvtColor(im_rgb_ra, cv2.COLOR_RGB2YCrCb)[:,:,1:]
        ycrcb = np.concatenate((y, crcb), -1)
        ir_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        return rearrange(torch.from_numpy(ir_rgb), 'h w c -> c h w')

    def _get_crcb(self, im):
        im_ra = rearrange(im, 'c h w -> h w c').numpy()
        crcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)[:,:,1:]
        return rearrange(torch.from_numpy(crcb), 'h w c -> c h w')
    
    def __getitem__(self, idx):     
        vi_3 = read_image(self.vis_file_list[idx], ImageReadMode.RGB)
        ir = read_image(self.ir_file_list[idx], ImageReadMode.GRAY)

        vi_3, ir = self.augment(vi_3, ir)
        vi_1 = self._to_y(vi_3)

        vi_3 = vi_3 / 255.
        vi_1 = vi_1 / 255.
        ir = ir / 255.

        id = random.randrange(0, self.prompt_len)

        if vi_1.std() < 0.05 and ir.std() < 0.05:
            return self.__getitem__(idx)
        else:
            return {'vi_y': vi_1, 
                    'ir_y': ir,
                    'vi_n': vi_1 + 0.03*torch.randn_like(vi_1), # data noise augment
                    'ir_n': ir + 0.03*torch.randn_like(ir),
                    'prompt_feature': self.prompt_feature[id],
                    'prompt_dir': self.prompt_dir[id]}

    # data augment
    def augment(self, vi_3, ir):     
        transform = transforms.Compose([transforms.RandomCrop([self.im_h,self.im_w], 
                                                            pad_if_needed=True, 
                                                            padding_mode='reflect'),
                                        transforms.RandomAffine(degrees=[-5,5],
                                                                translate=[0,0.1],
                                                                scale=[1,1.5],
                                                                shear=5)])
        vi_ir = torch.cat([vi_3, ir], dim=0)        
        vi_ir_t = transform(vi_ir)
        vi_t, ir_t = torch.split(vi_ir_t, [3, 1], dim=0)

        return vi_t, ir_t



        
