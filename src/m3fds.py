import os
import random
import cv2
import torch
from einops import rearrange
from torchvision import transforms
from torchvision.io.image import read_image, ImageReadMode
from torch.utils.data import Dataset
import numpy as np


class M3fdMix(Dataset):
    def __init__(self,
                prompt_feature,
                prompt_dir,
                im_h=256, 
                im_w=256, 
                if_transform=True,
                if_pair=True,
                repeat=[1],
                step=[1]):
        self.prompt_feature = prompt_feature
        self.prompt_dir = prompt_dir
        self.prompt_len = len(prompt_feature)
        self.im_h = im_h
        self.im_w = im_w
        self.if_transform = if_transform
        self.if_pair = if_pair

        data_dir = ['/home/hosthome/Datasets/M3FD/M3FD_Detection/']
        vis_folder = ['vi/']
        ir_folder = ['ir/']       

        self.vis_file_list = []
        self.ir_file_list = []
        for ind, ins in enumerate(data_dir):
            vis_dir = ins + vis_folder[ind]
            ir_dir = ins + ir_folder[ind]
            vis_file = [vis_dir + i for i in os.listdir(vis_dir)]
            ir_file = [ir_dir + i for i in os.listdir(ir_dir)]
            self.vis_file_list.extend(vis_file[::step[ind]] * repeat[ind])
            self.ir_file_list.extend(ir_file[::step[ind]] * repeat[ind])

        if not if_pair:
            random.shuffle(self.vis_file_list)
            random.shuffle(self.ir_file_list)

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
        vi_rgb = read_image(self.vis_file_list[idx], ImageReadMode.RGB)
        ir = read_image(self.ir_file_list[idx], ImageReadMode.RGB)

        vi_rgb, ir = self.augment(vi_rgb, ir)

        crcb = self._get_crcb(vi_rgb)
        ir_rgb = self._to_rgb(vi_rgb, ir)
        vi_y = self._to_y(vi_rgb)
        ir_y = self._to_y(ir)

        id = random.randrange(0, self.prompt_len)

        return {'vi_rgb': vi_rgb / 255., 
                'ir_rgb': ir_rgb / 255., 
                'vi_y': vi_y / 255., 
                'ir_y': ir_y / 255.,
                'vi_y_n': vi_y / 255. + 0.03*torch.randn_like(vi_y*1.), 
                'ir_y_n': ir_y / 255. + 0.03*torch.randn_like(ir_y*1.),
                'prompt_feature': self.prompt_feature[id],
                'prompt_dir': self.prompt_dir[id],
                'crcb': crcb / 255.}

    def augment(self, viy, iry):     
        resize = transforms.Resize(min(self.im_h,self.im_w))
        vi_rs = resize(viy)
        ir_rs = resize(iry)
        vi_rs = vi_rs[:,:ir_rs.shape[1],:ir_rs.shape[2]]
        ir_rs = ir_rs[:,:vi_rs.shape[1],:vi_rs.shape[2]]

        if self.if_transform:
            transform = transforms.Compose([transforms.RandomAffine(degrees=[-15,15],
                                                                    translate=[0,0.1],
                                                                    scale=[1,1.5],
                                                                    shear=15),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomCrop([self.im_h,self.im_w], 
                                                                   pad_if_needed=True, 
                                                                   padding_mode='reflect')])
        else:
            transform = transforms.CenterCrop([self.im_h,self.im_w])       

        if self.if_pair:
            vi_ir = torch.cat([vi_rs, ir_rs], dim=0)        
            vi_ir_t = transform(vi_ir)
            viy_t, iry_t = torch.chunk(vi_ir_t, 2, 0)
        else:
            viy_t = transform(vi_rs)
            iry_t = transform(ir_rs)

        return viy_t, iry_t

        