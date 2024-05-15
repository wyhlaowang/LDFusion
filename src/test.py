import os
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn.functional as F
from sys import platform
from PIL import Image
from torchvision.io.image import read_image, ImageReadMode
from config import args as args_config
from einops import rearrange
from models import Encoder, Decoder
from template import imagenet_templates


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def rgb_y(im):
    im_ra = rearrange(im, 'c h w -> h w c').numpy()
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:,:,0]).unsqueeze(0)  
    return im_y 

def clip_norm(im):      
    DEV = im.device   
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im_re = F.interpolate(im.repeat(1,3,1,1) if im.shape[1]==1 else im, size=224, mode='bilinear', align_corners=False)
    im_norm = (im_re - mean) / std
    return im_norm

def to_rgb(im_rgb, im_y):
    im_rgb_ra = rearrange(im_rgb.squeeze(0), 'c h w -> h w c').cpu().numpy()
    im_y_ra = rearrange(im_y.squeeze(0), 'c h w -> h w c').cpu().numpy()
    y = np.expand_dims(im_y_ra[:,:,0], -1)
    crcb = cv2.cvtColor(im_rgb_ra, cv2.COLOR_RGB2YCrCb)[:,:,1:]
    ycrcb = np.concatenate((y, crcb), -1)
    ir_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return rearrange(torch.from_numpy(ir_rgb), 'h w c -> c h w').unsqueeze(0).to(device=im_y.device)

def load(pt_path, load_epoch, device='cuda:1'):
    Enc = Encoder(in_channels=1, dim=16, n_downsample=3, n_residual=2).to(device=device).eval()
    Dec = Decoder(out_channels=1, dim=256, n_upsample=3, n_residual=2).to(device=device).eval()

    Enc_w = torch.load(str(pt_path + f'/Enc_{load_epoch:05d}.pt'), map_location=device)
    Enc.load_state_dict(Enc_w)

    Dec_w = torch.load(str(pt_path + f'/Dec_{load_epoch:05d}.pt'), map_location=device)
    Dec.load_state_dict(Dec_w)

    Atten = None

    print('=== Pretrained models load done ===')
    
    return Enc, Dec, Atten

def t4d_save(t4d, epoch, save_path, save_file_name, if_print=False):
    C = t4d.shape[1]
    if C == 1:
        im = 255 * t4d.cpu().squeeze(0).squeeze(0).clamp(0,1)
        im = Image.fromarray(im.numpy().astype('uint8'))
    else:
        im = 255 * t4d.cpu().squeeze(0).clamp(0,1)
        im = Image.fromarray(rearrange(im, 'c h w -> h w c').numpy().astype('uint8'))
    
    if epoch == 'null':
        save_file = save_path + 'single/' + save_file_name
        im.save(save_file, quality=100)
    else:
        save_file = save_path + str(epoch) + '_' + save_file_name
        im.save(save_file, quality=100)
        
    if if_print: print(f'Saved: {save_file}')


def test(dev='cuda:0', data_type=2, epoch=1, pt_folder=args_config.save_dir): 
    Enc, Dec, Atten = load(pt_folder, epoch, dev)

    if platform == 'win32':
        data_folder = ['D:/dataset/FusionData/TNO/tno/', 
                       'D:/dataset/FusionData/RoadScene/', 
                       'D:/dataset/FusionData/M3FD/M3FD_Detection/']  
    elif platform == 'linux':
        data_folder = ['./test_imgs/TNO_test', 
                       './test_imgs/RoadScene_test', 
                       '/home/hosthome/Datasets/M3FD/M3FD_Fusion']  
    
    save_path = ['./self_results/TNO_test/', 
                 './self_results/RoadScene_test/', 
                 './self_results/M3FD/']
    
    ir_folder = ['ir', 'ir', 'ir']
    vis_folder = ['vi', 'vi', 'vi']

    ir_path = data_folder[data_type] + '/' + ir_folder[data_type] + '/'
    vis_path = data_folder[data_type] + '/' + vis_folder[data_type] + '/'

    file_list = os.listdir(ir_path)

    # clip    
    print(f'Testing ... ')

    with torch.no_grad():
        for i in file_list:
        # for i in ["38.png"]:
            vis = read_image(vis_path + i, ImageReadMode.RGB) / 255.
            ir = read_image(ir_path + i, ImageReadMode.RGB) / 255.

            vis_y = rgb_y(vis).unsqueeze(0).to(device=dev)
            ir_y = rgb_y(ir).unsqueeze(0).to(device=dev)

            B, C, H, W = vis_y.shape

            vi_c, ir_c, vi_s, ir_s = Enc(vis_y, ir_y)

            fu = Dec(vi_c, ir_c, vi_s, ir_s)
            fu = fu[:,:,:H,:W]

            if data_type == 1 or data_type == 2:
                vis_y_r = vis_y.repeat(1,3,1,1)
                ir_y_r = ir_y.repeat(1,3,1,1)
                vis_rgb = vis.unsqueeze(0).to(device=dev)
                fu_rgb = to_rgb(vis_rgb, fu)
                cat = torch.cat([torch.cat([vis_y_r, ir_y_r], dim=3), torch.cat([vis_rgb, fu_rgb], dim=3)], dim=2)
                t4d_save(cat, epoch, save_path[data_type], 'rgb_'+i, if_print=True)
                t4d_save(fu_rgb, 'null', save_path[data_type], i, if_print=True)
            else:
                cat = torch.cat([vis_y, ir_y, fu], dim=3)
                t4d_save(cat, epoch, save_path[data_type], 'fu_'+i, if_print=True)
                t4d_save(fu, 'null', save_path[data_type], i, if_print=True)


if __name__ == '__main__':
    # config
    args = args_config

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':',  getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')

    print('\n')
    test(dev='cuda', 
        data_type=2, 
        epoch=100, 
        pt_folder='./experiments/final/')


