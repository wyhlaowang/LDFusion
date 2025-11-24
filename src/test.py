import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.io.image import read_image, ImageReadMode
from einops import rearrange
from models import Encoder, Decoder


def rgb_y(im):
    DEV = im.device
    im_ra = rearrange(im, 'c h w -> h w c').cpu().numpy()
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:,:,0]).unsqueeze(0).to(device=DEV)
    return im_y


def to_rgb(im_3, im_1):
    DEV = im_1.device
    im_3 = rearrange(im_3, 'c h w -> h w c').cpu().numpy()
    im_1 = rearrange(im_1, 'c h w -> h w c').cpu().numpy()
    crcb = cv2.cvtColor(im_3, cv2.COLOR_RGB2YCrCb)[:,:,1:]
    ycrcb = np.concatenate((im_1, crcb), -1)
    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return rearrange(torch.from_numpy(rgb), 'h w c -> c h w').to(device=DEV)


def load(pt_path, device='cuda:1'):
    Enc = Encoder(in_channels=1, dim=16, n_downsample=3, n_residual=2, style_dim=8).to(device=device).eval()
    Dec = Decoder(out_channels=1, dim=256, n_upsample=3, n_residual=2, style_dim=16).to(device=device).eval()

    Enc_w = torch.load(str(pt_path + f'/Enc_final.pt'), map_location=device)
    Enc.load_state_dict(Enc_w)

    Dec_w = torch.load(str(pt_path + f'/Dec_final.pt'), map_location=device)
    Dec.load_state_dict(Dec_w)

    print('=== Pretrained models load done ===')
    
    return Enc, Dec


def t4d_save(t4d, save_path, save_file_name, if_print=False):
    C = t4d.shape[1]
    if C == 1:
        im = 255 * t4d.cpu().squeeze(0).squeeze(0).clamp(0,1)
        im = Image.fromarray(im.numpy().astype('uint8'))
    else:
        im = 255 * t4d.cpu().squeeze(0).clamp(0,1)
        im = Image.fromarray(rearrange(im, 'c h w -> h w c').numpy().astype('uint8'))
    
    save_file = save_path + save_file_name
    im.save(save_file, quality=100)

    if if_print: print(f'Saved: {save_file}')


def test(dev='cuda:0', pt_folder='./weight/'): 
    Enc, Dec = load(pt_folder, dev)
  
    ir_path = "./test_imgs/ir/"
    vis_path = "./test_imgs/vi/"

    file_list = os.listdir(ir_path)

    print(f'Testing ... ')

    with torch.no_grad():
        for i in file_list:
            vis_o = read_image(vis_path + i, ImageReadMode.RGB).to(device=dev) / 255.
            ir_o = read_image(ir_path + i, ImageReadMode.GRAY).to(device=dev) / 255.
            vis = vis_o.to(device=dev)
            ir = ir_o.to(device=dev)
            vi_1 = rgb_y(vis)
            _, H, W = vis.shape

            vi_c, ir_c, vi_s, ir_s = Enc(vi_1.unsqueeze(0), ir.unsqueeze(0))
            fu = Dec(vi_c, ir_c, vi_s, ir_s)
            
            fu = fu[:,:,:H,:W].squeeze(0)
            fu_3 = to_rgb(vis, fu)
            cat = torch.cat([ir_o.repeat(3,1,1), vis_o, fu_3], dim=2).unsqueeze(0)
            t4d_save(fu_3.unsqueeze(0), "./results/", i, if_print=True)
            t4d_save(cat, "./results/", 'c'+i, if_print=True)


if __name__ == '__main__':
    test(dev='cuda:0', 
        pt_folder='./weight/')

