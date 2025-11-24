import time
import os
import itertools
import clip
import random
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm
# self lib
from models import *
from config import args as args_config
from dataset import TrainData
from vgg import VGG


class Trainer(object):
    def __init__(self,
                args):
        super().__init__()
        self.args = args

        self.if_resume = False

        self.sim_weight = 6
        self.prompt_weight_g = 2
        self.prompt_weight_p = 10
        self.perception_weight = 10
        self.patch_select = 6

        # device
        self.cuda = torch.cuda.is_available()
        self.dev = 'cuda:'+str(GLOBAL_RANK) if self.cuda else 'cpu'

        # clip config
        self.clip_model, _ = clip.load("ViT-B/32")
        self.vgg = VGG()

        vi_text = ['a visible gray image']
        ir_text = ['an infrared image']

        self.prompt = ['a clear image with detailed background and salient objects']
        print(f'prompt: {self.prompt}')

        vit_token = clip.tokenize(vi_text).to(self.dev)
        self.vit_feature = self.clip_model.encode_text(vit_token).detach().to(torch.float)
        self.vit_feature = self.vit_feature.mean(dim=0, keepdim=True)
        self.vit_dir = self.vit_feature / self.vit_feature.norm(dim=-1, keepdim=True)

        irt_token = clip.tokenize(ir_text).to(self.dev)
        self.irt_feature = self.clip_model.encode_text(irt_token).detach().to(torch.float)
        self.irt_feature = self.irt_feature.mean(dim=0, keepdim=True)
        self.irt_dir = self.irt_feature / self.irt_feature.norm(dim=-1, keepdim=True)

        prompt_token = clip.tokenize(self.prompt).to(self.dev)
        self.prompt_feature = self.clip_model.encode_text(prompt_token).detach().to(torch.float)
        self.prompt_feature = self.prompt_feature.mean(dim=0, keepdim=True)
        self.prompt_dir = self.prompt_feature / self.prompt_feature.norm(dim=-1, keepdim=True)

        # Initialize encoders, generators
        self.Enc = Encoder(in_channels=1, dim=16, n_downsample=3, n_residual=2, style_dim=8)
        self.Dec = Decoder(out_channels=1, dim=256, n_upsample=3, n_residual=2, style_dim=16)
        
        # Initialize weights
        if self.if_resume:
            self.load(load_epoch=100, pt_path='./experiments/pretrained/')
        else:
            self.Enc.apply(weights_init_normal)
            self.Dec.apply(weights_init_normal)

        if self.cuda:
            self.Enc.cuda()
            self.Dec.cuda()
            self.clip_model.cuda()
            self.vgg.cuda()

        if MUL_GPU:
            self.Enc = DDP(self.Enc, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
            self.Dec = DDP(self.Dec, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

        train_dataset = TrainData(self.prompt_feature.clone().cpu(), self.prompt_dir.clone().cpu())            

        if MUL_GPU:
            train_sampler = DistributedSampler(train_dataset, 
                                               shuffle=True, 
                                               drop_last=True)
            self.train_dataloader = DataLoader(dataset=train_dataset, 
                                               batch_size=args.train_batch, 
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)
        else:
            self.train_dataloader = DataLoader(dataset=train_dataset, 
                                               batch_size=args.train_batch, 
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)            

        # Optimizers
        self.optimizer_G = Adam(itertools.chain(self.Enc.parameters(),
                                                self.Dec.parameters()),
                                lr=args.lr, 
                                betas=args.betas)

        # Learning rate update schedulers
        self.warm_lr_G = LambdaLR(self.optimizer_G, lr_lambda=lambda x:(x+1)/len(self.train_dataloader))
        self.lr_scheduler_G = MultiStepLR(self.optimizer_G, 
                                            milestones=self.args.lr_mstone, 
                                            gamma=self.args.lr_decay_gamma)
                        
        if GLOBAL_RANK == 0:                
            self.results_folder = Path(self.args.save_dir)
            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)


    def dynamic_cropper(self, vi, ir, fusion, num_crops=64):
        transform = transforms.Compose([transforms.RandomCrop(int(224*random.uniform(0.5,0.8))),
                                        transforms.RandomAffine(degrees=[-5,5],
                                                                translate=[0,0.1],
                                                                scale=[1,1.5],
                                                                shear=5),
                                        transforms.Resize(224)])  
        vi = vi.repeat(1,3,1,1) if vi.shape[1] == 1 else vi
        ir = ir.repeat(1,3,1,1) if ir.shape[1] == 1 else ir
        fusion = fusion.repeat(1,3,1,1) if fusion.shape[1] == 1 else fusion

        vi_cropped = []
        ir_cropped = []
        fu_cropped = []

        for _ in range(num_crops):
            catted = torch.cat([vi, ir, fusion], dim=0)
            transed = transform(catted)
            chunked = torch.chunk(transed, 3, dim=0)
            vi_cropped.append(chunked[0])
            ir_cropped.append(chunked[1])
            fu_cropped.append(chunked[2])        

        return torch.cat(vi_cropped, dim=0), torch.cat(ir_cropped, dim=0), torch.cat(fu_cropped, dim=0)


    def clip_norm(self, im):      
        DEV = im.device   
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        im_re = F.interpolate(im.repeat(1,3,1,1) if im.shape[1]==1 else im, size=224, mode='bilinear', align_corners=False)
        im_norm = (im_re - mean) / std
        return im_norm


    def clip_prompt_loss(self, fu_feature, vi_feature, ir_feature, prompt_feature, prompt_dir, if_foucs=False, select=None):
        BP = fu_feature.shape[0]
        BG = prompt_feature.shape[0]

        prompt_feature = prompt_feature.repeat(BP//BG, 1)
        prompt_dir = prompt_dir.repeat(BP//BG, 1)

        vi_feature = vi_feature.repeat(BP//BG, 1)
        ir_feature = ir_feature.repeat(BP//BG, 1)

        # ----- im feature -----
        vi_len = vi_feature.norm(dim=-1, keepdim=True)
        vi_dir = vi_feature / vi_len
        
        ir_len = ir_feature.norm(dim=-1, keepdim=True)
        ir_dir = ir_feature / ir_len

        fusion_len = fu_feature.norm(dim=-1, keepdim=True)
        fusion_dir = fu_feature / fusion_len

        # ----- im direction -----
        vi_fu = fusion_dir - vi_dir
        vi_fu_dir = vi_fu / vi_fu.norm(dim=-1, keepdim=True)

        ir_fu = fusion_dir - ir_dir
        ir_fu_dir = ir_fu / ir_fu.norm(dim=-1, keepdim=True)

        # ----- text direction -----
        vit_pt = prompt_dir - self.vit_dir.repeat(BP, 1)
        vit_pt_dir = vit_pt / vit_pt.norm(dim=-1, keepdim=True)

        irt_pt = prompt_dir - self.irt_dir.repeat(BP, 1)
        irt_pt_dir = irt_pt / irt_pt.norm(dim=-1, keepdim=True)

        # ----- calc loss -----
        loss_prompt = 0.5 * (1 - torch.cosine_similarity(vi_fu_dir, vit_pt_dir, dim=1)) + \
                      0.5 * (1 - torch.cosine_similarity(ir_fu_dir, irt_pt_dir, dim=1))

        if select is not None:
            loss_prompt = loss_prompt * select

        return loss_prompt.mean()


    def clip_sim_loss(self, fu_feature, vi_feature, ir_feature):
        vi_len = vi_feature.norm(dim=-1, keepdim=True)
        vi_dir = vi_feature / vi_len
        
        ir_len = ir_feature.norm(dim=-1, keepdim=True)
        ir_dir = ir_feature / ir_len

        fusion_len = fu_feature.norm(dim=-1, keepdim=True)
        fusion_dir = fu_feature / fusion_len

        loss_dir = 0.5 * (1 - torch.cosine_similarity(fusion_dir, vi_dir, dim=-1)) + \
                   0.5 * (1 - torch.cosine_similarity(fusion_dir, ir_dir, dim=-1))

        return loss_dir.mean()


    def vgg_loss(self, fu, vi, ir):
        fu_vgg = self.vgg.get_features(fu, False)
        vi_vgg = self.vgg.get_features(vi, False)
        ir_vgg = self.vgg.get_features(ir, False)

        feat2 = torch.cat([vi_vgg['conv2_1'].unsqueeze(2), ir_vgg['conv2_1'].unsqueeze(2)], dim=2)
        feat2 = torch.max(feat2, dim=2, keepdim=False)[0]

        feat3 = torch.cat([vi_vgg['conv3_1'].unsqueeze(2), ir_vgg['conv3_1'].unsqueeze(2)], dim=2)
        feat3 = torch.max(feat3, dim=2, keepdim=False)[0]

        feat4 = torch.cat([vi_vgg['conv4_2'].unsqueeze(2), ir_vgg['conv4_2'].unsqueeze(2)], dim=2)
        feat4 = torch.max(feat4, dim=2, keepdim=False)[0]

        loss_2 = F.mse_loss(fu_vgg['conv2_1'], feat2, reduction='mean')
        loss_3 = F.mse_loss(fu_vgg['conv3_1'], feat3, reduction='mean')
        loss_4 = F.mse_loss(fu_vgg['conv4_2'], feat4, reduction='mean')

        return loss_2 + loss_3 + loss_4


    def train(self):
        for self.epoch in range(1, self.args.epochs+1):
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            if GLOBAL_RANK == 0:
                print('=== Epoch {:5d} / {:5d} | Lr : {:.4f} | {} | {} ==='
                    .format(self.epoch, self.args.epochs, self.optimizer_G.param_groups[0]['lr'], current_time, args.save_dir))
                
            tqdm_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)) if GLOBAL_RANK == 0 else enumerate(self.train_dataloader)

            for _, sample in tqdm_bar:   
                # Set model input
                X1 = sample['vi_y'].to(device=self.dev)
                X2 = sample['ir_y'].to(device=self.dev)
                X1_N = sample['vi_n'].to(device=self.dev)
                X2_N = sample['ir_n'].to(device=self.dev)
                PF = sample['prompt_feature'].to(device=self.dev)
                PD = sample['prompt_dir'].to(device=self.dev)
                B, C, H, W = X1.shape

                self.optimizer_G.zero_grad()

                c_code_1, c_code_2, s_code_1, s_code_2 = self.Enc(X1_N, X2_N)              

                fusion_y = self.Dec(c_code_1, c_code_2, s_code_1, s_code_2)
                fusion_y = fusion_y[:,:,:H,:W]
                vi_patch, ir_patch, fu_patch = self.dynamic_cropper(X1, X2, fusion_y, 32) # at least 32 pathces 

                vi_feature = self.clip_model.encode_image(self.clip_norm(X1))
                ir_feature = self.clip_model.encode_image(self.clip_norm(X2))
                fu_feature = self.clip_model.encode_image(self.clip_norm(fusion_y))
                fu_patch_feature = self.clip_model.encode_image(self.clip_norm(fu_patch))

                vip_en = entropy_t4d(vi_patch, if_norm=False)
                irp_en = entropy_t4d(ir_patch, if_norm=False)
                valid_index = (vip_en > self.patch_select) * (irp_en > self.patch_select)
                valid_index = valid_index.to(vi_patch.device)

                loss_prompt_g = self.clip_prompt_loss(fu_feature, vi_feature, ir_feature, PF, PD, False)
                loss_prompt_p = self.clip_prompt_loss(fu_patch_feature, vi_feature, ir_feature, PF, PD, True, valid_index)

                loss_sim = self.clip_sim_loss(fu_feature, vi_feature, ir_feature) 

                loss_prompt = self.prompt_weight_g * loss_prompt_g + \
                              self.prompt_weight_p * loss_prompt_p
                loss_sim = self.sim_weight * loss_sim
                    
                perception_loss = self.perception_weight * self.vgg_loss(fusion_y, X1, X2)

                loss = loss_sim + loss_prompt + perception_loss

                loss.backward()
                self.optimizer_G.step() 

                detail = 0.002 * get_detail(fusion_y)

                if self.args.if_warm_up and self.epoch == 1:
                    self.warm_lr_G.step() 
                    self.lr_scheduler_G.step()
                    if GLOBAL_RANK == 0:
                        current_lr = self.optimizer_G.param_groups[0]['lr']
                        s += f'Lr: {current_lr:.2e} | ' 

                if GLOBAL_RANK == 0:
                    s = f'Train | tt:{loss:.2f} | dt:{detail:.2f} | '
                    tqdm_bar.set_description(s)
            
            if GLOBAL_RANK == 0:
                self.save()


    def save(self):
        torch.save(self.Enc.module.state_dict() if MUL_GPU else self.Enc.state_dict(), '{}/Enc_{:05d}.pt'.format(self.args.save_dir, self.epoch))
        torch.save(self.Dec.module.state_dict() if MUL_GPU else self.Dec.state_dict(), '{}/Dec_{:05d}.pt'.format(self.args.save_dir, self.epoch))         


    def load(self, load_epoch, pt_path):
        dev = 'cuda:'+str(GLOBAL_RANK) if torch.cuda.is_available() else 'cpu'

        Enc_w = torch.load(str(pt_path + f'/Enc_{load_epoch:05d}.pt'), map_location=dev)
        self.Enc.load_state_dict(Enc_w)

        Dec_w = torch.load(str(pt_path + f'/Dec_{load_epoch:05d}.pt'), map_location=dev)
        self.Dec.load_state_dict(Dec_w)

        print('Pretrained weight load done !')


if __name__ == '__main__':
    MUL_GPU = False if torch.cuda.device_count() <= 1 else True

    print('MUL_GPU == ', MUL_GPU)

    if MUL_GPU:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args_config.local_rank)
        GLOBAL_RANK = dist.get_rank()
    else:
        GLOBAL_RANK = 0

    # config
    args = args_config

    if GLOBAL_RANK == 0:
        print('\n\n=== Arguments ===')
        cnt = 0
        for key in sorted(vars(args)):
            print(key, ':',  getattr(args, key), end='  |  ')
            cnt += 1
            if (cnt + 1) % 5 == 0:
                print('')
        print('\n')

    trainer = Trainer(args)
    trainer.train()
