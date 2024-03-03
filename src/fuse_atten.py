import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# cross Learnable Attention
class CLA(nn.Module):
    def __init__(self, 
                 in_channels=256, 
                 embed_dim=64,
                 out_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # input layer
        self.vi_in = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode="reflect"),
                                   nn.GroupNorm(1, in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode="reflect"),
                                   nn.GroupNorm(1, in_channels),

                                   nn.Flatten(-2, -1),
                                   nn.AdaptiveAvgPool1d(embed_dim),

                                   nn.Linear(embed_dim, embed_dim),
                                   nn.LayerNorm(embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(embed_dim, embed_dim),
                                   nn.LayerNorm(embed_dim))
        
        self.ir_in = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode="reflect"),
                                   nn.GroupNorm(1, in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode="reflect"),
                                   nn.GroupNorm(1, in_channels),

                                   nn.Flatten(-2, -1),
                                   nn.AdaptiveAvgPool1d(embed_dim),
                                   
                                   nn.Linear(embed_dim, embed_dim),
                                   nn.LayerNorm(embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(embed_dim, embed_dim),
                                   nn.LayerNorm(embed_dim))
        
        self.cat_in = nn.Sequential(nn.Conv2d(2*in_channels, 2*in_channels, 3, 1, 1, padding_mode="reflect"),
                                    nn.GroupNorm(1, 2*in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(2*in_channels, 2*in_channels, 3, 1, 1, padding_mode="reflect"),
                                    nn.GroupNorm(1, 2*in_channels),

                                    nn.Flatten(-2, -1),
                                    nn.AdaptiveAvgPool1d(embed_dim),
                                   
                                    nn.Linear(embed_dim, embed_dim),
                                    nn.LayerNorm(embed_dim),
                                    nn.ReLU(),
                                    nn.Linear(embed_dim, embed_dim),
                                    nn.LayerNorm(embed_dim))
        
        self.q_vi = nn.Linear(embed_dim, embed_dim)
        self.q_ir = nn.Linear(embed_dim, embed_dim)
        self.k_vi = nn.Linear(embed_dim, embed_dim)
        self.k_ir = nn.Linear(embed_dim, embed_dim)
        self.w_vi = nn.Sequential(nn.Flatten(-2, -1),
                                  nn.Linear(2*in_channels*embed_dim, 256),
                                  nn.Linear(256, 2*in_channels*in_channels))
        self.w_ir = nn.Sequential(nn.Flatten(-2, -1), 
                                  nn.Linear(2*in_channels*embed_dim, 256),
                                  nn.Linear(256, 2*in_channels*in_channels))

        self.conv_out = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 1, 1, 0),
                                      nn.InstanceNorm2d(out_channels))
        
    def forward(self, vif, irf):
        _, _, H, W = vif.shape
        catf = torch.cat([vif, irf], dim=1)
        catf_re = rearrange(catf, "b c h w -> b c (h w)")
        
        vif = self.vi_in(vif)
        irf = self.ir_in(irf)
        catf = self.cat_in(catf)

        q_vi = self.q_vi(vif)
        q_ir = self.q_ir(irf)
        k_vi = self.k_vi(catf)
        k_ir = self.k_ir(catf)
        w_vi = rearrange(self.w_vi(catf), "b (c1 c2) -> b c1 c2", c1=self.in_channels, c2=2*self.in_channels)
        w_ir = rearrange(self.w_ir(catf), "b (c1 c2) -> b c1 c2", c1=self.in_channels, c2=2*self.in_channels)

        score_vi = (torch.matmul(q_vi, k_ir.transpose(-2, -1)) + w_vi) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attention_weight_vi = F.softmax(score_vi, dim=-1)
        weighted_feat_vi = rearrange(torch.bmm(attention_weight_vi, catf_re), "b c (h w) -> b c h w", h=H, w=W)

        score_ir = (torch.matmul(q_ir, k_vi.transpose(-2, -1)) + w_ir) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attention_weight_ir = F.softmax(score_ir, dim=-1)
        weighted_feat_ir = rearrange(torch.bmm(attention_weight_ir, catf_re), "b c (h w) -> b c h w", h=H, w=W)

        y = self.conv_out(torch.cat([weighted_feat_vi, weighted_feat_ir], dim=1))

        # import ipdb
        # ipdb.set_trace()
        return y



if __name__ == "__main__":
    a = torch.rand([4,64,128,128])
    embed = CLA(64, 32, 64)
    ae = embed(a, a)
    print(ae.shape)
    print(' ')
    