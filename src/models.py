import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super(Encoder, self).__init__()
        self.content_encoder1 = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.content_encoder2 = ContentEncoder(1, dim, n_residual, n_downsample)
        self.style_encoder1 = StyleEncoder(in_channels, dim, n_downsample, style_dim)
        self.style_encoder2 = StyleEncoder(1, dim, n_downsample, style_dim)

    def forward(self, x1, x2):
        content_code1 = self.content_encoder1(x1)
        content_code2 = self.content_encoder2(x2)
        style_code1 = self.style_encoder1(x1)
        style_code2 = self.style_encoder2(x2)

        return content_code1, content_code2, style_code1, style_code2


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, n_residual=3, n_upsample=2, style_dim=16):
        super(Decoder, self).__init__()
        in_dim = dim
        
        layers_local = []
        # Residual blocks
        for _ in range(n_residual):
            layers_local += [ResidualBlock(dim, norm="adain")]

        layers_up = []
        # Upsampling
        for n in range(n_upsample):
            layers_up += [
                nn.ConvTranspose2d(dim if n==0 else dim, dim//2, 5, 2, 0),
                LayerNorm(dim//2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers_up += [nn.ReflectionPad2d(3), 
                      nn.Conv2d(dim, out_channels, 7), 
                      nn.Tanh()]

        self.model_local = nn.Sequential(*layers_local)
        self.model_up = nn.Sequential(*layers_up)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, vi_content_code, ir_content_code, vi_style_code, ir_style_code):
        content_code = torch.cat([vi_content_code, ir_content_code], dim=1)
        style_code = torch.cat([vi_style_code, ir_style_code], dim=1)
        self.assign_adain_params(self.mlp(style_code))
        feat_local = self.model_local(content_code)
        im = self.model_up(feat_local).clamp(0,1)

        return im



class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels, dim, 7),
                  nn.InstanceNorm2d(dim),
                  nn.ReLU(inplace=True)]

        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                       nn.InstanceNorm2d(dim * 2),
                       nn.ReLU(inplace=True)]
            dim *= 2

        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()

        # Initial conv block
        layers = [nn.ReflectionPad2d(3), 
                  nn.Conv2d(in_channels, dim, 7), 
                  nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), 
                       nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), 
                       nn.ReLU(inplace=True)]

        # Average pool and output layer
        layers += [nn.AdaptiveAvgPool2d(1), 
                   nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
