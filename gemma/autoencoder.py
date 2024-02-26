from torch import nn
import torch, torchaudio, torchvision
import torchaudio.functional as audio_F
import torchaudio.transforms as audio_T
import librosa
import numpy as np

def precompute_freqs_cis_2d(dim: int,
                            height: int,
                            width: int,
                            theta: float = 10000.0) -> torch.Tensor:
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t_h = torch.arange(height, device=freqs_h.device)
    t_w = torch.arange(width, device=freqs_w.device)
    freqs_h = torch.outer(t_h, freqs_h).float()
    freqs_w = torch.outer(t_w, freqs_w).float()
    freqs_2d = torch.einsum('h,w->hw', [freqs_h, freqs_w])
    freqs_cis_2d = torch.polar(torch.ones_like(freqs_2d), freqs_2d)

    return freqs_cis_2d

def precompute_freqs_cis_3d(dim: int, height: int, width: int, time: int, theta: float = 10000.0) -> torch.Tensor:
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t_h = torch.arange(height, device=freqs_h.device)
    t_w = torch.arange(width, device=freqs_w.device)
    t_t = torch.arange(time, device=freqs_t.device)
    freqs_h = torch.outer(t_h, freqs_h).float()
    freqs_w = torch.outer(t_w, freqs_w).float()
    freqs_t = torch.outer(t_t, freqs_t).float()
    freqs_3d = torch.einsum('h,w,t->hwt', [freqs_h, freqs_w, freqs_t])
    freqs_cis_3d = torch.polar(torch.ones_like(freqs_3d), freqs_3d)

    return freqs_cis_3d


class Encoder1d(nn.Module):
    def __init__(self, resize=None, patch_size=128, embed_dim=2048, model_path=None):
        super(Encoder1d, self).__init__()
        self.resize = resize
        self.patchfier1d = nn.Conv1d(2, embed_dim, patch_size, patch_size)
        self.conv1d = nn.Conv1d(embed_dim, embed_dim, 3, 1, 1)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        if x.shape[-3] < 2:
            x = torch.concatenate([x, x], -2)
        if self.resize is not None:
            x = audio_F.resample(x, x.shape[-1], self.resize, lowpass_filter_width=6)
        x = self.patchfier1d(x)
        residual = x
        x = self.conv1d(x)
        x = x + residual
        return x.permute(0,2,1)
    
class Decoder1d(nn.Module):
    def __init__(self, resize=None, patch_size=128, embed_dim=2048, model_path=None):
        super(Decoder1d, self).__init__()
        self.resize = resize
        self.conv1d = nn.ConvTranspose1d(embed_dim, embed_dim, 3, 1, 1)
        self.patchfier1d = nn.ConvTranspose1d(embed_dim, 2, patch_size, patch_size)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        residual = x
        x = self.conv1d(x)
        x = x + residual
        x = self.patchfier1d(x)
        if self.resize is not None:
            x = audio_F.resample(x, x.shape[-1], self.resize, lowpass_filter_width=6)
        return x
    

class Encoder2d(nn.Module):
    def __init__(self, resize=None, patch_size=(16,16), embed_dim=2048, model_path=None):
        super(Encoder2d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.patchfier2d = nn.Conv2d(4, embed_dim, patch_size, patch_size)
        self.conv2d = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        freqs_cis = precompute_freqs_cis_2d(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)
        if x.shape[-3] < 4:
            mean_x = torch.mean(x, -3).unsqueeze(-3).repeat(1,4-x.shape[-3],1,1)
            x = torch.concatenate([x, mean_x], -3)
        if self.resize is not None:
            x = torchvision.transforms.Resize(self.resize)(x)
        x = self.patchfier2d(x)
        residual = x
        x = self.conv2d(x)
        x = x + residual
        self.feature_shape = x.shape
        x = x.reshape(1, self.embed_dim, -1)
        return x.permute(0,2,1)
    
class Decoder2d(nn.Module):
    def __init__(self, resize=None, patch_size=(16,16), embed_dim=2048, model_path=None):
        super(Decoder2d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.conv2d = nn.ConvTranspose2d(embed_dim, embed_dim, 3, 1, 1)
        self.patchfier2d = nn.ConvTranspose2d(embed_dim, 4, patch_size, patch_size)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        if self.feature_shape is not None:
            x = x.reshape(self.feature_shape)
        residual = x
        x = self.conv2d(x)
        x = x + residual
        x = self.patchfier2d(x)
        if self.resize is not None:
            x = torchvision.transforms.Resize(self.resize)(x)
        return x
    

class Encoder3d(nn.Module):
    def __init__(self, resize=None, patch_size=(6,6,6), embed_dim=2048, model_path=None):
        super(Encoder3d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.patchfier3d = nn.Conv3d(4, embed_dim, patch_size, patch_size)
        self.conv3d = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)

    def forward(self, x):
        if len(x.shape) != 5:
            x = x.unsqueeze(0)
        if x.shape[-4] < 4:
            mean_x = torch.mean(x, -4).unsqueeze(-4).repeat(1,4-x.shape[-4],1,1,1)
            x = torch.concatenate([x, mean_x], -4)
        # if self.resize is not None:
        #     x = torchvision.transforms.Resize(self.resize)(x)
        x = self.patchfier3d(x)
        residual = x
        x = self.conv3d(x)
        x = x + residual
        self.feature_shape = x.shape
        x = x.reshape(1, self.embed_dim, -1)
        return x.permute(0,2,1)
    
class Decoder3d(nn.Module):
    def __init__(self, resize=None, patch_size=(6,6,6), embed_dim=2048, model_path=None):
        super(Decoder3d, self).__init__()
        self.resize = resize
        self.embed_dim = embed_dim
        self.feature_shape = None
        self.conv3d = nn.ConvTranspose3d(embed_dim, embed_dim, 3, 1, 1)
        self.patchfier3d = nn.ConvTranspose3d(embed_dim, 4, patch_size, patch_size)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path), False)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        if self.feature_shape is not None:
            x = x.reshape(self.feature_shape)
        residual = x
        x = self.conv3d(x)
        x = x + residual
        x = self.patchfier3d(x)
        # if self.resize is not None:
        #     x = torchvision.transforms.Resize(self.resize)(x)
        return x
    
if __name__ == '__main__':
    print("Start testing...")
    
    ## audio
    # test_audio = torch.rand(1, 1, 5200)
    # encoder = Encoder1d(resize=520)
    # decoder = Decoder1d(resize=5200)
    # print(decoder(encoder(test_audio)).shape)

    ## image
    # test_image = torch.rand(1, 3, 800, 600)
    # encoder = Encoder2d(resize=(400,300))
    # decoder = Decoder2d(resize=(400,300))
    # feature = encoder(test_image)
    # decoder.feature_shape = encoder.feature_shape
    # print(decoder(feature).shape)

    # video
    test_video = torch.rand(1, 3, 800, 600, 6)
    encoder = Encoder3d(resize=(400,300))
    decoder = Decoder3d(resize=(400,300))
    feature = encoder(test_video)
    decoder.feature_shape = encoder.feature_shape
    print(decoder(feature).shape)