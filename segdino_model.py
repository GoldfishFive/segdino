import torch
import torch.nn as nn
from utils import trunc_normal_
from vision_transformer  import Block
from torch.nn.functional import upsample_nearest

class SegDINO(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone):
        super(SegDINO, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        # self.neck = neck
        # self.head = head

    def forward(self, x):
        # print(len(x))
        # convert to list
        if not isinstance(x, list):
            x = [x]
        # print(torch.tensor([inp.shape[-1] for inp in x]))

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        #batch中每个图像大小都一样
        # print("idx_crops:",idx_crops)#idx_crops: tensor([1])

        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # print(_out.size())
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        # return self.head(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x, t):
        B, N, C = x.shape
        B, TN, TC = t.shape

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q(t)
        k = self.k(x)
        v = self.v(x)
        # print("k.size():", k.size())
        # print(k.transpose(-2,-1).size())
        attn = (q @ k.transpose(-2,-1)) * self.scale
        # print("attn.size():", attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, TN, TC)
        # x = self.proj(x).reshape(B, 64, 64)
        x = self.proj(x)
        # print("x.size():",x.size())
        return x


class Neck(nn.Module):
    def __init__(self): # 补上masking操作后输出到head
        super().__init__()
        self.input_dim = 384
        self.decoder_emd_dim = 512
        self.out_dim = 256

        self.x_fc = nn.Linear(self.input_dim,self.decoder_emd_dim)
        self.t_fc = nn.Linear(self.input_dim,self.decoder_emd_dim)
        self.x_relu = nn.ReLU()
        self.t_relu = nn.ReLU()

        self.cross_attention = CrossAttention(dim=self.decoder_emd_dim, out_dim=self.out_dim, num_heads=8, qkv_bias=False, qk_scale=None)
        # BxLX256
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(self.out_dim,1) # BxLX1
        # self.upSampling =  nn.Upsample(scale_factor=(4), mode='nearest')#
        self.upSampling =  nn.Upsample(size=(64,64), mode='nearest')#

        self._initialize_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, t):
        x = self.x_fc(x)
        t = self.t_fc(t)
        x = self.x_relu(x)
        t = self.t_relu(t)

        t, mask, ids_restore = self.random_masking(t,0.8)
        # print(t.size())

        # x = torch.unsqueeze(x,2)
        # t = torch.unsqueeze(t,2)
        # print(x,x.size())
        # print(t,t.size())
        x= self.cross_attention(x,t)
        x = self.relu2(x)
        x = self.fc(x)
        B, N, C = x.shape # B,1024,1

        # x = torch.unsqueeze(x,1)
        # print("x.size():",x.size())
        # x = torch.squeeze(x,2)
        # print("x.size():",x.size())
        # x= self.upSampling(x.reshape(B, 32,32)) # B,1024,1
        x= self.upSampling(x.reshape(B, 1, 17, 12)) # [2, 204, 1]=>B,1,17,12=>B,1,64,64
        # x = nn.functional.interpolate(x.reshape(B, 1, 17, 12), size=[64, 64], mode="nearest")
        # print("x.size():",x.size())

        x =  x.reshape(B,-1)
        # x = torch.squeeze(x,2)
        # print("x.size():",x.size())


        # print("upSampling(x).size():",x.size())
        # print("pre_x.size():",x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

