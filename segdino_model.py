import math

import torch
import torch.nn as nn
from utils import trunc_normal_
from vision_transformer  import Block, Mlp
from torch.nn.functional import upsample_nearest
from torch.nn import functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5 # self.scale=0.125
        head_dim = dim
        self.scale = head_dim ** -0.5 # self.scale=0.125

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x, t):
        B, N, C = x.shape
        B, TN, TC = t.shape

        #  using target as query
        q = self.q(t)
        k = self.k(x)
        v = self.v(x)
        # print("k.size():", k.size())
        # print("q.size():", q.size())
        attn = (q @ k.transpose(-2,-1)) * self.scale
        # print("attn.size():", attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, TN, TC)
        # x = (attn @ v)
        # x = self.proj(x).reshape(B, 64, 64)
        x = self.proj(x)
        # print("x.size():",x.size())

        #  #using anchor as query
        # q = self.q(x)
        # k = self.k(t)
        # v = self.v(t)
        # # print("k.size():", k.size()) #[B, 204, 512]
        # # print("q.size():", q.size())  #[B, 196, 512]
        # # print(k.transpose(-2,-1).size())
        # attn = (q @ k.transpose(-2,-1)) * self.scale
        # # print("attn.size():", attn.size()) #[B, 196, 204]
        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)

        return x

class CrossAttention_cropy(nn.Module):
    """
    https://github.com/IBM/CrossViT/blob/main/models/crossvit.py
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:36, ...]).reshape(B, 36, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        # print('q.size():',q.size())
        # print('k.size():',k.size())
        # print('v.size():',v.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)# print('attn.size():',attn.size())
        x = (attn @ v).transpose(1, 2).reshape(B, 36, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # print('x.size():',x.size())
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('x.size():',x.size())
        return x

class ConvHead(nn.Module):
    def __init__(self, num_inputs=256, dim_reduced=128, num_classes=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0),
            nn.BatchNorm2d(dim_reduced),nn.GELU(),
            nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0),
            nn.BatchNorm2d(dim_reduced),nn.GELU(),
            nn.ConvTranspose2d(dim_reduced, int(dim_reduced/2), 2, 2, 0),
            nn.BatchNorm2d(int(dim_reduced/2)),nn.GELU(),
            nn.ConvTranspose2d(int(dim_reduced/2), int(dim_reduced/4), 2, 2, 0),
            nn.BatchNorm2d(int(dim_reduced/4)),nn.GELU(),
            nn.ConvTranspose2d(int(dim_reduced/4), int(dim_reduced/8), 2, 2, 0),
            nn.BatchNorm2d(int(dim_reduced/8)),nn.GELU(),
        )
        self.mask_fcn_logits = nn.Conv2d(int(dim_reduced/8), num_classes, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # torch.Size([256, 128, 7, 7])
        # torch.Size([256, 64, 14, 14])
        # torch.Size([256, 64, 28, 28])
        # torch.Size([256, 64, 56, 56])
        # x = self.conv5_mask(x)
        # x = F.relu(x)
        # x = self.conv5_mask2(x)
        # x = F.relu(x)
        # x = self.conv5_mask3(x)
        # x = F.relu(x)
        # x = self.conv5_mask4(x)
        # x = F.relu(x)
        # x = self.conv5_mask5(x)
        # x = F.relu(x)
        x = self.conv_block(x)
        x = self.mask_fcn_logits(x)
        x = self.sigmoid(x)
        return x

class Neck(nn.Module):
    def __init__(self,input_dim=384, decoder_emd_dim=256, mask_ratio=0.8, act_layer=nn.GELU):
        super().__init__()
        self.input_dim = input_dim
        self.decoder_emd_dim = decoder_emd_dim
        self.mask_ratio = mask_ratio
        self.pos_embed_t = nn.Parameter(torch.zeros(1, 196, input_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, 36, input_dim))


        # for patch-based feature loss
        self.mlp_x = Mlp(in_features=self.input_dim, hidden_features=self.decoder_emd_dim, act_layer=nn.GELU, drop=0.1)
        # self.mlp_t = Mlp(in_features=self.input_dim, hidden_features=self.decoder_emd_dim, act_layer=nn.GELU, drop=0.)

        self.x_fc = nn.Linear(self.input_dim, self.decoder_emd_dim, bias=False)
        self.t_fc = nn.Linear(self.input_dim, self.decoder_emd_dim, bias=False)
        # self.norm_x = nn.LayerNorm(self.decoder_emd_dim)
        # self.norm_t = nn.LayerNorm(self.decoder_emd_dim)
        self.norm_x = nn.BatchNorm1d(36)
        self.norm_t = nn.BatchNorm1d(196)
        self.x_relu = act_layer()
        self.t_relu = act_layer()

        # self.cross_attention = CrossAttention(dim=self.decoder_emd_dim, out_dim=self.decoder_emd_dim,
        #                                       num_heads=8, qkv_bias=False, qk_scale=None)# B x L*(1-mask_ratio) x 256

        # self.cross_attention = CrossAttention_cropy(dim=self.decoder_emd_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.cross_attention = nn.MultiheadAttention(embed_dim = self.decoder_emd_dim, num_heads = 8, dropout = 0.1, batch_first=True)
        # B x L*(1-mask_ratio) x 256
        # self.norm_cross = nn.LayerNorm(self.cross_atte_out_dim)
        # self.norm_cross = nn.BatchNorm1d(49) #[B, 49, 128]
        self.norm_cross = nn.BatchNorm1d(196) #[4, 196, 256]
        self.relu_cross = act_layer()

        ## ConvHead
        self.convhead = ConvHead(num_inputs=self.decoder_emd_dim, dim_reduced=64, num_classes=1)
        self.upSampling_conv =  nn.Upsample(size=(224, 224), mode='bilinear') # predict a Bx64x64 result

        # ## change for full conv head
        self.fc_head = nn.Linear(self.decoder_emd_dim,1, bias=False) # # B x L*(1-mask_ratio) X 1
        # self.upSampling =  nn.Upsample(scale_factor=(4), mode='nearest')#
        self.upSampling =  nn.Upsample(size=(64, 64), mode='nearest') # predict a Bx64x64 resul

        self._initialize_weights()

    def forward(self, t, x):
        # t_ = self.mlp_t(t) # B, 196, 512/B, 36, 512
        t_ = t # teacher feature don't via projector

        t = t + self.interpolate_pos_encoding(t, 14, 14, self.pos_embed_t)
        t = self.t_fc(t)
        t = self.norm_t(t)
        t = self.t_relu(t)
        # t, mask, ids_restore = self.random_masking(t, self.mask_ratio)

        _x_list = [] # for feature msn loss
        x_list = [] # for BCEloss
        for idx in range(len(x)):
            # print('idx',idx)
            # _x_list.append(self.mlp_x(x[idx]))# B, 1024, 512/B, 196, 512
            # y = self.x_fc(x[idx])
            x_add_pos =x[idx] + self.interpolate_pos_encoding(x[idx], 6, 6, self.pos_embed_x)
            _x_list.append(self.mlp_x(x_add_pos))# B, 1024, 512/B, 196, 512
            y = self.x_fc(x_add_pos)
            y = self.norm_x(y)
            y = self.x_relu(y)

            # s = torch.cat([y,t],dim=1)
            # # print("y_shape:", y.size()) #[4, 36, 256]
            # # print("t_shape:", t.size()) #[4, 196, 256]
            # # print("s.size():", s.size()) #[4, 232, 256]
            # y = self.cross_attention(s)

            y, attn_output_weights = self.cross_attention(t, y, y)

            # y = self.cross_attention(y, t)
            # print("y_shape:", y.size()) #[256, 49, 128]
            y = self.norm_cross(y)
            y = self.relu_cross(y)

            # y = self.fc_head(y)
            B, N, C = y.shape # when using masking and mask_ratio=0.8 ## B,1024*0.2,1 = B,204,1
            feature_h = feature_w = math.sqrt(N) # masking 剩下的特征大小要能开平方得到整数
            y = y.transpose(1,2).reshape(B,C,int(feature_h),int(feature_w))
            y = self.convhead(y)
            y = self.upSampling_conv(y)

            # # x= self.upSampling(x.reshape(B, 1, 17, 12)) # [B, 204, 1]=>B,1,17,12=>B,1,64,64 掩模的比例不同，剩下的特征数不一样，需要reshape后才能上采样输出特征图
            # # y= self.upSampling(y.reshape(B, 1, 7, 7)) # [B, 204, 1]=>B,1,7,7=>B,1,64,64 掩模的比例不同，剩下的特征数不一样，需要reshape后才能上采样输出特征图
            # y= self.upSampling(y.reshape(B, 1, int(feature_h), int(feature_w))) # [B, 204, 1]=>B,1,7,7=>B,1,64,64 掩模的比例不同，剩下的特征数不一样，需要reshape后才能上采样输出特征图
            # # x = nn.functional.interpolate(x.reshape(B, 1, 17, 12), size=[64, 64], mode="nearest")
            # # print("y.size():",y.size())
            y = y.reshape(B, -1)
            # # print("y.size():",y.size())
            x_list.append(y)

        return x_list, _x_list, t_

    def interpolate_pos_encoding(self, x, w, h, patch_pos_embed):
        B, N, dim = x.shape
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return  patch_pos_embed

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class SegDINO(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, student, teacher, neck):
        super(SegDINO, self).__init__()
        student.fc, student.head = nn.Identity(), nn.Identity()
        teacher.fc, teacher.head = nn.Identity(), nn.Identity()
        self.student = student
        self.teacher = teacher
        self.neck = neck

    def forward(self,t, x):
        # print(len(x))
        # convert to list
        if not isinstance(x, list):
            x = [x]
        # print(torch.tensor([inp.shape[-1] for inp in x]))

        teacher_output = self.teacher(t)  # only the 2 global views pass through the teacher
        student_output = []
        for idx in range(len(x)):
            student_output.append(self.student(x[idx]))

        return self.neck(teacher_output, student_output)
