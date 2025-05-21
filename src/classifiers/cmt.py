import math
import logging
from functools import partial
from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay


# PaddlePaddle equivalent of timm functions
def to_2tuple(x):
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1.):
    paddle.nn.initializer.TruncatedNormal(mean=mean, std=std)(tensor)


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)
        output = x.divide(keep_prob) * random_tensor
        return output


class SwishImplementation(nn.Layer):
    def forward(self, x):
        return x * F.sigmoid(x)


class MemoryEfficientSwish(nn.Layer):
    def __init__(self):
        super().__init__()
        self.swish = SwishImplementation()

    def forward(self, x):
        return self.swish(x)


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.conv1 = nn.Sequential(
            nn.Conv2D(in_features, hidden_features, 1, 1, 0, bias_attr=True),
            nn.GELU(),
            nn.BatchNorm2D(hidden_features, epsilon=1e-5),
        )
        self.proj = nn.Conv2D(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias_attr=True)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2D(hidden_features, epsilon=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2D(hidden_features, out_features, 1, 1, 0, bias_attr=True),
            nn.BatchNorm2D(out_features, epsilon=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., qk_ratio=1,
                 sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias_attr=True),
                nn.BatchNorm2D(dim, epsilon=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.num_heads, self.qk_dim // self.num_heads]).transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(x_).reshape([B, C, -1]).transpose([0, 2, 1])
            k = self.k(x_).reshape([B, -1, self.num_heads, self.qk_dim // self.num_heads]).transpose([0, 2, 1, 3])
            v = self.v(x_).reshape([B, -1, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])
        else:
            k = self.k(x).reshape([B, N, self.num_heads, self.qk_dim // self.num_heads]).transpose([0, 2, 1, 3])
            v = self.v(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale + relative_pos
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2D(dim, dim, 3, 1, 1, groups=dim, bias_attr=True)

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        cnn_feat = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose([0, 2, 1])
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Layer):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class CMT(nn.Layer):
    def __init__(self, img_size=1024, in_chans=3, num_classes=1000, embed_dims=[46, 92, 184, 368], stem_channel=16,
                 fc_dim=1280,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, qk_scale=None,
                 representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 depths=[2, 2, 10, 2], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)

        self.stem_conv1 = nn.Conv2D(in_chans, stem_channel, kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2D(stem_channel, epsilon=1e-5)

        self.stem_conv2 = nn.Conv2D(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2D(stem_channel, epsilon=1e-5)

        self.stem_conv3 = nn.Conv2D(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias_attr=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2D(stem_channel, epsilon=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = self.create_parameter(
            [num_heads[0], self.patch_embed_a.num_patches,
             self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]],
            default_initializer=nn.initializer.Normal())
        self.relative_pos_b = self.create_parameter(
            [num_heads[1], self.patch_embed_b.num_patches,
             self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]],
            default_initializer=nn.initializer.Normal())
        self.relative_pos_c = self.create_parameter(
            [num_heads[2], self.patch_embed_c.num_patches,
             self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]],
            default_initializer=nn.initializer.Normal())
        self.relative_pos_d = self.create_parameter(
            [num_heads[3], self.patch_embed_d.num_patches,
             self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]],
            default_initializer=nn.initializer.Normal())

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks_a = nn.LayerList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.LayerList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.LayerList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.LayerList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self._fc = nn.Conv2D(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2D(fc_dim, epsilon=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2D(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Conv2D):
            nn.initializer.KaimingNormal()(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.initializer.Constant(0)(m.bias)
            nn.initializer.Constant(1.0)(m.weight)
        elif isinstance(m, nn.BatchNorm2D):
            nn.initializer.Constant(0)(m.bias)
            nn.initializer.Constant(1.0)(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)

        for blk in self.blocks_a:
            x = blk(x, H, W, self.relative_pos_a)

        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        x, (H, W) = self.patch_embed_b(x)
        for blk in self.blocks_b:
            x = blk(x, H, W, self.relative_pos_b)

        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        x, (H, W) = self.patch_embed_c(x)
        for blk in self.blocks_c:
            x = blk(x, H, W, self.relative_pos_c)

        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        x, (H, W) = self.patch_embed_d(x)
        for blk in self.blocks_d:
            x = blk(x, H, W, self.relative_pos_d)

        B, N, C = x.shape
        x = self._fc(x.transpose([0, 2, 1]).reshape([B, C, H, W]))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(1)
        x = self._drop(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x