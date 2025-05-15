import paddle.nn as nn
import paddle.nn.functional as F

import paddle as paddle


class ChannelAttention(nn.Layer):
    def __init__(self, in_channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1 = nn.Conv2D(in_channel, in_channel // ratio, kernel_size=1, bias_attr=True)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Conv2D(in_channel // ratio, in_channel, kernel_size=1, bias_attr=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.gelu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.gelu1(self.fc1(self.max_pool(x))))
        atten = self.sigmoid(avg_out + max_out)
        outputs = x * atten
        return outputs


'''
SAM(SpatialAttentionModule)
'''


class SpatialAttention(nn.Layer):
    def __init__(self, in_channel, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2D(2, in_channel, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


'''
Channel and Spatial Combined Attention (CBAM)
'''


# Construction
class CBAM(nn.Layer):
    def __init__(self, in_channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelAttention = ChannelAttention(in_channel, ratio=ratio)
        self.spatialAttention = SpatialAttention(in_channel=in_channel, kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelAttention(x)
        x = x * self.spatialAttention(x)
        return x


# Initialization
trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


# An identity convolutional layer
class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


# Regularization
class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ConvNeXt Block
class Block(nn.Layer):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, epsilon=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(
                value=layer_scale_init_value)) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Layer Scale
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# LN
class LayerNorm(nn.Layer):

    def __init__(self, normalized_shape, epsilon=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = paddle.create_parameter(
            shape=[normalized_shape],
            dtype='float32',
            default_initializer=ones_)

        self.bias = paddle.create_parameter(
            shape=[normalized_shape],
            dtype='float32',
            default_initializer=zeros_)

        self.epsilon = epsilon
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.epsilon)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.epsilon)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Layer):

    def __init__(self, in_chans=3, num_classes=3, num_patch=8,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        dim_len = len(dims)

        self.num_patch = num_patch

        self.downsample_layers = nn.LayerList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2D(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], epsilon=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(dim_len - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], epsilon=1e-6, data_format="channels_first"),
                nn.Conv2D(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        '''********************************* Attention Mechanism *************************************'''
        self.attention_layers = nn.LayerList()
        for i in range(dim_len):
            attention_layer = CBAM(in_channel=dims[i])
            self.attention_layers.append(attention_layer)
        '''********************************************************************************'''
        self.stages = nn.LayerList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(dim_len):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.set_value(self.head.weight * head_init_scale)
        self.head.bias.set_value(self.head.bias * head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            zeros_(m.bias)

    def split_and_process(self, x, num_patches, layer_idx):
        """
        Split an image into num_patches (which must be a power of 2),
        process each patch, then reconstruct the image.

        Args:
            x: Input tensor of shape (B, C, H, W)
            num_patches: Total number of patches (must be a power of 2)
            layer_idx: Index of the layer to process patches with

        Returns:
            Processed and reconstructed tensor
        """
        # Validate num_patches is a power of 2
        if num_patches & (num_patches - 1) != 0:
            raise ValueError(f"Number of patches ({num_patches}) must be a power of 2")

        B, C, H, W = x.shape

        # Determine the grid dimensions (h_patches × w_patches)
        import math
        grid_dim = int(math.sqrt(num_patches))

        # If num_patches is not a perfect square, find the closest factors
        if grid_dim * grid_dim != num_patches:
            # Find the largest power of 2 that is <= sqrt(num_patches)
            h_patches = 1
            while h_patches * 2 <= math.sqrt(num_patches):
                h_patches *= 2

            # Calculate w_patches based on h_patches
            w_patches = num_patches // h_patches
        else:
            h_patches = grid_dim
            w_patches = grid_dim

        # Calculate patch dimensions
        patch_h, patch_w = H // h_patches, W // w_patches

        # Validate that the image can be evenly divided
        if H % h_patches != 0 or W % w_patches != 0:
            raise ValueError(
                f"Image dimensions ({H}×{W}) must be divisible by grid dimensions ({h_patches}×{w_patches})")

        # Split, process, and collect patches
        patches = []
        for i in range(h_patches):
            for j in range(w_patches):
                # Extract patch
                patch = x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]

                # Process patch
                patch = self.downsample_layers[layer_idx](patch)
                patch = self.stages[layer_idx](patch)
                patch = self.attention_layers[layer_idx](patch)

                # Store processed patch
                patches.append(patch)

        # Ensure all patches have the same shape
        first_patch_shape = patches[0].shape
        for i, patch in enumerate(patches):
            if patch.shape != first_patch_shape:
                raise ValueError(f"Patch {i} has shape {patch.shape}, which differs from {first_patch_shape}")

        # Reshape the list to match the original grid structure
        patch_grid = []
        for i in range(h_patches):
            row = []
            for j in range(w_patches):
                row.append(patches[i * w_patches + j])
            patch_grid.append(row)

        # Concatenate patches horizontally within each row
        rows = []
        for i, row in enumerate(patch_grid):
            try:
                concatenated_row = paddle.concat(row, axis=3)
                rows.append(concatenated_row)
            except Exception as e:
                print(f"Error concatenating row {i}: {e}")
                print(f"Shapes in this row: {[p.shape for p in row]}")
                raise

        # Concatenate rows vertically
        try:
            x = paddle.concat(rows, axis=2)
        except Exception as e:
            print(f"Error in vertical concatenation: {e}")
            print(f"Row shapes: {[r.shape for r in rows]}")
            raise

        # Clean up to save memory
        del patches
        del patch_grid
        del rows
        return x

    def forward_features(self, x):
        patch_numbers = [self.num_patch]

        patch_len = 0
        if patch_numbers[0] > 0:
            for i in range(0, len(patch_numbers)):
                x = self.split_and_process(x, patch_numbers[i], i)
            patch_len = len(patch_numbers)

        for i in range(patch_len, len(self.downsample_layers)):
            # Final layer (no patching needed)
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.attention_layers[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x