import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from .utils import *


class ConvLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 use_norm=True,
                 use_act=True,
                 act_type='relu',
                 neg_slope=0.1,
                 norm_type='batch_norm_2d',
                 ):
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        block = nn.Sequential()

        conv_layer = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias,
                               padding_mode=padding_mode)

        block.add_sublayer(name="conv", sublayer=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(num_features=out_channels, norm_type=norm_type)
            block.add_sublayer(name="norm", sublayer=norm_layer)

        self.act_name = None

        if act_type is not None and use_act:
            act_layer = get_activation_fn(act_type=act_type,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_sublayer(name="act", sublayer=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer._kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x):
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ', normalization={}'.format(self.norm_name)

        if self.act_name is not None:
            repr_str += ', activation={}'.format(self.act_name)
        repr_str += ', bias={})'.format(self.bias)
        return repr_str


class TransposeConvLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 use_norm=True,
                 use_act=True,
                 padding=(0, 0),
                 auto_padding=True,
                 neg_slope=0.1,
                 act_type='relu',
                 norm_type='batch_norm_2d',):
        """
        Applies a 2D Transpose Convolution over an input signal composed of several input planes.
        :param opts: over an input signal composed of several input planes.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: move the kernel by this amount during convolution operation
        :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
        :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
        :param bias: Add bias or not
        :param padding_mode: Padding mode. Default is zeros
        :param use_norm: Use normalization layer after convolution layer or not. Default is True.
        :param use_act: Use activation layer after convolution layer/convolution layer followed by batch normalization
                        or not. Default is True.
        :param padding: Padding
        :param auto_padding: Compute padding automatically
        """
        super(TransposeConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, (tuple, list)):
            dilation = dilation[0]

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, int)

        if auto_padding:
            padding = (int((kernel_size[0] - 1)) * dilation, int((kernel_size[1] - 1)) * dilation)

        block = nn.Sequential()
        conv_layer = nn.Conv2DTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias,
                                        padding_mode=padding_mode)

        block.add_sublayer(name="conv", sublayer=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(num_features=out_channels, norm_type=norm_type)
            block.add_sublayer(name="norm", sublayer=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None

        if act_type is not None and use_act:
            act_layer = get_activation_fn(act_type=act_type,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_sublayer(name="act", sublayer=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias

    def forward(self, x):
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ', normalization={}'.format(self.norm_name)

        if self.act_name is not None:
            repr_str += ', activation={}'.format(self.act_name)
        repr_str += ')'
        return repr_str



# class NormActLayer(nn.Layer):
#     def __init__(self, opts, num_features):
#         """
#         Applies a normalization layer followed by activation layer over an input tensor
#         :param opts: arguments
#         :param num_features: number of feature planes in the input tensor
#         """
#         super(NormActLayer, self).__init__()
#
#         block = nn.Sequential()
#
#         self.norm_name = None
#         norm_layer = get_normalization_layer(opts=opts, num_features=num_features)
#         block.add_sublayer(name="norm", module=norm_layer)
#         self.norm_name = norm_layer.__class__.__name__
#
#         self.act_name = None
#         act_type = getattr(opts, "model.activation.name", "prelu")
#         neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
#         inplace = getattr(opts, "model.activation.inplace", False)
#         act_layer = get_activation_fn(act_type=act_type,
#                                       inplace=inplace,
#                                       negative_slope=neg_slope,
#                                       num_parameters=num_features)
#         block.add_sublayer(name="act", module=act_layer)
#         self.act_name = act_layer.__class__.__name__
#
#         self.block = block
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.block(x)
#
#     def profile_module(self, input: Tensor) -> (Tensor, float, float):
#         # compute parameters
#         params = sum([p.numel() for p in self.parameters()])
#         macs = 0.0
#         return input, params, macs
#
#     def __repr__(self):
#         repr_str = '{}(normalization={}, activation={})'.format(self.__class__.__name__, self.norm_type, self.act_type)
#         return repr_str


class InvertedResidual(nn.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 act_type="relu",
                 ):
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_sublayer(name="exp_1x1",
                               sublayer=ConvLayer(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                                  use_act=True, use_norm=True, act_type=act_type))
        block.add_sublayer(
            name="conv_3x3",
            sublayer=ConvLayer(in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                               groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation, act_type=act_type)
        )

        block.add_sublayer(name="red_1x1",
                           sublayer=ConvLayer(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                              use_act=False, use_norm=True, act_type=act_type))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def __repr__(self) -> str:
        return '{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp, self.dilation
        )


class MultiHeadAttention(nn.Layer):
    '''
    This layer applies a multi-head attention as described in "Attention is all you need" paper
    https://arxiv.org/abs/1706.03762
    '''

    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_dropout=0.0,
                 bias=True):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param attn_dropout: Attention dropout
        :param bias: Bias
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias_attr=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias_attr=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        # [B x N x C]
        b_sz, n_patches, in_channels = x.shape

        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (
            self.qkv_proj(x)
                .reshape([b_sz, n_patches, 3, self.num_heads, -1])
        )
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose([0, 3, 2, 1, 4])

        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # QK^T
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = paddle.matmul(query, key, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = paddle.matmul(attn, value)

        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose([0, 2, 1, 3]).reshape([b_sz, n_patches, -1])
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Layer):
    """
        This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embed_dim,
                 ffn_latent_dim,
                 num_heads=8,
                 attn_dropout=0.0,
                 dropout=0.1,
                 ffn_dropout=0.0,
                 transformer_norm_layer="layer_norm"):
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias_attr=True),
            nn.Swish(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias_attr=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Layer):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(self,
                 in_channels,
                 transformer_dim,
                 ffn_dim,
                 n_transformer_blocks=2,
                 head_dim=32,
                 attn_dropout=0.1,
                 dropout=0.1,
                 ffn_dropout=0.1,
                 patch_h=8,
                 patch_w=8,
                 transformer_norm_layer="layer_norm",
                 conv_ksize=3,
                 dilation=1,
                 var_ffn=False,
                 no_fusion=False,
                 act_type='swish'):
        conv_3x3_in = ConvLayer(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation, act_type=act_type
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False, act_type=act_type
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True, act_type=act_type
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                in_channels=2 * in_channels, out_channels=in_channels,
                kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, act_type=act_type
            )
        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_sublayer(name="conv_3x3", sublayer=conv_3x3_in)
        self.local_rep.add_sublayer(name="conv_1x1", sublayer=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx],
                               num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout,
                               transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.var_ffn = var_ffn
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tconv_in_dim={}, conv_out_dim={}, dilation={}, conv_ksize={}".format(self.cnn_in_dim,
                                                                                             self.cnn_out_dim,
                                                                                             self.dilation,
                                                                                             self.conv_ksize)
        repr_str += "\n\tpatch_h={}, patch_w={}".format(self.patch_h, self.patch_w)
        repr_str += "\n\ttransformer_in_dim={}, transformer_n_heads={}, transformer_ffn_dim={}, dropout={}, " \
                    "ffn_dropout={}, attn_dropout={}, blocks={}".format(
            self.cnn_out_dim,
            self.n_heads,
            self.ffn_dim,
            self.dropout,
            self.ffn_dropout,
            self.attn_dropout,
            self.n_blocks
        )
        if self.var_ffn:
            repr_str += "\n\t var_ffn_min_mult={}, var_ffn_max_mult={}".format(
                self.ffn_min_dim, self.ffn_max_dim
            )

        repr_str += "\n)"
        return repr_str

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape([batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w])
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose([0, 2, 1, 3])
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape([batch_size, in_channels, num_patches, patch_area])
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose([0, 3, 2, 1])
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape([batch_size * patch_area, num_patches, -1])

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.reshape([info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1])

        batch_size, pixels, num_patches, channels = patches.shape
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose([0, 3, 2, 1])

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape([batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w])
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose([0, 2, 1, 3])
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            [batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w])
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x):
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(
                paddle.concat((res, fm), axis=1)
            )
        return fm
