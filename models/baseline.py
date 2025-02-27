import logging
import sys
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from timm.layers import AttentionPoolLatent, RmsNorm, SwiGLU
from timm.models.vision_transformer import global_pool_nlc

from models.encoder import Encoder
from models.patch_embeddings import ConvPatchEmbedding, PatchEmbedding, PosEmbedding

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIGS = {
    'baseline-tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4,
    },
    'baseline-small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
    },
    'baseline-base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'baseline-large': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'baseline-huge': {
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'baseline-giant': {
        'embed_dim': 1408,
        'depth': 40,
        'num_heads': 16,
        'mlp_ratio': 48/11,
    },
    'baseline-gigantic': {
        'embed_dim': 1664,
        'depth': 48,
        'num_heads': 16,
        'mlp_ratio': 64/13,
    }
}


class Baseline(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            'baseline', # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'baseline-tiny',
            'baseline-small',
            'baseline-base',
            'baseline-large',
            'baseline-huge',
            'baseline-giant',
            'baseline-gigantic'
        ] = 'baseline',
        input_shape=(1, 6, 64, 64, 1),
        modes=15,
        lateral_patch_size=16,
        axial_patch_size=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        fixed_dropout_depth=False,
        global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'avgmax',
        norm_layer: nn.Module = partial(RmsNorm, eps=1e-5),
        act_layer: nn.Module = nn.SiLU,
        mlp_layer: nn.Module = SwiGLU,
        use_conv_proj=False,
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config['depth']
            self.embed_dim = config['embed_dim']
            self.num_heads = config['num_heads']
            self.mlp_ratio = config['mlp_ratio']
        else:
            self.depth = depth
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.mlp_ratio = mlp_ratio

        self.input_shape = input_shape
        self.img_size = input_shape[-2]
        self.in_chans = input_shape[-1]
        self.num_frames = input_shape[1]

        self.axial_patch_size = axial_patch_size
        self.lateral_patch_size = lateral_patch_size

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate
        self.fixed_dropout_depth = fixed_dropout_depth

        self.init_std = init_std
        self.global_pool = global_pool
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.mlp_layer = mlp_layer
        self.norm = norm_layer(self.embed_dim) if norm_layer is not None else nn.Identity()

        if use_conv_proj:
            self.patch_embedding = ConvPatchEmbedding(
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                embed_dim=self.embed_dim,
            )
        else:
            self.patch_embedding = PatchEmbedding(
                input_fmt="BZYXC",
                input_shape=self.input_shape,
                lateral_patch_size=self.lateral_patch_size,
                axial_patch_size=self.axial_patch_size,
                embed_dim=self.embed_dim,
                channels=self.in_chans,
            )

        self.pos_embedding = PosEmbedding(
            input_fmt="BZYXC",
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            embed_dim=self.embed_dim,
            channels=self.in_chans,
            cls_token=False
        )

        self.encoder = Encoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            proj_drop_rate=self.proj_drop_rate,
            att_drop_rate=self.att_drop_rate,
            drop_path_rate=self.drop_path_rate,
            fixed_dropout_depth=self.fixed_dropout_depth,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
            mlp_layer=self.mlp_layer,
            init_std=self.init_std
        )

        self.global_pool = global_pool

        if global_pool == 'map':
            self.att_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.att_pool = None

        self.head = nn.Linear(self.embed_dim, modes) if modes > 0 else nn.Identity()
        self.head_drop = nn.Dropout(self.proj_drop_rate)

    @torch.jit.ignore
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def get_encoder(self):
        return self.encoder

    @torch.jit.ignore
    def get_num_patches(self):
        return self.pos_embedding.num_patches

    def pool(self, x, pool_type = None, num_prefix_tokens = 1):
        if self.att_pool is not None:
            x = self.att_pool(x)
            return x

        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=num_prefix_tokens)
        return x

    def forward_head(self, x):
        x = self.pool(x)
        x = self.norm(x)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, inputs):
        x = self.patch_embedding(inputs)
        x += self.pos_embedding(inputs)

        x = self.encoder(x)
        x = self.forward_head(x)
        return x
