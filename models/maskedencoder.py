import logging
import sys
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from timm.layers import RmsNorm, SwiGLU

from models.encoder import Encoder
from models.patch_embeddings import ConvPatchEmbedding, PatchEmbedding, PosEmbedding
from training.masking import apply_masks

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONFIGS = {
    'me-tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4,
    },
    'me-small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
    },
    'me-base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'me-large': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'me-huge': {
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'me-giant': {
        'embed_dim': 1408,
        'depth': 40,
        'num_heads': 16,
        'mlp_ratio': 48/11,
    },
    'me-gigantic': {
        'embed_dim': 1664,
        'depth': 48,
        'num_heads': 16,
        'mlp_ratio': 64/13,
    }
}


class MaskedEncoder(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            'me', # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'me-tiny',
            'me-small',
            'me-base',
            'me-large',
            'me-huge',
            'me-giant',
            'me-gigantic'
        ] = 'me',
        input_shape=(1, 6, 64, 64, 1),
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

    @torch.jit.ignore
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def get_encoder(self):
        return self.encoder

    @torch.jit.ignore
    def get_num_patches(self):
        return self.pos_embedding.num_patches

    def forward(self, inputs, masks=None, concat_masks=True):
        x, patches = self.patch_embedding(inputs, return_patches=True)
        x += self.pos_embedding(inputs)

        if masks is not None:
            x = apply_masks(x, masks, concat=concat_masks)

        x = self.encoder(x)
        x = self.norm(x)
        return x, patches
