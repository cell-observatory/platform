import logging
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_masks(x, masks, concat=True):
    if isinstance(masks, list):
        output = []
        for m in masks:
            mask_keep = m.unsqueeze(-1)
            if x.dim() > 2:
                mask_keep = mask_keep.repeat(1, 1, x.size(-1))

            output += [torch.gather(x, dim=1, index=mask_keep)]
        if not concat:
            return output

        return torch.cat(output, dim=0)
    else:
        indices = masks.unsqueeze(-1)
        if x.dim() > 2:
            indices = indices.repeat(1, 1, x.size(-1))

        # print(f"{x.shape=}, {masks.shape=}, {indices.shape=}")
        return torch.gather(x, dim=1, index=indices)

def mask_random_patches(inputs, num_patches, ratio, window_mask_shape=None):
    batch_size = inputs.shape[0]

    if window_mask_shape is not None:
        num_windows = np.prod(window_mask_shape)
        context_length = int(num_windows * (1 - ratio))
        noise = torch.rand(batch_size, num_windows, device=inputs.device)
        masks = torch.ones([batch_size, num_windows], device=inputs.device)
    else:
        context_length = int(num_patches * (1 - ratio))
        noise = torch.rand(batch_size, num_patches, device=inputs.device)
        masks = torch.ones([batch_size, num_patches], device=inputs.device)

    shuffle = torch.argsort(noise, dim=1)
    original_patch_indices = torch.argsort(shuffle, dim=1)
    context_masks = shuffle[:, :context_length]
    target_masks = shuffle[:, context_length:]

    masks[:, :context_length] = 0 # mask out context patches
    masks = torch.gather(masks, dim=1, index=original_patch_indices) # reorder

    return masks, context_masks, target_masks, original_patch_indices


class MaskGenerator(object):

    def __init__(
        self,
        patchify_scheme='blocks',
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        lateral_range=(0.2, 0.8),
        axial_range=(.5, 1.0),
        num_blocks=8,

    ):
        super(MaskGenerator, self).__init__()
        self.patchify_scheme = patchify_scheme
        self.input_shape = input_shape
        self.lateral_patch_size = lateral_patch_size
        self.axial_patch_size = axial_patch_size

        self.depth = input_shape[1] // axial_patch_size
        self.height  = input_shape[2] // lateral_patch_size
        self.width = input_shape[3] // lateral_patch_size

        self.lateral_range = lateral_range
        self.axial_range = axial_range
        self.num_blocks = num_blocks

    def get_random_block(self, generator):
        if self.axial_range[0] < self.axial_range[1]:
            d = torch.randint(
                size=(1,),
                low=int(round(self.axial_range[0] * self.depth)),
                high=int(round(self.axial_range[1] * self.depth)),
                generator=generator,
                dtype=torch.int16
            ).item()
        else:
            d = self.depth

        if (self.lateral_range[0] < 1) and (self.lateral_range[1] < 1) and (self.lateral_range[0] < self.lateral_range[1]):
            h = torch.randint(
                size=(1,),
                low=int(round(self.lateral_range[0] * self.height)),
                high=int(round(self.lateral_range[1] * self.height)),
                generator=generator,
                dtype=torch.int16
            ).item()

            w = torch.randint(
                size=(1,),
                low=int(round(self.lateral_range[0] * self.width)),
                high=int(round(self.lateral_range[1] * self.width)),
                generator=generator,
                dtype=torch.int16
            ).item()
        else:
            raise Exception("Scale should be between 0 and 1")

        z = torch.randint(0, self.depth - d + 1, (1,))
        y = torch.randint(0, self.height - h + 1, (1,))
        x = torch.randint(0, self.width - w + 1, (1,))

        block = torch.ones((self.depth, self.height, self.width), dtype=torch.int16)
        block[z:z+d, y:y+h, x:x+w] = 0

        return block

    def mask_random_blocks(self, generator, plot_path=None):
        mask = torch.ones((self.depth, self.height, self.width), dtype=torch.int16)

        if plot_path is not None:
            fig, axes = plt.subplots(nrows=self.depth, ncols=self.num_blocks + 1)

        for b in range(self.num_blocks):
            context_ratio = 1 - (torch.count_nonzero(mask) / (self.height * self.width))

            if context_ratio < self.lateral_range[1]:
                m = self.get_random_block(generator=generator)
                mask *= m

                if plot_path is not None:
                    for i in range(self.depth):
                        axes[i, b].imshow(m[i])

        if plot_path is not None:
            for i in range(self.depth):
                axes[i, -1].imshow(mask[i])
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, pad_inches=.25)

        return mask

    def mask_random_tracks(self, generator, plot_path=None):

        lateral_mask_ratio = (self.lateral_range[1] - self.lateral_range[0]) * torch.rand(1, generator=generator) + self.lateral_range[0]
        if self.axial_range[0] < self.axial_range[1]:
            d = torch.randint(
                size=(1,),
                low=int(round(self.axial_range[0] * self.depth)),
                high=int(round(self.axial_range[1] * self.depth)),
                generator=generator,
                dtype=torch.int16
            ).item()
        else:
            d = self.depth

        tile = torch.ones((self.height, self.width), dtype=torch.int16)
        n = int(lateral_mask_ratio * self.height * self.width)
        idx = torch.randperm(self.height * self.width, generator=generator)[:n]
        tile.view(-1)[idx] = 0
        tube = torch.tile(tile, (d, 1, 1))
        z = torch.randint(0, self.depth - d + 1, (1,))

        mask = torch.ones((self.depth, self.height, self.width), dtype=torch.int16)
        mask[z:z+d] = tube

        if plot_path is not None:
            fig, axes = plt.subplots(nrows=self.depth)
            for i in range(self.depth):
                axes[i].imshow(mask[i])
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, pad_inches=.25)

        return mask

    def mask_random_patches(self, batch_size, generator):
        ratio = (self.lateral_range[1] - self.lateral_range[0]) * torch.rand(1, generator=generator) + self.lateral_range[0]
        num_patches = self.depth * self.height * self.width
        context_length = int(num_patches * (1 - ratio))
        mask = torch.argsort(torch.rand(batch_size, num_patches), dim=1)
        target_masks = torch.argsort(mask, dim=1)
        context_masks = mask[:, :context_length]
        return context_masks, target_masks

    def __call__(self, batch_size):
        gen = torch.Generator()
        context_bound = complement_bound = self.depth * self.height * self.width

        context_masks, target_masks = [], []
        for i in range(batch_size):
            if self.patchify_scheme == 'space_only':
                # outdir = Path("../masks/random_tracks")
                # outdir.mkdir(exist_ok=True, parents=True)
                complement = self.mask_random_tracks(
                    generator=gen,
                    # plot_path=f"{outdir}/mask_{i}.png"
                )
            elif self.patchify_scheme == 'blocks':
                # outdir = Path("../masks/random_blocks")
                # outdir.mkdir(exist_ok=True, parents=True)
                complement = self.mask_random_blocks(
                    generator=gen,
                    # plot_path=f"{outdir}/mask_{i}.png"
                )
            else:
                raise Exception("Patchify scheme not supported")

            complement = complement.flatten()
            mask = torch.argwhere(complement == 0).squeeze()
            complement = torch.nonzero(complement).squeeze()

            empty = len(complement) == 0
            if not empty:
                context_bound = min(context_bound, len(mask))
                complement_bound = min(complement_bound, len(complement))
                context_masks.append(mask)
                target_masks.append(complement)
            else:
                raise Exception("Mask is empty")

        context_masks = [cm[:context_bound] for cm in context_masks]
        target_masks = [cm[:complement_bound] for cm in target_masks]

        return context_masks, target_masks



class MaskCollator(object):

    def __init__(
        self,
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        lateral_range=(0.2, 0.8),
        axial_range=(.5, 1.0),
        num_blocks=8,
        patchify_scheme='blocks',
        collator_func=torch.utils.data.default_collate,
    ):
        super(MaskCollator, self).__init__()

        self.input_shape = input_shape
        self.lateral_patch_size = lateral_patch_size
        self.axial_patch_size = axial_patch_size
        self.lateral_range = lateral_range
        self.axial_range = axial_range
        self.num_blocks = num_blocks
        self.patchify_scheme = patchify_scheme
        self.collator_func = collator_func

        self.mask_generator = MaskGenerator(
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_range=self.lateral_range,
            axial_range=self.axial_range,
            num_blocks=self.num_blocks,
            patchify_scheme=self.patchify_scheme,
        )

    def __call__(self, batch):
        context_masks, target_masks = self.mask_generator(len(batch))
        return self.collator_func(batch), self.collator_func(context_masks), self.collator_func(target_masks)
