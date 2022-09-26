"""
#### Code adapted from the source code of ArtEmis dataset paper
"""
"""
Rensnet Wrapper.

The MIT License (MIT)
Originally created in late 2019, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""


import torch
from torch import nn
import pdb 
import timm


class vitEncoder(nn.Module):
    def __init__(self, backbone, adapt_image_size=None, drop=2, pretrained=True, verbose=False):
        if drop == 0 and adapt_image_size is not None:
            raise ValueError('Trying to apply adaptive pooling while keeping the entire model (drop=0).')

        super().__init__()

        self.name = backbone
        self.drop = drop

        self.vit = timm.create_model(backbone, pretrained=True)

        # Remove linear and last adaptive pool layer
        if drop > 0:
            modules = list(self.vit.children())
            if verbose:
                print('Removing the last {} layers of a {}'.format(drop, self.name))
                print(modules[-drop:])
            modules = modules[:-drop]
            self.vit = nn.Sequential(*modules)

        self.adaptive_pool = None
        if adapt_image_size is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((adapt_image_size, adapt_image_size))

        if pretrained:
            for p in self.vit.parameters():
                p.requires_grad = False

    def __call__(self, images):
        """Forward prop.
            :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
            :return: encoded images
        """
        out = self.vit(images) # (B, F, ceil(image_size/32), ceil(image_size/32))  #torch.Size([8, 512, 7, 7])


        if self.adaptive_pool is not None: #adaptive_pool = AdaptiveAvgPool2d(output_size=(224, 224))
            out = self.adaptive_pool(out)  # (B, F, adapt_image_size, adapt_image_size) torch.Size([8, 512, 224, 224])


        out = out.reshape(out.shape[0],-1, out.shape[-1]) 
        return out

    def unfreeze(self, level=5, verbose=False):
        """Allow or prevent the computation of gradients for blocks after level.
        The smaller the level, the less pretrained the vit will be.
        """
        all_layers = list(self.vit.children())

        if verbose:
            ll = len(all_layers)
            print('From {} layers, you are unfreezing the last {}'.format(ll, ll-level))

        for c in all_layers[level:]:
            for p in c.parameters():
                p.requires_grad = True
        return self

    def embedding_dimension(self):
        if 'vit' in self.name:
            return 768 
        raise NotImplementedError

