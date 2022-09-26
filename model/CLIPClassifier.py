"""
#### Code adapted from the source code of CLIP model paper at https://github.com/openai/CLIP
"""   

from model.func_train import AverageMeter
from model.argument import set_seed
import torch.nn.functional as F
import numpy as np
import torch 
from torch import nn
from typing import Tuple, Union
from model.CLIP_model import ModifiedResNet,Transformer,VisionTransformer,LayerNorm,QuickGELU
import pdb 
import datetime
from collections import OrderedDict

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class CLIPClassifier(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 ):
        super().__init__()


        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        

        self.MLP = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("l2", nn.Linear(embed_dim * 4, 1))
        ]))

        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.MLP.l1.weight, std=0.01)
        nn.init.normal_(self.MLP.l2.weight, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    
    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    

    def forward(self, image):
        image_features = self.encode_image(image)
        IdCprob =  torch.sigmoid(self.MLP(image_features)).reshape(-1)
        return IdCprob



#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for name, p in model.named_parameters():
        if p.requires_grad == True: 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float()

def train_func(train_loader, accum_iter, model,optimizer, epoch, device,targetType):
    criterion = nn.BCELoss().to(device) 
    batch_time = AverageMeter()  # forward prop. + back prop. time
    loss_meter = AverageMeter()
    model.train()
    
    for i, batch in enumerate(train_loader):
        imgs = batch['image'].to(device)
        prob_targets = batch['prob'].to(device)
        if targetType == 1:
            prob_targets[prob_targets>0] = 1
        b_size = len(imgs)
        preds = model(imgs)
        loss = criterion(preds,prob_targets.to(preds.dtype))
        loss_meter.update(loss.item(), b_size)

        # backward pass
        loss.backward()
        
        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            if device == "cpu":
                optimizer.step()
            else : 
                #convert_models_to_fp32(model)
                optimizer.step()
                #convert_weights(model)

            model.zero_grad()
        
        if ((i + 1) % 50 == 0) or (i + 1 == len(train_loader)):
            print(str(datetime.datetime.now()),'Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                   loss=loss_meter))

    return loss_meter.avg
 
@torch.no_grad()
def val_func(val_loader, model, device,targetType):
    loss_meter = AverageMeter()  
    model.eval()

    for i, batch in enumerate(val_loader):
        imgs = batch['image'].to(device)
        prob_targets = batch['prob'].to(device)
        if targetType == 1:
            prob_targets[prob_targets>0] = 1
        b_size = len(imgs)
        preds = model(imgs)
        loss =  F.binary_cross_entropy(preds,prob_targets.to(preds.dtype))
        loss_meter.update(loss.item(), b_size)


    print('####Validation: loss: {loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss_meter))
    return loss_meter.avg



