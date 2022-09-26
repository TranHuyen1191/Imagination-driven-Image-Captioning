"""
#### Code adapted from the source code of CLIP model paper at https://github.com/openai/CLIP
"""   

from model.func_train_v2 import AverageMeter
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


class adapterCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

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

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.MLP_g_score = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("l2", nn.Linear(embed_dim * 4, 1))
        ]))
        self.MLP_ita1Text = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("l2", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.MLP_ita2Text = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("l2", nn.Linear(embed_dim * 4, embed_dim))
        ]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.MLP_g_score.l1.weight, std=0.01)
        nn.init.normal_(self.MLP_g_score.l2.weight, std=0.01)
        nn.init.normal_(self.MLP_ita1Text.l1.weight, std=0.01)
        nn.init.normal_(self.MLP_ita1Text.l2.weight, std=0.01)
        nn.init.normal_(self.MLP_ita2Text.l1.weight, std=0.01)
        nn.init.normal_(self.MLP_ita2Text.l2.weight, std=0.01)

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

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, no_tokens, emb_dim]
        x = x[:,:self.context_length,:]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # take features from the eot embedding (eot_token is the highest number in each sequence)
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features_T1 = self.MLP_ita1Text(self.encode_text(text))
        text_features_T2 = self.MLP_ita2Text(self.encode_text(text))
        text_features_T3 = self.MLP_g_score(self.encode_text(text))
        g_pen =  torch.sigmoid(text_features_T3).reshape(-1)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_T1 = text_features_T1 / text_features_T1.norm(dim=-1, keepdim=True)
        text_features_T2 = text_features_T2 / text_features_T2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        ita1_scores_per_image = (image_features @ text_features_T1.t()-(-1))/2 # normalize cos similarity to [0,1]
        ita1_scores_per_text = ita1_scores_per_image.t()

        # cosine similarity as logits
        ita2_scores_per_image = (image_features @ text_features_T2.t()-(-1))/2 # normalize cos similarity to [0,1]
        ita2_scores_per_text = ita2_scores_per_image.t()
        scores = (ita1_scores_per_image + ita2_scores_per_image)/2 - g_pen

        return ita1_scores_per_image, ita1_scores_per_text,ita2_scores_per_image, ita2_scores_per_text,g_pen,scores

    def get_score(self, image, text,alpha):
        image_features = self.encode_image(image)
        text_features_T1 = self.MLP_ita1Text(self.encode_text(text))
        text_features_T2 = self.MLP_ita2Text(self.encode_text(text))
        text_features_T3 = self.MLP_g_score(self.encode_text(text))
        g_pen =  torch.sigmoid(text_features_T3).reshape(-1)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_T1 = text_features_T1 / text_features_T1.norm(dim=-1, keepdim=True)
        text_features_T2 = text_features_T2 / text_features_T2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        ita1_scores_per_image = (image_features @ text_features_T1.t()-(-1))/2 # normalize cos similarity to [0,1]
        ita1_scores = torch.diagonal(ita1_scores_per_image)

        # cosine similarity as logits
        ita2_scores_per_image = (image_features @ text_features_T2.t()-(-1))/2 # normalize cos similarity to [0,1]
        ita2_scores = torch.diagonal(ita2_scores_per_image)
        scores = (ita1_scores + ita2_scores)/2 - alpha*g_pen

        return ita1_scores, ita2_scores,g_pen,scores

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for name, p in model.named_parameters():
        if p.requires_grad == True: 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float()

def train_adapterCLIP(train_loader, accum_iter, model,optimizer, epoch, device):
    criterion_1 = nn.CrossEntropyLoss().to(device) 
    criterion_2 = nn.CrossEntropyLoss().to(device) 
    criterion_3 = nn.CrossEntropyLoss().to(device) 
    criterion_4 = nn.CrossEntropyLoss().to(device) 
    criterion_5 = nn.BCELoss().to(device) 

    batch_time = AverageMeter()  # forward prop. + back prop. time
    total_loss_meter = AverageMeter()
    ent_loss_meter = AverageMeter()
    ent_img_loss_meter = AverageMeter()
    ent_text_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    model.train()
    
    for i, batch in enumerate(train_loader):
        imgs = batch['image'].to(device)
        capts = batch['captset'].to(device)
        b_size = len(imgs)
        noCapt_perImg = capts.size(1)
        
        
        # Consider as having Batchsize*noCapt_perImg captions
        capts_input = capts.reshape(-1,capts.size(-1))
        outputs = model(imgs,capts_input)
        ita1_scores_per_image, ita1_scores_per_text,ita2_scores_per_image, ita2_scores_per_text,g_pen,scores = outputs
        ## ita_scores_per_image: BSx(BS*noCapt_perImg) = (7,28) or (Image,Text) ## Each row corresponds to one image and its score with all texts
        ## ita_scores_per_text: BS*noCapt_perImg)xBS = (28,7) or (Text,Image) ## Each row corresponds to one text and its score with all images

        ## Image-text alignment cross entropy loss
        a1 = np.arange(0,noCapt_perImg*b_size,noCapt_perImg) ## position of natural captions
        b1 = np.arange(1,noCapt_perImg*b_size,noCapt_perImg) ## position of Error-Type-I captions 
        sel_idx = np.vstack((a1,b1)).T.reshape(-1) 
        ita1_scores_per_text = ita1_scores_per_text[sel_idx]  #[21,7]
        ita1_scores_per_image = ita1_scores_per_text.T #[7,21]
        
        noCapt_perImg_temp =  int(ita1_scores_per_image.size(1)/ ita1_scores_per_image.size(0))
        targets_per_image = torch.arange(0,noCapt_perImg_temp*b_size,noCapt_perImg_temp).to(torch.long).to(device)
        targets_per_text = torch.arange(0,b_size).to(torch.long).to(device)

        ent1_loss_img = criterion_1(ita1_scores_per_image, targets_per_image)
        ita1_scores_per_text =  ita1_scores_per_text[0:-1:noCapt_perImg_temp] #(BS*noCapt_perImg)xBS = (21,7) --> BSxBS = (7,7) Only using natural caption
        ent_loss_text = criterion_2(ita1_scores_per_text, targets_per_text)
        

        ## Image-text alignment cross entropy loss
        a1 = np.arange(0,noCapt_perImg*b_size,noCapt_perImg) ## position of natural captions
        b1 = np.arange(2,noCapt_perImg*b_size,noCapt_perImg) ## position of Error-Type-II captions 
        sel_idx = np.vstack((a1,b1)).T.reshape(-1) 
        ita2_scores_per_text = ita2_scores_per_text[sel_idx]  #[21,7]
        ita2_scores_per_image = ita2_scores_per_text.T #[7,21]
        ent2_loss_img = criterion_3(ita2_scores_per_image, targets_per_image)
        #ita2_scores_per_text =  ita2_scores_per_text[0:-1:noCapt_perImg_temp] #(BS*noCapt_perImg)xBS = (21,7) --> BSxBS = (7,7) Only using natural caption
        #ent2_loss_text = criterion_4(ita2_scores_per_text, targets_per_text) 
        
        targets_scores = torch.arange(0,noCapt_perImg*b_size,noCapt_perImg).to(torch.long).to(device)
        ent_loss = criterion_4(scores, targets_scores)
        
        ## Grammatical mistake Binary Cross Entropy loss
        g_targets = torch.Tensor([0,1]*b_size).to(device)
        a1 = np.arange(0,noCapt_perImg*b_size,noCapt_perImg) ## position of natural captions
        b1 = np.arange(noCapt_perImg-1,noCapt_perImg*b_size,noCapt_perImg) ## position of Error-Type-III captions 
        c1 = np.vstack((a1,b1))
        sel_idx = np.vstack((a1,b1)).T.reshape(-1)
        g_pen = g_pen[sel_idx] 
        bce_loss = criterion_5(g_pen,g_targets.to(g_pen.dtype))

        total_loss = ent1_loss_img +  ent2_loss_img +ent_loss_text+ ent_loss + bce_loss
        total_loss_meter.update(total_loss.item(), b_size)
        ent_loss_meter.update(ent_loss.item(), b_size)
        ent_img_loss_meter.update((ent1_loss_img + ent2_loss_img).item(), b_size)
        ent_text_loss_meter.update(ent_loss_text.item(), b_size)
        bce_loss_meter.update(bce_loss.item(), b_size)
        
        # backward pass
        total_loss.backward()
        
        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            if device == "cpu":
                optimizer.step()
            else : 
                #convert_models_to_fp32(model)
                optimizer.step()
                #convert_weights(model)
            model.zero_grad()
        
        if ((i + 1) % 100 == 0) or (i + 1 == len(train_loader)):
            print(str(datetime.datetime.now()),'Epoch: [{0}][{1}/{2}]\t'
                  'TotalLoss: {tloss.val:.4f} ({tloss.avg:.4f}), '
                  'CELoss: {entloss.val:.4f} ({entloss.avg:.4f}), '
                  'CE_I_Loss: {entIloss.val:.4f} ({entIloss.avg:.4f}), '
                  'CE_T_Loss: {entTloss.val:.4f} ({entTloss.avg:.4f}), '
                  'BCELoss: {bceloss.val:.4f} ({bceloss.avg:.4f})'.format(epoch, i, len(train_loader),
                   tloss=total_loss_meter,entloss=ent_loss_meter,entIloss=ent_img_loss_meter,
                   entTloss=ent_text_loss_meter,bceloss=bce_loss_meter))

    return total_loss_meter.avg
 
@torch.no_grad()
def val_adapterCLIP(val_loader, model, device):
    total_loss_meter = AverageMeter()  
    ent_loss_meter = AverageMeter()  
    ent_img_loss_meter = AverageMeter()
    ent_text_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()  
    model.eval()

    for i, batch in enumerate(val_loader):
        imgs = batch['image'].to(device)
        capts = batch['captset'].to(device)
        b_size = len(imgs)
        noCapt_perImg = capts.size(1)
        
        # Consider as having Batchsize*noCapt_perImg captions
        capts_input = capts.reshape(-1,capts.size(-1))
        outputs = model(imgs,capts_input)
        ita1_scores_per_image, ita1_scores_per_text,ita2_scores_per_image, ita2_scores_per_text,g_pen,scores = outputs
        ## ita_scores_per_image: BSx(BS*noCapt_perImg) = (7,28) or (Image,Text) ## Each row corresponds to one image and its score with all texts
        ## ita_scores_per_text: BS*noCapt_perImg)xBS = (28,7) or (Text,Image) ## Each row corresponds to one text and its score with all images

        ## Image-text alignment cross entropy loss
        a1 = np.arange(0,noCapt_perImg*b_size,noCapt_perImg) ## position of natural captions
        b1 = np.arange(1,noCapt_perImg*b_size,noCapt_perImg) ## position of Error-Type-I captions 
        sel_idx = np.vstack((a1,b1)).T.reshape(-1) 
        ita1_scores_per_text = ita1_scores_per_text[sel_idx]  #[21,7]
        ita1_scores_per_image = ita1_scores_per_text.T #[7,21]
        
        noCapt_perImg_temp =  int(ita1_scores_per_image.size(1)/ ita1_scores_per_image.size(0))
        targets_per_image = torch.arange(0,noCapt_perImg_temp*b_size,noCapt_perImg_temp).to(torch.long).to(device)
        targets_per_text = torch.arange(0,b_size).to(torch.long).to(device)

        ent1_loss_img = F.cross_entropy(ita1_scores_per_image, targets_per_image)
        ita1_scores_per_text =  ita1_scores_per_text[0:-1:noCapt_perImg_temp] #(BS*noCapt_perImg)xBS = (21,7) --> BSxBS = (7,7) Only using natural caption
        ent_loss_text = F.cross_entropy(ita1_scores_per_text, targets_per_text)
        

        ## Image-text alignment cross entropy loss
        a1 = np.arange(0,noCapt_perImg*b_size,noCapt_perImg) ## position of natural captions
        b1 = np.arange(2,noCapt_perImg*b_size,noCapt_perImg) ## position of Error-Type-II captions 
        sel_idx = np.vstack((a1,b1)).T.reshape(-1) 
        ita2_scores_per_text = ita2_scores_per_text[sel_idx]  #[21,7]
        ita2_scores_per_image = ita2_scores_per_text.T #[7,21]
        ent2_loss_img = F.cross_entropy(ita2_scores_per_image, targets_per_image)
        #ita2_scores_per_text =  ita2_scores_per_text[0:-1:noCapt_perImg_temp] #(BS*noCapt_perImg)xBS = (21,7) --> BSxBS = (7,7) Only using natural caption
        #ent2_loss_text = criterion_4(ita2_scores_per_text, targets_per_text) 
        
        targets_scores = torch.arange(0,noCapt_perImg*b_size,noCapt_perImg).to(torch.long).to(device)
        ent_loss = F.cross_entropy(scores, targets_scores)
        
        ## Grammatical mistake Binary Cross Entropy loss
        g_targets = torch.Tensor([0,1]*b_size).to(device)
        a1 = np.arange(0,noCapt_perImg*b_size,noCapt_perImg) ## position of natural captions
        b1 = np.arange(noCapt_perImg-1,noCapt_perImg*b_size,noCapt_perImg) ## position of Error-Type-III captions 
        c1 = np.vstack((a1,b1))
        sel_idx = np.vstack((a1,b1)).T.reshape(-1)
        g_pen = g_pen[sel_idx] 
        bce_loss = F.binary_cross_entropy(g_pen,g_targets.to(g_pen.dtype))

        total_loss = ent1_loss_img +  ent2_loss_img +ent_loss_text+ ent_loss + bce_loss
        total_loss_meter.update(total_loss.item(), b_size)
        ent_loss_meter.update(ent_loss.item(), b_size)
        ent_img_loss_meter.update((ent1_loss_img + ent2_loss_img).item(), b_size)
        ent_text_loss_meter.update(ent_loss_text.item(), b_size)
        bce_loss_meter.update(bce_loss.item(), b_size)

        
        
    print('####Validation: TotalLoss: {tloss.val:.4f} ({tloss.avg:.4f}), '
                  'CELoss: {entloss.val:.4f} ({entloss.avg:.4f}), '
                  'CE_I_Loss: {entIloss.val:.4f} ({entIloss.avg:.4f}), '
                  'CE_T_Loss: {entTloss.val:.4f} ({entTloss.avg:.4f}), '
                  'BCELoss: {bceloss.val:.4f} ({bceloss.avg:.4f})'.format(tloss=total_loss_meter,
                   entloss=ent_loss_meter,entIloss=ent_img_loss_meter,entTloss=ent_text_loss_meter,
                   bceloss=bce_loss_meter))
    return total_loss_meter.avg

