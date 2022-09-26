import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
import pdb
from typing import Tuple, Union
from model.CLIP_model import CLIP_visual_encoder,LayerNorm,Transformer,Transformer_decoder,QuickGELU
from .commom_blocks import *
from collections import OrderedDict
from .resnet_encoder import ResnetEncoder
from .ViT_Encoder import vitEncoder

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from .AdapterCLIPMetric_3MLP import adapterCLIP as adapterCLIP_model
from .func_train_v2 import load_state_dicts
from PIL import Image

class woSGArc(nn.Module):
    def __init__(self, 
                 backboneVisEnc:str,
                 image_resolution: int,
                 # caption
                 context_length: int,
                 vocab_size: int,
                 sos_token: int,
                 eos_token: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 droprate: float,
                 ):
        super().__init__()
        
        if backboneVisEnc == 'ViTB16':
            embed_dim = 512
            vision_layers = 12
            vision_width  = 768
            vision_patch_size = 16
            self.vis_enc = CLIP_visual_encoder(embed_dim,image_resolution,vision_layers,vision_width,vision_patch_size)
            embed_dim_dec = self.vis_enc.visual.output_dim_xNdim
        elif backboneVisEnc == 'ViTB32':
            embed_dim = 512
            vision_layers = 12
            vision_width  = 768
            vision_patch_size = 32
            self.vis_enc = CLIP_visual_encoder(embed_dim,image_resolution,vision_layers,vision_width,vision_patch_size)
            embed_dim_dec = self.vis_enc.visual.output_dim_xNdim
        elif 'resnet' in backboneVisEnc:
            self.vis_enc = ResnetEncoder(backboneVisEnc).unfreeze()
            embed_dim_dec = self.vis_enc.embedding_dimension()
        else:
            self.vis_enc = vitEncoder(backboneVisEnc).unfreeze()
            embed_dim_dec = self.vis_enc.embedding_dimension()
            
        self.LCgen = text_generator_Trans(embed_dim_dec,context_length,vocab_size,sos_token,
            eos_token,transformer_heads,transformer_layers,droprate)
        self.IdCgen = text_generator_Trans(embed_dim_dec,context_length,vocab_size,sos_token,
            eos_token,transformer_heads,transformer_layers,droprate)




    def forward(self, image,tokens_encoded):
        vis_emb  = self.vis_enc(image)
        gen_LC,_ =    self.LCgen(vis_emb, tokens_encoded) #torch.Size([64, 65, 10509]),torch.Size([64, 1, 512])
        gen_IdC,_ =    self.IdCgen(vis_emb, tokens_encoded) #torch.Size([64, 65, 10509]),torch.Size([64, 1, 512])
        return gen_LC,gen_IdC


    def caption_generate(self,image,beam_size,temperature,unk_token=None,and_token=None,drop_unk=True,drop_bigrams=True):
        vis_emb  = self.vis_enc(image)
        gen_LC_exp_scores,gen_LC_exp,_ =  self.LCgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)
        gen_IdC_scores,gen_IdC_exp,_ =  self.IdCgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)

        gen_LC_exp = gen_LC_exp.tolist()
        gen_LC_exp_scores = gen_LC_exp_scores.tolist()
        gen_IdC_exp = gen_IdC_exp.tolist()
        gen_IdC_scores = gen_IdC_scores.tolist()
        return gen_LC_exp,gen_LC_exp_scores,gen_IdC_exp,gen_IdC_scores

    @torch.no_grad()
    def sel_caption_generate(self,image,image_files,vocab,beam_size,temperature,unk_token=None,and_token=None,drop_unk=True,drop_bigrams=True,device="cpu"):
        ## Using CScorer to score the generated imagination-driven captions
        
        # Create LC and IdC
        vis_emb  = self.vis_enc(image)
        gen_LC_exp_scores,gen_LC_exp,_ =  self.LCgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)
        gen_IdC_scores,gen_IdC_exp,_ =  self.IdCgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)

        gen_LC_exp = gen_LC_exp.tolist()
        gen_LC_exp_scores = gen_LC_exp_scores.tolist()
        gen_IdC_exp = gen_IdC_exp.tolist()
        gen_IdC_scores = gen_IdC_scores.tolist()

        # Calculate scores
        CScorer_alpha = 0.2
        CScorer_ck = f"output/adapterCLIP_3MLP/RN50x16_F1/checkpoints/best_model.pt"

        CScorer_embed_dim = 768
        CScorer_image_resolution = 384
        CScorer_vision_layers = (6, 8, 18, 8)
        CScorer_vision_width = 96
        CScorer_vision_patch_size = None
        CScorer_context_length_CLIP = 77
        CScorer_vocab_size_CLIP = 49408
        CScorer_transformer_width = 768
        CScorer_transformer_heads = 12
        CScorer_transformer_layers = 12

        CScorer_img_transform = Compose([ 
                                Resize(CScorer_image_resolution, interpolation=BICUBIC),
                                CenterCrop(CScorer_image_resolution),
                                lambda image: image.convert("RGB"),
                                ToTensor(),
                                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])


        CScorer = adapterCLIP_model(CScorer_embed_dim,CScorer_image_resolution,CScorer_vision_layers,CScorer_vision_width,
                    CScorer_vision_patch_size,CScorer_context_length_CLIP,CScorer_vocab_size_CLIP,CScorer_transformer_width,
                    CScorer_transformer_heads,CScorer_transformer_layers)

       
        loaded_epoch = load_state_dicts(CScorer_ck, map_location='cpu', model=CScorer)
        #print("Loaded CScorer at epoch = ",loaded_epoch)
        CScorer.to(device)
        CScorer.eval()
        

        
        assert len(image_files) == 1 # batchsize of test set = 1
        img = Image.open(image_files[0]+'.jpg')
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        img = CScorer_img_transform(img)
        image_inputs = img.unsqueeze(0)

        eos_token = vocab.eos
        gen_IdC = gen_IdC_exp[0]
        length = np.where(np.array(gen_IdC) == eos_token)[0][0] if len(np.where(np.array(gen_IdC) == eos_token)[0])>0 else len(gen_IdC)+1
        gen_IdC = ' '.join(vocab.decode(gen_IdC[1:length]))
        
        gen_LC = gen_LC_exp[0]
        length = np.where(np.array(gen_LC) == eos_token)[0][0] if len(np.where(np.array(gen_LC) == eos_token)[0])>0 else len(gen_LC)+1
        gen_LC = ' '.join(vocab.decode(gen_LC[1:length]))

        text_IdCs  = tokenize(gen_IdC)
        outputs = CScorer.get_score(image_inputs.to(device),text_IdCs.to(device),CScorer_alpha)
        ita1_score, ita2_score,g_pen,score = outputs
        if round(float(score),1)  >= 0.5:
            sel_C = gen_IdC
        else:
            sel_C = gen_LC
        return sel_C,gen_LC,gen_IdC

class oneGenArc(nn.Module):
    def __init__(self, 
                 backboneVisEnc:str,
                 image_resolution: int,
                 # caption
                 context_length: int,
                 vocab_size: int,
                 sos_token: int,
                 eos_token: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 droprate: float,
                 ):
        super().__init__()
        
        if backboneVisEnc == 'ViTB16':
            embed_dim = 512
            vision_layers = 12
            vision_width  = 768
            vision_patch_size = 16
            self.vis_enc = CLIP_visual_encoder(embed_dim,image_resolution,vision_layers,vision_width,vision_patch_size)
            embed_dim_dec = self.vis_enc.visual.output_dim_xNdim
        elif backboneVisEnc == 'ViTB32':
            embed_dim = 512
            vision_layers = 12
            vision_width  = 768
            vision_patch_size = 32
            self.vis_enc = CLIP_visual_encoder(embed_dim,image_resolution,vision_layers,vision_width,vision_patch_size)
            embed_dim_dec = self.vis_enc.visual.output_dim_xNdim
        elif 'resnet' in backboneVisEnc:
            self.vis_enc = ResnetEncoder(backboneVisEnc).unfreeze()
            embed_dim_dec = self.vis_enc.embedding_dimension()
        else:
            self.vis_enc = vitEncoder(backboneVisEnc).unfreeze()
            embed_dim_dec = self.vis_enc.embedding_dimension()
            
        self.textgen = text_generator_Trans(embed_dim_dec,context_length,vocab_size,sos_token,
            eos_token,transformer_heads,transformer_layers,droprate)

    def forward(self, image,tokens_encoded):
        vis_emb  = self.vis_enc(image)
        gen_C,_ =    self.textgen(vis_emb, tokens_encoded) #torch.Size([64, 65, 10509]),torch.Size([64, 1, 512])
        return gen_C


    def caption_generate(self,image,beam_size,temperature,unk_token=None,
                        and_token=None,drop_unk=True,drop_bigrams=True):
        vis_emb  = self.vis_enc(image)
        gen_C_exp_scores,gen_C_exp,_ =  self.textgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)
        
        gen_C_exp = gen_C_exp.tolist()
        gen_C_exp_scores = gen_C_exp_scores.tolist()
        return gen_C_exp,gen_C_exp_scores


class fullArc(nn.Module):
    def __init__(self, 
                 backboneVisEnc:str,
                 image_resolution: int,
                 # caption
                 context_length: int,
                 vocab_size: int,
                 sos_token: int,
                 eos_token: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 droprate: float,
                 ):
        super().__init__()
        if backboneVisEnc == 'ViTB16':
            embed_dim = 512
            vision_layers = 12
            vision_width  = 768
            vision_patch_size = 16
            self.vis_enc = CLIP_visual_encoder(embed_dim,image_resolution,vision_layers,vision_width,vision_patch_size)
            embed_dim_dec = self.vis_enc.visual.output_dim_xNdim
        elif backboneVisEnc == 'ViTB32':
            embed_dim = 512
            vision_layers = 12
            vision_width  = 768
            vision_patch_size = 32
            self.vis_enc = CLIP_visual_encoder(embed_dim,image_resolution,vision_layers,vision_width,vision_patch_size)
            embed_dim_dec = self.vis_enc.visual.output_dim_xNdim
        elif 'resnet' in backboneVisEnc:
            self.vis_enc = ResnetEncoder(backboneVisEnc).unfreeze()
            embed_dim_dec = self.vis_enc.embedding_dimension()
        else:
            self.vis_enc = vitEncoder(backboneVisEnc).unfreeze()
            embed_dim_dec = self.vis_enc.embedding_dimension()
        
        self.IdCSubgen = text_generator_Trans(embed_dim_dec,context_length,vocab_size,sos_token,
            eos_token,transformer_heads,transformer_layers,droprate)
        self.IdCgen = text_generator_Trans(embed_dim_dec,context_length,vocab_size,sos_token,
            eos_token,transformer_heads,transformer_layers,droprate)
        self.LCgen = text_generator_Trans(embed_dim_dec,context_length,vocab_size,sos_token,
            eos_token,transformer_heads,transformer_layers,droprate)
 
    def forward(self, image, subjects_encoded,tokens_encoded):
        vis_emb  = self.vis_enc(image)
        gen_LC,_ = self.LCgen(vis_emb, tokens_encoded)
        
        gen_subject,subject_emb_1dim =  self.IdCSubgen(vis_emb, subjects_encoded)
        vis_sub_emb = torch.cat([subject_emb_1dim, vis_emb], dim=1) 
        gen_IdC,_ = self.IdCgen(vis_sub_emb, tokens_encoded) #torch.Size([64, 65, 10509]),torch.Size([64, 1, 512])
        return gen_LC,gen_subject,gen_IdC

    def subject_generate(self,image,beam_size,temperature,unk_token=None,and_token=None,drop_unk=True,drop_bigrams=True):
        #pdb.set_trace()
        vis_emb  = self.vis_enc(image)
        gen_subjects_scores,gen_subjects,subjects_emb_1dim =  self.IdCSubgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)
        
        return gen_subjects_scores,gen_subjects,subjects_emb_1dim,vis_emb

    def caption_generate(self,vis_emb,subject_emb_1dim,beam_size,temperature,subject,unk_token=None,
                        and_token=None,drop_unk=True,drop_bigrams=True):
        gen_LC_exp_scores,gen_LC_exp,_ =  self.LCgen.gen_text(vis_emb,
            beam_size,temperature,None,unk_token,and_token,drop_unk,drop_bigrams)
        
        vis_sub_emb = torch.cat([subject_emb_1dim, vis_emb], dim=1) 
        gen_IdC_scores,gen_IdC_exp,_ =  self.IdCgen.gen_text(vis_sub_emb,
            beam_size,temperature,subject,unk_token,and_token,drop_unk,drop_bigrams)
        
        gen_LC_exp = gen_LC_exp.tolist()
        gen_LC_exp_scores = gen_LC_exp_scores.tolist()
        gen_IdC_exp = gen_IdC_exp.tolist()
        gen_IdC_scores = gen_IdC_scores.tolist()
        
        return gen_LC_exp,gen_LC_exp_scores,gen_IdC_exp,gen_IdC_scores




## Tokenizer of CLIP used for CScorer
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Any, Union, List
_tokenizer = _Tokenizer()
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
