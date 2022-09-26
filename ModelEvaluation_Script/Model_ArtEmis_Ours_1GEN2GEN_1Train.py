#!/usr/bin/env python
# coding: utf-8
"""
#### Code adapted from the source code of ArtEmis dataset paper
####################################################################
Training a neural-speaker.

The MIT License (MIT)
Originally created at 6/16/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
####################################################################
"""

import torch
import time
import numpy as np
import os
import os.path as osp
from torch import nn
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from ast import literal_eval
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import pdb 
from model.argument import parse_train_speaker_arguments,set_seed
from model.datasets_v2 import preprocess_dataset
from model.func_train_v2 import copy_params,check_copy_params,create_logger,load_state_dicts,AverageMeter,save_state_dicts

### Settings
data_dir = '../Dataset/ArtEmis/ArtEmis'
file_name = f'ArtEmis.csv'
no_transform = True 
#img_dir = f"{data_dir}/../OriginalArtEmis/wikiart"
img_dir = f"{data_dir}/../OriginalArtEmis/Images"

use_visualCLIP = True
use_textCLIP = False
use_vocabFAM = True

debug = False
random_seed = 2021
no_epochs = 200 # CHECK HERE
modelname = 'CLIPViTB16_1Gen' #'CLIPViTB16_full','CLIPViTB16_woSG','CLIPViTB16_1Gen','INRN34_full','INRN34_woSG','INViTB16_full','INViTB16_woSG'
output_dir = f'output/Ours_ArtEmis/{modelname}'
os.mkdir(output_dir)

FreezeCase = 0   #Other: Train all #1: Freeze: 'vis_enc' #2: Freeze: 'vis_enc', Transformer1, token_embedding, positional_embedding of 'capt_gen' # CHECK HERE
droprate = 0
lr_patience = 2
train_patience = 5
transformer_layers = 8
transformer_heads = 8
image_resolution = 224

if random_seed != -1:
    set_seed(random_seed)

if modelname == 'CLIPViTB16_full':
    backboneVisEnc = 'ViTB16' 
    modeltype = 'full'  
    batchsize = 32
    accum_iter = 2
    lr_visEnc =  1e-7
    lr_textEnc =  1e-3
    lr_others =  1e-3
elif modelname == 'CLIPViTB16_woSG':
    backboneVisEnc = 'ViTB16' 
    modeltype = 'woSG' 
    batchsize = 32 
    accum_iter = 2 
    lr_visEnc =  1e-7
    lr_textEnc =  1e-3
    lr_others =  1e-3
elif modelname == 'CLIPViTB16_1Gen':
    backboneVisEnc = 'ViTB16' 
    modeltype = '1Gen' 
    batchsize = 32 
    accum_iter = 2 
    lr_visEnc =  1e-7
    lr_textEnc =  1e-3
    lr_others =  1e-3
elif modelname == 'INRN34_full':
    backboneVisEnc = 'resnet34' 
    modeltype = 'full'  
    batchsize = 32
    accum_iter = 2 
    lr_visEnc =  1e-5
    lr_textEnc =  1e-3
    lr_others =  1e-3
elif modelname == 'INRN34_woSG':
    batchsize = 32
    accum_iter = 2
    lr_visEnc =  1e-5
    lr_textEnc =  1e-3
    lr_others =  1e-3
    backboneVisEnc = 'resnet34' 
    modeltype = 'woSG'  
elif modelname == 'INViTB16_full':
    backboneVisEnc = 'vit_base_patch16_224'
    modeltype = 'full'  
    batchsize = 32
    accum_iter = 2
    lr_visEnc =  1e-5
    lr_textEnc =  5e-3
    lr_others =  5e-3
elif modelname == 'INViTB16_woSG':
    backboneVisEnc = 'vit_base_patch16_224'
    modeltype = 'woSG' 
    batchsize = 32
    accum_iter = 2
    lr_visEnc =  1e-5
    lr_textEnc =  5e-3
    lr_others =  5e-3

if no_transform:
    img_transform = None
    if 'CLIP' in modelname:
        img_dir = img_dir + f'/CLIP_{image_resolution}'
    elif 'INRN34' in modelname:
        img_dir = img_dir + f'/IN_RN34_{image_resolution}'
    elif 'INViTB16'in modelname:
        img_dir = img_dir + f'/INViTB16_{image_resolution}'
    else:
        raise ValueError(f"Do not support model = {modelname}")
else:
    if 'CLIP' in modelname:
        img_transform = Compose([
            Resize(image_resolution, interpolation=BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif 'INRN34' in modelname:
        image_net_mean = [0.485, 0.456, 0.406]
        image_net_std = [0.229, 0.224, 0.225]
        resample_method = Image.LANCZOS
        normalize = Normalize(mean=image_net_mean, std=image_net_std)
        img_transform = Compose([Resize((image_resolution, image_resolution), resample_method),ToTensor(),normalize])
    elif 'INViTB16'in modelname:
        img_transform = Compose([
            Resize(size=248, interpolation=BICUBIC, max_size=None, antialias=None),
            CenterCrop(size=(224, 224)),
            ToTensor(),
            Normalize((0.5000, 0.5000, 0.5000), (0.5000, 0.5000, 0.5000))])
    else:
        raise ValueError(f"Do not support model = {modelname}")

if modeltype == 'full': 
    from model.model_v2 import fullArc as idcmodel
    from model.func_train_v2 import single_epoch_train_capGen as single_epoch_train_capGen 
    from model.func_train_v2 import val_loss_cal_capGen as val_loss_cal_capGen
elif modeltype == 'woSG': 
    from model.model_v2 import woSGArc as idcmodel 
    from model.func_train_v2 import single_epoch_train_capGen_woSub as single_epoch_train_capGen 
    from model.func_train_v2 import val_loss_cal_capGen_woSub as val_loss_cal_capGen 
elif modeltype == '1Gen': 
    from model.model_v2 import oneGenArc as idcmodel 
    from model.func_train_v2 import single_epoch_train_capGen_1Gen as single_epoch_train_capGen 
    from model.func_train_v2 import val_loss_cal_capGen_1Gen as val_loss_cal_capGen 
else:
    raise ValueError(f"Do not support modeltype = {modeltype}!!!")


## Load dataset
df = pd.read_csv(osp.join(data_dir, file_name))
df = df.where(pd.notnull(df), 'None')
if debug:
    df = df.sample(50)
print(f'Loaded {len(df)} captions!!!')
print(f'Number of literal captions:',len(df[df.subject_encoded == 'None']))
print(f'Number of non-literal captions:',len(df[df.subject_encoded != 'None']))
df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
df.subject_encoded = df.subject_encoded.apply(literal_eval)
df.predicate_encoded = df.predicate_encoded.apply(literal_eval)

if use_vocabFAM:
    from model.vocabulary import Vocabulary
    vocab = Vocabulary.load(osp.join(data_dir, 'ArtEmis_Vocab.pkl'))
    vocab_size = len(vocab)
    context_length = max(df.tokens_len) + 2 
    eos_token = 2
    sos_token = 1
else: # Use original vocabulary of CLIP
    vocab_size = vocab_size_CLIP
    context_length = context_length_CLIP
    eos_token = args.vocab_size-1 # eos_token is the last number
    sos_token = args.vocab_size-2 #sos_token is the last second number
    df['tokens_encoded'] = df['CLIP_tokens']



args = parse_train_speaker_arguments(['-output_dir',output_dir,'-data_dir',data_dir,'-img_dir', img_dir,
         '--modeltype', modeltype,'--backboneVisEnc',backboneVisEnc,'--image_resolution',image_resolution,                
         '--context_length',context_length,'--vocab_size',vocab_size,
         '--transformer_heads',transformer_heads,'--transformer_layers',transformer_layers,
         '--use_vocabFAM',use_vocabFAM,
         '--droprate',droprate, '--batch_size',batchsize,'--gpu',0,
         '--lr_visEnc',lr_visEnc,'--lr_textEnc',lr_textEnc,'--lr_others',lr_others,
         '--max_epochs',no_epochs,'--train_patience',train_patience,'--lr_patience',lr_patience,
         '--FreezeCase',FreezeCase,'--accum_iter',accum_iter,'--random_seed',random_seed,                           
         '--debug',debug,'--save_each_epoch',False,'--no_transform',no_transform])

print(args.output_dir)


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:" + str(args.gpu)) #CHECK HERE


data_loaders, _ = preprocess_dataset(df, args,img_transform)
print('Will use {} annotations for training.'.format(len(data_loaders['train'].dataset)))
print('Will use {} annotations for validation.'.format(len(data_loaders['val'].dataset)))
print('Will use {} annotations for testing.'.format(len(data_loaders['test'].dataset)))

## Describe model
model = idcmodel(args.backboneVisEnc,args.image_resolution,args.context_length,
                    args.vocab_size,sos_token,eos_token,
                    args.transformer_heads,args.transformer_layers,args.droprate)


if use_visualCLIP and 'ViT' in args.backboneVisEnc:
    from clip import clip
    if args.backboneVisEnc == 'ViTB16':
        clipmodel = 'ViT-B/16' 
    elif args.backboneVisEnc == 'ViTB32':
        clipmodel = "ViT-B/32"
    CLIPmodel,CLIPtransform,CLIPsettings = clip.load(clipmodel,device='cpu')
    embed_dim,image_resolution, vision_layers, vision_width, vision_patch_size,context_length_CLIP, vocab_size_CLIP, transformer_width, transformer_heads, _ = CLIPsettings
    if hasattr(model,'vis_enc'):
        copy_params(CLIPmodel.visual,model.vis_enc.visual)

model.to(device)

if FreezeCase == 1:
    for name, param in model.named_parameters():
        if 'vis_enc'  in name :
            param.requires_grad=False    
        else:
            param.requires_grad=True  
    print("Training parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == True: 
            print(name)
    print("Freezing parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == False: 
            print(name)
elif FreezeCase == 2:
    for name, param in model.named_parameters():
        if 'vis_enc'  in name or  'transformer_dec' in name or 'token_embedding' in name or  'positional_embedding' in name :
            param.requires_grad=False    
        else:
            param.requires_grad=True  
    print("Training parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == True: 
            print(name)
    print("Freezing parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == False: 
            print(name)
else:
    for name, param in model.named_parameters():
        param.requires_grad=True 
    print("Freezing parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == False: 
            print(name)
    print("Training parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == True: 
            print(name)


## Prepare Loss/Optimization
if args.lr_visEnc == args.lr_textEnc and args.lr_visEnc == args.lr_others:
    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr_others}])
    
else:
    
    if modeltype == 'full':   
        optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.vis_enc.parameters()), 'lr': args.lr_visEnc},

                {'params': filter(lambda p: p.requires_grad, model.IdCSubgen.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.IdCSubgen.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.IdCSubgen.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.IdCSubgen.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.IdCSubgen.text_projection]), 'lr': args.lr_others},

                {'params': filter(lambda p: p.requires_grad, model.IdCgen.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.IdCgen.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.IdCgen.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.IdCgen.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.IdCgen.text_projection]), 'lr': args.lr_others},

                {'params': filter(lambda p: p.requires_grad, model.LCgen.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.LCgen.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.LCgen.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.LCgen.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.LCgen.text_projection]), 'lr': args.lr_others}
        ])
    elif modeltype == 'woSG':   
        optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.vis_enc.parameters()), 'lr': args.lr_visEnc},

                {'params': filter(lambda p: p.requires_grad, model.LCgen.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.LCgen.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.LCgen.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.LCgen.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.LCgen.text_projection]), 'lr': args.lr_others},

                {'params': filter(lambda p: p.requires_grad, model.IdCgen.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.IdCgen.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.IdCgen.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.IdCgen.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.IdCgen.text_projection]), 'lr': args.lr_others}
        ])
    elif modeltype == '1Gen':   
        optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.vis_enc.parameters()), 'lr': args.lr_visEnc},
                {'params': filter(lambda p: p.requires_grad, model.textgen.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.textgen.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.textgen.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.textgen.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.textgen.text_projection]), 'lr': args.lr_others}
        ])
    else:
        raise ValueError(f"Do not support modeltype = {modeltype}!!!")
    


    
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.5,
                                                              patience=args.lr_patience,
                                                              verbose=True, 
                                                              min_lr=5e-7)

if modeltype == 'full': 
    criterion = [nn.CrossEntropyLoss().to(device)] * 3
elif modeltype == 'woSG': 
    criterion = [nn.CrossEntropyLoss().to(device)]*2
elif modeltype == '1Gen': 
    criterion = [nn.CrossEntropyLoss().to(device)]*1
else:
    raise ValueError(f"Do not support modeltype = {modeltype}!!!")


# Misc.
best_epoch = -1
best_val_loss = np.Inf
print_freq = 100# CHECK HERE
start_training_epoch = 1
no_improvement = 0
model_dir = osp.join(args.output_dir, 'checkpoints')
tb_log_dir = osp.join(args.output_dir, 'tb_log')
try:
    os.mkdir(tb_log_dir)
except: pass
tb_writer = SummaryWriter(tb_log_dir)
try:
    os.mkdir(model_dir)
except: pass
logger = create_logger(args.output_dir)
train_args = dict()

## Train.
logger.info('Starting the training of the speaker.')

out_name = osp.join(model_dir, 'model_epoch_0.pt')
save_state_dicts(out_name, 0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    
for epoch in range(start_training_epoch, args.max_epochs + 1):
    start_time = time.time()
    epoch_loss = single_epoch_train_capGen(data_loaders['train'], accum_iter, model, criterion, optimizer, epoch, device, 
                                           eos_token, print_freq=print_freq, 
                                           tb_writer=tb_writer,  **train_args)
    logger.info('####### Epoch loss {:.3f} time {:.1f}'.format(epoch_loss, (time.time() - start_time) / 60))
    
   
    val_loss = val_loss_cal_capGen(data_loaders['val'], model, device, eos_token,print_freq)
    logger.info('Validation loss {:.3f}'.format(val_loss))
    lr_scheduler.step(val_loss)
    if val_loss < best_val_loss:
        logger.info('Validation loss, *improved* @epoch {}'.format(epoch))
        best_val_loss = val_loss
        best_epoch = epoch
        out_name = osp.join(model_dir,  'best_model.pt')
        save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        no_improvement = 0
    else:
        logger.info('Validation loss did NOT improve @epoch {}'.format(epoch))
        no_improvement += 1
    if args.save_each_epoch:
        out_name = osp.join(model_dir, 'model_epoch_' + str(epoch) + '.pt')
        save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    tb_writer.add_scalar('training-loss-per-epoch', epoch_loss, epoch)
    tb_writer.add_scalar('val-loss-per-epoch', val_loss, epoch)
    tb_writer.add_scalar('learning-rate-per-epoch', optimizer.param_groups[0]['lr'], epoch)
    if no_improvement == args.train_patience:
        logger.warning('Stopping the training @epoch-{} due to lack of progress in '
                       'validation-reduction (patience hit {} '
                       'epochs'.format(epoch, args.train_patience))
        break
with open(osp.join(model_dir, 'final_result.txt'), 'w') as f_out:
    msg = ('Best Validation NLL: {:.4f} (achieved @epoch {})'.format(best_val_loss, best_epoch))
    f_out.write(msg)

logger.info('Finished training properly.')
tb_writer.close()


print(args.output_dir)
