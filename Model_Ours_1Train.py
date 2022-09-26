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
from model.argument import parse_train_speaker_arguments,set_seed
from model.datasets import preprocess_dataset_capGen,IdCDataset

from ast import literal_eval
import pandas as pd
from model.func_train import copy_params,check_copy_params,create_logger
from model.func_train import load_state_dicts
from model.func_train import AverageMeter,save_state_dicts

data_dir = '../Dataset/ArtEmis/ArtEmis_IdC'

## Load dataset
file_name = f'ArtEmis_IdCI.csv'
df = pd.read_csv(osp.join(data_dir, file_name))
print(f'Loaded {len(df)} captions!!!')

use_visualCLIP = True
use_textCLIP = False
use_vocabFAM = True

debug = False
random_seed = 2021
no_epochs = 200
modelname = 'CLIPViTB16_full' #'CLIPViTB16_full','CLIPViTB16_woSG','INRN34_full','INRN34_woSG','INViTB16_full','INViTB16_woSG'
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
        
output_dir = f'output/{modelname}'
os.mkdir(output_dir)

FreezeCase = 0   #Other: Train all
                #1: Freeze: 'vis_enc'
                #2: Freeze: 'vis_enc', Transformer1, token_embedding, positional_embedding of 'capt_gen' 
droprate = 0
lr_patience = 2
train_patience = 5
transformer_layers = 8
transformer_heads = 8
image_resolution = 224

if modeltype == 'full': 
    from model.model import fullArc as idcmodel
    from model.func_train import single_epoch_train_capGen as single_epoch_train_capGen 
    from model.func_train import val_loss_cal_capGen as val_loss_cal_capGen
elif modeltype == 'woSG': 
    from model.model import woSGArc as idcmodel 
    from model.func_train import single_epoch_train_capGen_woSub as single_epoch_train_capGen 
    from model.func_train import val_loss_cal_capGen_woSub as val_loss_cal_capGen 
else:
    raise ValueError(f"Do not support modeltype = {modeltype}!!!")

if use_vocabFAM:
    from model.vocabulary import Vocabulary
    vocab = Vocabulary.load(osp.join(data_dir, 'ArtEmis_IdCI_Vocab.pkl'))
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

img_dir = f"{data_dir}/Images/CLIP_{image_resolution}"




args = parse_train_speaker_arguments(['-output_dir',output_dir,'-data_dir',data_dir,'-img_dir', img_dir,
         '--modeltype', modeltype,'--backboneVisEnc',backboneVisEnc,'--image_resolution',image_resolution,                
         '--context_length',context_length,'--vocab_size',vocab_size,
         '--transformer_heads',transformer_heads,'--transformer_layers',transformer_layers,
         '--use_vocabFAM',use_vocabFAM,
         '--droprate',droprate, '--batch_size',batchsize,'--gpu',0,
         '--lr_visEnc',lr_visEnc,'--lr_textEnc',lr_textEnc,'--lr_others',lr_others,
         '--max_epochs',no_epochs,'--train_patience',train_patience,'--lr_patience',lr_patience,
         '--FreezeCase',FreezeCase,'--accum_iter',accum_iter,'--random_seed',random_seed,                           
         '--debug',debug,'--save_each_epoch',False])

print(args.output_dir)
if args.random_seed != -1:
    set_seed(args.random_seed)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:" + str(args.gpu))



def padding(x):
    padd_size = args.context_length - len(x) 
    x = x+[0]*padd_size
    return x

df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
df.subject_encoded = df.subject_encoded.apply(literal_eval)
df.subject_encoded = df.subject_encoded.apply(padding)
df.predicate_encoded = df.predicate_encoded.apply(literal_eval)

data_loaders, _ = preprocess_dataset_capGen(df, args)
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
else:
    for name, param in model.named_parameters():
        param.requires_grad=True 
    print("Freezing parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad == False: 
            print(name)


## Prepare Loss/Optimization
if args.lr_visEnc == args.lr_textEnc and args.lr_visEnc == args.lr_others:
    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr_others}])
    
else:
    
    if modeltype == 'full': 
        optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.vis_enc.parameters()), 'lr': args.lr_visEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_prefix.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_prefix.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.capt_gen_prefix.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_prefix.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.capt_gen_prefix.text_projection]), 'lr': args.lr_others},

                {'params': filter(lambda p: p.requires_grad, model.capt_gen_full.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_full.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.capt_gen_full.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_full.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.capt_gen_full.text_projection]), 'lr': args.lr_others}
        ])
    elif modeltype == 'woSG': 
        optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.vis_enc.parameters()), 'lr': args.lr_visEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_full.transformer_dec.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_full.token_embedding.parameters()), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, [model.capt_gen_full.positional_embedding]), 'lr': args.lr_textEnc},
                {'params': filter(lambda p: p.requires_grad, model.capt_gen_full.ln_final.parameters()), 'lr': args.lr_others},
                {'params': filter(lambda p: p.requires_grad, [model.capt_gen_full.text_projection]), 'lr': args.lr_others}
        ])
    else:
        raise ValueError(f"Do not support modeltype = {modeltype}!!!")
    


    
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.5,
                                                              patience=args.lr_patience,
                                                              verbose=True, 
                                                              min_lr=5e-7)

if modeltype == 'full': 
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    criterion = [criterion1, criterion2]
elif modeltype == 'woSG': 
    criterion = [nn.CrossEntropyLoss().to(device)]
else:
    raise ValueError(f"Do not support modeltype = {modeltype}!!!")


# Misc.
best_epoch = -1
best_val_loss = np.Inf
print_freq = 10
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
