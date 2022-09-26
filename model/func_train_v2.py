#!/usr/bin/env python
# coding: utf-8
"""
#### Code adapted from the source code of ArtEmis dataset paper
"""
import pdb
import torch
import os.path as osp
import time 
def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """ Save torch items with a state_dict
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)

def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """ Load torch items from saved state_dictionaries
    """
    #pdb.set_trace()
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch

import sys
import logging
def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(osp.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def copy_params(model_init,model):
    state_dict_init = model_init.state_dict()
    #for key in ["proj", "ln_post.weight", "ln_post.bias"]:
    #    if key in state_dict_init:
    #        del state_dict_init[key]
    model.load_state_dict(state_dict_init)
    return True
def check_copy_params(model_init,model):
    for named_param, named_param_init in zip(model.named_parameters(), model_init.named_parameters()):
        name,param = named_param
        name_init,param_init = named_param_init
        print(name,name_init)
        if param.ndim == 1:
            print(param[0])
            print(param_init[0])
        elif param.ndim == 2:
            print(param[0][0])
            print(param_init[0][0])
        else:
            print(param[0][0][0])
            print(param_init[0][0][0])
        break    

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

@torch.no_grad()
def val_loss_cal_capGen(val_loader, model, device, eos_token,print_freq=10,debug=False):
    
    entropy_valloss_meter = AverageMeter()  # entropy loss (per word decoded)
    model.eval()

    for i, batch in enumerate(val_loader):
        imgs = batch['image'].to(device)
        capts = batch['tokens_encoded'].to(device)
        subjects = batch['subjects_encoded'].to(device)
        predicates = batch['predicates_encoded'].to(device)    
        IdCflags = batch['IdCflags'].to(device)   
        b_size = len(imgs)

        outputs = model(imgs,subjects,capts)
        gen_LCs,gen_subjects,gen_IdCs = outputs

        #extract generated predicates from gen_IdCs
        gen_predicates = torch.zeros(gen_IdCs.size()).to(device)
        for idx_i,(gen_capt,subject) in enumerate(zip(gen_IdCs,subjects)):
            pos_eos = torch.where(subject == eos_token)[0].tolist()
            if len(pos_eos) == 0:
                pos_eos = gen_subjects.size(0)-1
            else:
                pos_eos = pos_eos[0] #Get the position of the first eos token
            gen_predicate = gen_capt[pos_eos-1:]
            gen_predicates[idx_i,:len(gen_predicate)] = gen_predicate

        # remove sos_token and get lengths of predicate_targes and subject_targets
        # Note: Literal captions --> length = 0
        subject_targets = subjects[:,1:]
        predicate_targets = predicates[:,1:]
        subject_lengths = []
        predicate_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 1
        for cnt,(IdCflag,subject_target,predicate_target) in enumerate(zip(IdCflags,subject_targets,predicate_targets)):
            if IdCflag == 1:
                sel_idx.append(cnt)
                subject_lengths.append((torch.where(subject_target == eos_token)[0] + 1).tolist()[0])
                predicate_lengths.append((torch.where(predicate_target == eos_token)[0] + 1).tolist()[0])
        

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        if len(sel_idx)>0:
            predicate_logits = pack_padded_sequence(gen_predicates[sel_idx], predicate_lengths, batch_first=True,enforce_sorted = False)
            predicate_targets = pack_padded_sequence(predicate_targets[sel_idx], predicate_lengths, batch_first=True,enforce_sorted = False)
            subject_logits = pack_padded_sequence(gen_subjects[sel_idx], subject_lengths, batch_first=True,enforce_sorted = False)
            subject_targets = pack_padded_sequence(subject_targets[sel_idx], subject_lengths, batch_first=True,enforce_sorted = False)
            Sub_ent_loss = F.cross_entropy(subject_logits.data, subject_targets.data)
            Pre_ent_loss = F.cross_entropy(predicate_logits.data, predicate_targets.data)
        else:
            Sub_ent_loss = 0 
            Pre_ent_loss = 0

        LC_targets = capts[:,1:]
        LC_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 0
        for cnt,(IdCflag,LC_target) in enumerate(zip(IdCflags,LC_targets)):
            if IdCflag == 0:
                sel_idx.append(cnt)
                LC_lengths.append((torch.where(LC_target == eos_token)[0] + 1).tolist()[0])

        if len(sel_idx)>0:
            LC_logits = pack_padded_sequence(gen_LCs[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)
            LC_targets = pack_padded_sequence(LC_targets[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)        
            LC_ent_loss = F.cross_entropy(LC_logits.data, LC_targets.data)
        else:
            LC_ent_loss = 0
        ent_loss = (Sub_ent_loss + Pre_ent_loss + LC_ent_loss)/3
        entropy_valloss_meter.update(ent_loss.item(), b_size)
    return entropy_valloss_meter.avg

import numpy as np
def single_epoch_train_capGen(train_loader, accum_iter, model, criterion, optimizer, epoch, device, eos_token,tb_writer=None,debug=False, **kwargs):
    print_freq = kwargs.get('print_freq', 100)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    entropy_loss_meter = AverageMeter()  # entropy loss (per word decoded)
    total_loss_meter = AverageMeter()
    start = time.time()
    steps_taken = (epoch-1) * len(train_loader.dataset)
    model.train()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - start)
        imgs = batch['image'].to(device)
        capts = batch['tokens_encoded'].to(device)
        subjects = batch['subjects_encoded'].to(device)
        predicates = batch['predicates_encoded'].to(device)    
        IdCflags = batch['IdCflags'].to(device)   
        b_size = len(imgs)

        outputs = model(imgs,subjects,capts)
        gen_LCs,gen_subjects,gen_IdCs = outputs

        #extract generated predicates from gen_IdCs
        gen_predicates = torch.zeros(gen_IdCs.size()).to(device)
        for idx_i,(gen_capt,subject) in enumerate(zip(gen_IdCs,subjects)):
            pos_eos = torch.where(subject == eos_token)[0].tolist()
            if len(pos_eos) == 0:
                pos_eos = gen_subjects.size(0)-1
            else:
                pos_eos = pos_eos[0] #Get the position of the first eos token
            gen_predicate = gen_capt[pos_eos-1:]
            gen_predicates[idx_i,:len(gen_predicate)] = gen_predicate

        # remove sos_token and get lengths of predicate_targes and subject_targets
        # Note: Literal captions --> length = 0
        subject_targets = subjects[:,1:]
        predicate_targets = predicates[:,1:]
        subject_lengths = []
        predicate_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 1
        for cnt,(IdCflag,subject_target,predicate_target) in enumerate(zip(IdCflags,subject_targets,predicate_targets)):
            if IdCflag == 1:
                sel_idx.append(cnt)
                subject_lengths.append((torch.where(subject_target == eos_token)[0] + 1).tolist()[0])
                predicate_lengths.append((torch.where(predicate_target == eos_token)[0] + 1).tolist()[0])
        

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        if len(sel_idx)>0:
            predicate_logits = pack_padded_sequence(gen_predicates[sel_idx], predicate_lengths, batch_first=True,enforce_sorted = False)
            predicate_targets = pack_padded_sequence(predicate_targets[sel_idx], predicate_lengths, batch_first=True,enforce_sorted = False)
            subject_logits = pack_padded_sequence(gen_subjects[sel_idx], subject_lengths, batch_first=True,enforce_sorted = False)
            subject_targets = pack_padded_sequence(subject_targets[sel_idx], subject_lengths, batch_first=True,enforce_sorted = False)
            Sub_ent_loss = criterion[0](subject_logits.data, subject_targets.data)
            Pre_ent_loss = criterion[1](predicate_logits.data, predicate_targets.data)
        else:
            Sub_ent_loss = 0 
            Pre_ent_loss = 0

        LC_targets = capts[:,1:]
        LC_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 0
        for cnt,(IdCflag,LC_target) in enumerate(zip(IdCflags,LC_targets)):
            if IdCflag == 0:
                sel_idx.append(cnt)
                LC_lengths.append((torch.where(LC_target == eos_token)[0] + 1).tolist()[0])

        if len(sel_idx)>0:
            LC_logits = pack_padded_sequence(gen_LCs[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)
            LC_targets = pack_padded_sequence(LC_targets[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)        
            LC_ent_loss = criterion[2](LC_logits.data, LC_targets.data)
        else:
            LC_ent_loss = 0
        ent_loss = (Sub_ent_loss + Pre_ent_loss + LC_ent_loss)/3
        entropy_loss_meter.update(ent_loss.item(), b_size)
        steps_taken += b_size  
        ent_loss = ent_loss / accum_iter  # normalize loss to account for batch accumulation
        ent_loss.backward() # backward pass

        if i % print_freq == 0 or i == len(train_loader):
            print("Losses:",Sub_ent_loss, Pre_ent_loss, LC_ent_loss)
            if len(sel_idx)>0:
                pred_LCs = [np.array(el.cpu())[0] for el in torch.topk(LC_logits.data,1)[1]]
                print(f"##################################LC No_corr_tokens/total_length: {sum(np.array(pred_LCs)==np.array(LC_targets.data.cpu()))}/{sum(LC_lengths)}:")
            if b_size - len(sel_idx)>0:
                pred_tokens = [np.array(el.cpu())[0] for el in torch.topk(predicate_logits.data,1)[1]]
                pred_subject_tokens = [np.array(el.cpu())[0] for el in torch.topk(subject_logits.data,1)[1]]
                print(f"##################################Sub No_corr_tokens/total_length: {sum(np.array(pred_subject_tokens)==np.array(subject_targets.data.cpu()))}/{sum(subject_lengths)}:")
                print(f"##################################Pre No_corr_tokens/total_length: {sum(np.array(pred_tokens)==np.array(predicate_targets.data.cpu()))}/{sum(predicate_lengths)}:")
               
        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            batch_time.update(time.time() - start)
            start = time.time()
            # Print status
            if print_freq is not None: 
                if (i+1) % (print_freq) == 0 or (i + 1 == len(train_loader)):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  loss=entropy_loss_meter))
        if tb_writer is not None:
            tb_writer.add_scalar('training-entropy-loss-with-batch-granularity', entropy_loss_meter.avg, steps_taken)

    return entropy_loss_meter.avg

@torch.no_grad()
def val_loss_cal_capGen_woSub(val_loader, model, device, eos_token,print_freq=10,debug=False):
    
    entropy_valloss_meter = AverageMeter()  # entropy loss (per word decoded)
    model.eval()

    for i, batch in enumerate(val_loader):
        imgs = batch['image'].to(device)
        capts = batch['tokens_encoded'].to(device)  
        IdCflags = batch['IdCflags'].to(device)   
        b_size = len(imgs)

        outputs = model(imgs,capts)
        gen_LCs,gen_IdCs = outputs

        IdC_targets = capts[:,1:]
        IdC_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 1
        for cnt,(IdCflag,IdC_target) in enumerate(zip(IdCflags,IdC_targets)):
            if IdCflag == 1:
                sel_idx.append(cnt)
                IdC_lengths.append((torch.where(IdC_target == eos_token)[0] + 1).tolist()[0])

        if len(sel_idx)>0:
            IdC_logits = pack_padded_sequence(gen_IdCs[sel_idx], IdC_lengths, batch_first=True,enforce_sorted = False)
            IdC_targets = pack_padded_sequence(IdC_targets[sel_idx], IdC_lengths, batch_first=True,enforce_sorted = False)        
            IdC_ent_loss = F.cross_entropy(IdC_logits.data, IdC_targets.data)
        else:
            IdC_ent_loss = 0

        LC_targets = capts[:,1:]
        LC_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 0
        for cnt,(IdCflag,LC_target) in enumerate(zip(IdCflags,LC_targets)):
            if IdCflag == 0:
                sel_idx.append(cnt)
                LC_lengths.append((torch.where(LC_target == eos_token)[0] + 1).tolist()[0])

        if len(sel_idx)>0:
            LC_logits = pack_padded_sequence(gen_LCs[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)
            LC_targets = pack_padded_sequence(LC_targets[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)        
            LC_ent_loss = F.cross_entropy(LC_logits.data, LC_targets.data)
        else:
            LC_ent_loss = 0
        ent_loss = (IdC_ent_loss + LC_ent_loss)/2
        entropy_valloss_meter.update(ent_loss.item(), b_size)
    return entropy_valloss_meter.avg

import numpy as np
def single_epoch_train_capGen_woSub(train_loader, accum_iter, model, criterion, optimizer, epoch, device, eos_token,tb_writer=None,debug=False, **kwargs):
    print_freq = kwargs.get('print_freq', 100)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    entropy_loss_meter = AverageMeter()  # entropy loss (per word decoded)
    total_loss_meter = AverageMeter()
    start = time.time()
    steps_taken = (epoch-1) * len(train_loader.dataset)
    model.train()

    for i, batch in enumerate(train_loader):
        imgs = batch['image'].to(device)
        capts = batch['tokens_encoded'].to(device)  
        IdCflags = batch['IdCflags'].to(device)   
        b_size = len(imgs)

        outputs = model(imgs,capts)
        gen_LCs,gen_IdCs = outputs

        IdC_targets = capts[:,1:]
        IdC_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 1
        for cnt,(IdCflag,IdC_target) in enumerate(zip(IdCflags,IdC_targets)):
            if IdCflag == 1:
                sel_idx.append(cnt)
                IdC_lengths.append((torch.where(IdC_target == eos_token)[0] + 1).tolist()[0])

        if len(sel_idx)>0:
            IdC_logits = pack_padded_sequence(gen_IdCs[sel_idx], IdC_lengths, batch_first=True,enforce_sorted = False)
            IdC_targets = pack_padded_sequence(IdC_targets[sel_idx], IdC_lengths, batch_first=True,enforce_sorted = False)        
            IdC_ent_loss = criterion[0](IdC_logits.data, IdC_targets.data)
        else:
            IdC_ent_loss = 0

        LC_targets = capts[:,1:]
        LC_lengths = []
        sel_idx = [] # Set of row indexes with IdCflag == 0
        for cnt,(IdCflag,LC_target) in enumerate(zip(IdCflags,LC_targets)):
            if IdCflag == 0:
                sel_idx.append(cnt)
                LC_lengths.append((torch.where(LC_target == eos_token)[0] + 1).tolist()[0])

        if len(sel_idx)>0:
            LC_logits = pack_padded_sequence(gen_LCs[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)
            LC_targets = pack_padded_sequence(LC_targets[sel_idx], LC_lengths, batch_first=True,enforce_sorted = False)        
            LC_ent_loss = criterion[1](LC_logits.data, LC_targets.data)
        else:
            LC_ent_loss = 0
        ent_loss = (IdC_ent_loss + LC_ent_loss)/2
        entropy_loss_meter.update(ent_loss.item(), b_size)
        steps_taken += b_size  
        ent_loss = ent_loss / accum_iter  # normalize loss to account for batch accumulation
        ent_loss.backward() # backward pass

        if i % print_freq == 0 or i == len(train_loader):
            print("Losses:",IdC_ent_loss, LC_ent_loss)
            if len(sel_idx)>0:
                pred_LCs = [np.array(el.cpu())[0] for el in torch.topk(LC_logits.data,1)[1]]
                print(f"##################################LC No_corr_tokens/total_length: {sum(np.array(pred_LCs)==np.array(LC_targets.data.cpu()))}/{sum(LC_lengths)}:")
            if b_size - len(sel_idx)>0:
                pred_IdCs = [np.array(el.cpu())[0] for el in torch.topk(IdC_logits.data,1)[1]]
                print(f"##################################IdC No_corr_tokens/total_length: {sum(np.array(pred_IdCs)==np.array(IdC_targets.data.cpu()))}/{sum(IdC_lengths)}:")
            
        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - start)
            start = time.time()
            
            #pdb.set_trace()
            # Print status
            if print_freq is not None: 
                if (i+1) % (print_freq) == 0 or (i + 1 == len(train_loader)):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  loss=entropy_loss_meter))
        if tb_writer is not None:
            tb_writer.add_scalar('training-entropy-loss-with-batch-granularity', entropy_loss_meter.avg, steps_taken)

    return entropy_loss_meter.avg


@torch.no_grad()
def val_loss_cal_capGen_1Gen(val_loader, model, device, eos_token,print_freq=10,debug=False):
    
    entropy_valloss_meter = AverageMeter()  # entropy loss (per word decoded)
    model.eval()

    for i, batch in enumerate(val_loader):
        imgs = batch['image'].to(device)
        capts = batch['tokens_encoded'].to(device)        
        b_size = len(imgs)
        
        gen_capts = model(imgs,capts)

        # remove sos_token 
        capts_targets = capts[:,1:]
        capts_lengths = (torch.where(capts_targets == eos_token)[1] + 1).tolist()

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        capts_logits = pack_padded_sequence(gen_capts, capts_lengths, batch_first=True,enforce_sorted = False)
        capts_targets = pack_padded_sequence(capts_targets, capts_lengths, batch_first=True,enforce_sorted = False)

        # Calculate loss
        #pdb.set_trace()
        ent_loss = F.cross_entropy(capts_logits.data, capts_targets.data)        

        entropy_valloss_meter.update(ent_loss.item(), b_size)
        
    return entropy_valloss_meter.avg

import numpy as np
def single_epoch_train_capGen_1Gen(train_loader, accum_iter, model, criterion, optimizer, epoch, device, eos_token,tb_writer=None,debug=False, **kwargs):
    """ Perform training for one epoch.
    :param train_loader: DataLoader for training data
    :param model: nn.ModuleDict with 'encoder', 'decoder' keys
    :param criterion: loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param device:
    """
    print_freq = kwargs.get('print_freq', 100)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    entropy_loss_meter = AverageMeter()  # entropy loss (per word decoded)
    total_loss_meter = AverageMeter()
    start = time.time()
    steps_taken = (epoch-1) * len(train_loader.dataset)
    model.train()

    for i, batch in enumerate(train_loader):
        imgs = batch['image'].to(device)
        capts = batch['tokens_encoded'].to(device)
        b_size = len(imgs)
        data_time.update(time.time() - start)

        gen_capts = model(imgs,capts)

        # remove sos_token 
        capts_targets = capts[:,1:]
        capts_lengths = (torch.where(capts_targets == eos_token)[1] + 1).tolist()

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        capts_logits = pack_padded_sequence(gen_capts, capts_lengths, batch_first=True,enforce_sorted = False)
        capts_targets = pack_padded_sequence(capts_targets, capts_lengths, batch_first=True,enforce_sorted = False)

        # Calculate loss
        #pdb.set_trace()
        ent_loss = criterion[0](capts_logits.data, capts_targets.data)
            
        # Keep track of metrics
        entropy_loss_meter.update(ent_loss.item(), b_size)
        steps_taken += b_size  

        # normalize loss to account for batch accumulation
        ent_loss = ent_loss / accum_iter 

        # backward pass
        ent_loss.backward()

        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - start)
            start = time.time()
            
            #pdb.set_trace()
            # Print status
            if print_freq is not None: 
                if (i+1) % (print_freq) == 0 or (i + 1 == len(train_loader)):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  loss=entropy_loss_meter))
        if tb_writer is not None:
            tb_writer.add_scalar('training-entropy-loss-with-batch-granularity', entropy_loss_meter.avg, steps_taken)

    return entropy_loss_meter.avg
