#!/usr/bin/env python
# coding: utf-8
"""
#### Code adapted from the source code of ArtEmis dataset paper
"""
import torch
import argparse
import json
import os.path as osp
import torch.nn.functional as F
import tqdm
import math
import pandas as pd
import numpy as np  
from model.datasets_v2 import ICDataset
from six.moves import cPickle

def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()
def splitall(path):
    """
    Examples:
        splitall('a/b/c') -> ['a', 'b', 'c']
        splitall('/a/b/c/')  -> ['/', 'a', 'b', 'c', '']

    NOTE: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = osp.split(path)
        if parts[0] == path:   # Sentinel for absolute paths.
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # Sentinel for relative paths.
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts
def wikiart_file_name_to_style_and_painting(filename):
    """
    Assumes a filename of a painting of wiki-art.
    :param filename:
    :return:
    """
    s = splitall(filename)
    art_style = s[-2]
    painting = s[-1]
    if '.jpg' in painting:
        painting = painting[:-len('.jpg')]
    if '.pt' in painting:
        painting = painting[:-len('.pt')]
    return art_style, painting
    
import pdb
def group_annotations_per_image(affective_dataset):
    """ Group the annotations per image.
    :param affective_dataset: an AffectiveCaptionDataset
    :return: for each image its tokens/emotions as pandas Dataframes
    """
    df = pd.concat([affective_dataset.image_files, affective_dataset.tokens_encoded, 
                    affective_dataset.subjects_encoded, affective_dataset.predicates_encoded], axis=1)
    print(df.columns)
    dfgroup = df.groupby('image_files',sort=False)

    tokens_encoded_grouped = dfgroup['tokens_encoded'].apply(list).reset_index(name='tokens_encoded')

    return tokens_encoded_grouped['image_files']
def grounding_dataset_per_image_dummy(loader, device=None):
    """
    Convenience function. Given a loader carrying an affective dataset, make a new loader only w.r.t.
    unique images of the dataset, & optionally add to each image the emotion predicted by the img2emo_clf.
    The new loader can be used to sample utterances over the unique images.
    :param loader:
    :param img2emo_clf:
    :param device:
    :return:
    """
    affective_dataset = loader.dataset
    img_files = group_annotations_per_image(affective_dataset)
    img_trans = affective_dataset.img_transform
    batch_size = loader.batch_size

 
    dummy_token = pd.Series(np.ones(len(img_files), dtype=int) * -1)

    
    new_dataset = ICDataset(img_files,tokens_encoded=dummy_token, subjects_encoded=dummy_token,
                                             predicates_encoded=dummy_token,img_transform=img_trans)

    new_loader = torch.utils.data.DataLoader(dataset=new_dataset, batch_size=batch_size,shuffle=False)
    return new_loader 

def read_saved_args(config_file, override_args=None, verbose=False):
    """
    :param config_file: json file containing arguments
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args

def versatile_caption_sampler(model, modeltype, data_loader ,vocab,device, sos_token,eos_token,and_token, unk_token,vocab_size, sampling_rule='beam',
                              beam_size=None, topk=None, temperature=1,drop_bigrams=False,drop_unk=True,max_utterance_len=63):
    dset = data_loader.dataset
    loader = torch.utils.data.DataLoader(dset,shuffle=False) # batch-size=1
    if modeltype == '1Gen': 
        beam_Cs,beam_Cs_s = sample_captions_search(model,modeltype,loader, beam_size,
                                                                         device,sos_token,eos_token,and_token, unk_token,vocab_size, 
                                                                         temperature=temperature,
                                                                         drop_bigrams=drop_bigrams,drop_unk=drop_unk,max_utterance_len=63)
        sel_capts = get_highest_prob_capt(beam_Cs,beam_Cs_s,vocab,model.textgen.context_length)
        df = captions_as_dataframe(data_loader.dataset, sel_capts)
    else:
        beam_LCs,beam_LCs_s,beam_IdCs,beam_IdCs_s = sample_captions_search(model,modeltype,loader, beam_size,
                                                                         device,sos_token,eos_token,and_token, unk_token,vocab_size, 
                                                                         temperature=temperature,
                                                                         drop_bigrams=drop_bigrams,drop_unk=drop_unk,max_utterance_len=63)
        sel_capts,sel_LCs,sel_IdCs,sel_LCs_s,sel_IdCs_s = get_highest_prob_capt_LIdC(beam_LCs,beam_LCs_s,beam_IdCs,beam_IdCs_s,vocab,
                                                  model.LCgen.context_length)
    
        df = captions_as_dataframe_LIdC(data_loader.dataset, sel_capts,sel_LCs,sel_IdCs,sel_LCs_s,sel_IdCs_s)
    return df

@torch.no_grad()
def sample_captions_search(model,modeltype, data_loader, beam_size, device,sos_token,eos_token,and_token, 
    unk_token,vocab_size, temperature=1,drop_bigrams=False,drop_unk=True,max_utterance_len=63):    
    if data_loader.batch_size != 1:
        raise ValueError('not implemented for bigger batch-sizes')

    model.eval()
    Cs = list()
    Cs_s = list()
    LCs = list()
    LCs_s = list()
    IdCs = list()
    IdCs_s = list()
    if modeltype == '1Gen': 
        for batch in tqdm.tqdm(data_loader):  # For each image (batch-size = 1)
            Cs_img = list()
            Cs_s_img = list()
            image = batch['image'].to(device)  # (1, 3, H, W)
            outputs = model.caption_generate(image,beam_size,temperature,unk_token,and_token,drop_unk,drop_bigrams)
            complete_seqs,complete_seqs_scores  = outputs

            for C,C_s in zip(complete_seqs,complete_seqs_scores):
                Cs_img.append(C)
                Cs_s_img.append(C_s)
                
            Cs.append(Cs_img)
            Cs_s.append(Cs_s_img)
        return Cs,Cs_s
    else:
        for batch in tqdm.tqdm(data_loader):  # For each image (batch-size = 1)
            LCs_img = list()
            LCs_s_img = list()
            IdCs_img = list()
            IdCs_s_img = list()
            Cs_img = list()
            Cs_s_img = list()
            image = batch['image'].to(device)  # (1, 3, H, W)

            if modeltype == 'full':
                if beam_size != 1:
                    print("CHECK HERE: Using beamsize = 1 for generating subjects!!!")
                #gen_pref_sco,gen_pref,subject_emb_1dim,vis_emb_Ndim = model.subject_generate(image,beam_size,temperature,unk_token,and_token,drop_unk,drop_bigrams) 
                gen_pref_sco,gen_pref,subject_emb_1dim,vis_emb_Ndim = model.subject_generate(image,1,temperature,unk_token,and_token,drop_unk,drop_bigrams) 
                outputs  = model.caption_generate(vis_emb_Ndim,subject_emb_1dim,beam_size,temperature,gen_pref,unk_token,and_token,drop_unk,drop_bigrams)
                complete_LC_seqs,complete_LC_seqs_scores,complete_IdC_seqs,complete_IdC_seqs_scores  = outputs
            elif modeltype == 'woSG': 
                outputs = model.caption_generate(image,beam_size,temperature,unk_token,and_token,drop_unk,drop_bigrams) 
                complete_LC_seqs,complete_LC_seqs_scores,complete_IdC_seqs,complete_IdC_seqs_scores  = outputs
            else:
                raise ValueError(f"Do not support modeltype = {modeltype}!!!")

            for LC,LC_s,IdC,IdC_s in zip(complete_LC_seqs,complete_LC_seqs_scores,complete_IdC_seqs,complete_IdC_seqs_scores):
                LCs_img.append(LC)
                LCs_s_img.append(LC_s)
                IdCs_img.append(IdC)
                IdCs_s_img.append(IdC_s)
                
            LCs.append(LCs_img)
            LCs_s.append(LCs_s_img)
            IdCs.append(IdCs_img)
            IdCs_s.append(IdCs_s_img)
        return LCs,LCs_s,IdCs,IdCs_s


def get_highest_prob_capt_LIdC(beam_LCs,beam_LCs_s,beam_IdCs,beam_IdCs_s,vocab,max_length = 63):
    sel_capts = []
    sel_LCs = []
    sel_LCs_s = []
    sel_IdCs = []
    sel_IdCs_s = []
    eos_token = vocab.eos
    def getBestCapt(capts_img,scores_img):
        lengths = [np.where(np.array(capt) == eos_token)[0][0] if len(np.where(np.array(capt) == eos_token)[0])>0 else max_length+1 for capt in capts_img]
        
        indexes = np.where(np.array(lengths)<=max_length)[0]
        if len(indexes)>0: #Only get captions including eos_token
            scores_img = [scores_img[i] for i in indexes]
            capts_img = [capts_img[i] for i in indexes]
            
        
        top_k_idx = np.argsort(scores_img)[::-1][0] #get highest prob
        capt = capts_img[top_k_idx]
        score = scores_img[top_k_idx]
        length = np.where(np.array(capt) == eos_token)[0]
        if len(length) == 0:
            length = max_length
        else:
            length = length[0]
        return capt[1:length],score
        
    for LCs_img,LCs_s_img,IdCs_img,IdCs_s_img in zip(beam_LCs,beam_LCs_s,beam_IdCs,beam_IdCs_s): #each image
        sel_LC,sel_LC_s = getBestCapt(LCs_img,LCs_s_img)
        sel_LCs_s.append(sel_LC_s)
        sel_LCs.append(' '.join(vocab.decode(sel_LC)))
        sel_IdC,sel_IdC_s = getBestCapt(IdCs_img,IdCs_s_img)
        sel_IdCs.append(' '.join(vocab.decode(sel_IdC)))
        sel_IdCs_s.append(sel_IdC_s)
        

        if sel_LC_s > sel_IdC_s:
            sel_capts.append(' '.join(vocab.decode(sel_LC)))
        else:
            sel_capts.append(' '.join(vocab.decode(sel_IdC)))

    return sel_capts,sel_LCs,sel_IdCs,sel_LCs_s,sel_IdCs_s

def get_highest_prob_capt(beam_Cs,beam_Cs_s,vocab,max_length = 63):
    sel_capts = []
    eos_token = vocab.eos
    def getBestCapt(capts_img,scores_img):
        lengths = [np.where(np.array(capt) == eos_token)[0][0] if len(np.where(np.array(capt) == eos_token)[0])>0 else max_length+1 for capt in capts_img]
        
        indexes = np.where(np.array(lengths)<=max_length)[0]
        if len(indexes)>0: #Only get captions including eos_token
            scores_img = [scores_img[i] for i in indexes]
            capts_img = [capts_img[i] for i in indexes]
            
        
        top_k_idx = np.argsort(scores_img)[::-1][0] #get highest prob
        capt = capts_img[top_k_idx]
        score = scores_img[top_k_idx]
        length = np.where(np.array(capt) == eos_token)[0]
        if len(length) == 0:
            length = max_length
        else:
            length = length[0]
        return capt[1:length],score
        
    for Cs_img,Cs_s_img in zip(beam_Cs,beam_Cs_s): #each image
        sel_C,sel_C_s = getBestCapt(Cs_img,Cs_s_img)
        sel_capts.append(' '.join(vocab.decode(sel_C)))
        
    return sel_capts

def captions_as_dataframe_LIdC(captions_dataset,sel_capts,sel_LCs,sel_IdCs,sel_LCs_s,sel_IdCs_s):
    """convert the dataset/predicted-utterances (captions) to a pandas dataframe."""
    temp = captions_dataset.image_files.apply(wikiart_file_name_to_style_and_painting)
    art_style, painting = zip(*temp)   
    df = pd.DataFrame([art_style, painting, sel_capts,sel_LCs,sel_LCs_s,sel_IdCs,sel_IdCs_s]).transpose()
    column_names = ['art_style', 'painting', 'captions_predicted', 'LC_predicted', 'LC_score_predicted', 'IdC_predicted', 'IdC_score_predicted']
    df.columns = column_names
    return df
def captions_as_dataframe(captions_dataset,sel_capts):
    """convert the dataset/predicted-utterances (captions) to a pandas dataframe."""
    temp = captions_dataset.image_files.apply(wikiart_file_name_to_style_and_painting)
    art_style, painting = zip(*temp)   
    df = pd.DataFrame([art_style, painting, sel_capts]).transpose()
    column_names = ['art_style', 'painting', 'captions_predicted']
    df.columns = column_names
    return df


"""
def versatile_caption_sampler_beamPref(model, data_loader ,device, sos_token,eos_token,and_token, unk_token,vocab_size, sampling_rule='beam',
                              beam_size=None, topk=None, temperature=1,drop_bigrams=False,drop_unk=True,max_utterance_len=63):
    dset = data_loader.dataset
    loader = torch.utils.data.DataLoader(dset,shuffle=False) # batch-size=1

    beam_captions, beam_scores,beam_prefs_scores,beam_prefs = sample_captions_beam_search_beamPref(model, loader, beam_size,
                                                                         device,sos_token,eos_token,and_token, unk_token,vocab_size, 
                                                                         temperature=temperature,
                                                                         drop_bigrams=drop_bigrams,drop_unk=drop_unk,max_utterance_len=63)
        
    
    return beam_captions, beam_scores,beam_prefs_scores,beam_prefs,beam_emotions

@torch.no_grad()
def sample_captions_beam_search_beamPref(model, data_loader, beam_size, device,sos_token,eos_token,and_token, 
    unk_token,vocab_size, temperature=1,drop_bigrams=False,drop_unk=True,max_utterance_len=63):    

    if data_loader.batch_size != 1:
        raise ValueError('not implemented for bigger batch-sizes')
    model.eval()
    captions = list()
    prefs = list()
    emotions = list()
    capt_sco = list()
    pref_sco = list()

    for batch in tqdm.tqdm(data_loader):  # For each image (batch-size = 1)
        captions_img = list()
        prefs_img = list()
        emotions_img = list()
        capt_sco_img = list()
        pref_sco_img = list()

        image = batch['image'].to(device)  # (1, 3, H, W)
        #print(image)
        outputs = model.subject_generate_emo(image,beam_size,temperature,unk_token,and_token,drop_unk,drop_bigrams) 
        pred_emo_dist_beam,gen_prefs_sco_beam,gen_prefs_beam,prefs_emb_1dim_beam,vis_emb_Ndim = outputs
        
        for pred_emo_dist,gen_pref_sco,gen_pref,subject_emb_1dim in zip(pred_emo_dist_beam,gen_prefs_sco_beam,gen_prefs_beam,prefs_emb_1dim_beam): 
            sel_dist_emos,sel_emos = torch.topk(pred_emo_dist,1)
            gen_pref = gen_pref.unsqueeze(0)
            #pdb.set_trace()
            sel_emos = sel_emos  
            for sel_emo in sel_emos:
                #pdb.set_trace()
                vis_emo_emb = model.get_vis_emo_emb(sel_emo.unsqueeze(0),vis_emb_Ndim)

                outputs  = model.caption_generate(vis_emo_emb,1,temperature,gen_pref,unk_token,and_token,drop_unk,drop_bigrams)

                complete_seqs,complete_seqs_scores = outputs
                for seq,score in zip(complete_seqs,complete_seqs_scores):
                    captions_img.append(seq)
                    capt_sco_img.append(score)
                    pref_sco_img.append(gen_pref_sco.tolist())
                    emotions_img.append(sel_emo.tolist())
                    prefs_img.append(gen_pref.squeeze(0).tolist()) #exclude batch dim
                    #pdb.set_trace()
                
        captions.append(captions_img)
        prefs.append(prefs_img)
        emotions.append(emotions_img)
        capt_sco.append(capt_sco_img)
        pref_sco.append(pref_sco_img)
    return captions,capt_sco,pref_sco,prefs,emotions

"""