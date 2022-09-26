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
                                                                         device,sos_token,eos_token,and_token, unk_token,vocab_size,vocab, 
                                                                         temperature=temperature,
                                                                         drop_bigrams=drop_bigrams,drop_unk=drop_unk,max_utterance_len=63)
        sel_capts = get_highest_prob_capt(beam_Cs,beam_Cs_s,vocab,model.textgen.context_length)
        df = captions_as_dataframe(data_loader.dataset, sel_capts)
    else:
        sel_Cs,gen_LCs,gen_IdCs = sample_captions_search(model,modeltype,loader, beam_size,device,sos_token,eos_token,and_token, unk_token,vocab_size,vocab, 
                                                                         temperature=temperature,
                                                                         drop_bigrams=drop_bigrams,drop_unk=drop_unk,max_utterance_len=63)
        
        df = captions_as_dataframe_LIdC(data_loader.dataset,sel_Cs,gen_LCs,gen_IdCs)
    return df


@torch.no_grad()
def sample_captions_search(model,modeltype, data_loader, beam_size, device,sos_token,eos_token,and_token, 
    unk_token,vocab_size,vocab, temperature=1,drop_bigrams=False,drop_unk=True,max_utterance_len=63):    
    if data_loader.batch_size != 1:
        raise ValueError('not implemented for bigger batch-sizes')

    model.eval()
    if modeltype == '1Gen': 
        Cs = list()
        Cs_s = list()
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
        sel_Cs = list()
        gen_LCs = list()
        gen_IdCs = list()
        for batch in tqdm.tqdm(data_loader):  # For each image (batch-size = 1)
            image = batch['image'].to(device)  # (1, 3, H, W)
            image_files = batch['image_file']

            if modeltype == 'woSG': 
                with torch.no_grad():
                    sel_C,gen_LC,gen_IdC = model.sel_caption_generate(image,image_files,vocab,beam_size,temperature,unk_token,and_token,drop_unk,drop_bigrams,device) 
            else:
                raise ValueError(f"Do not support modeltype = {modeltype}!!!")

            sel_Cs.append(sel_C)
            gen_LCs.append(gen_LC)
            gen_IdCs.append(gen_IdC)
        return sel_Cs,gen_LCs,gen_IdCs


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

def captions_as_dataframe_LIdC(captions_dataset,sel_Cs,gen_LCs,gen_IdCs):
    """convert the dataset/predicted-utterances (captions) to a pandas dataframe."""
    temp = captions_dataset.image_files.apply(wikiart_file_name_to_style_and_painting)
    art_style, painting = zip(*temp)   
    df = pd.DataFrame([art_style, painting, sel_Cs,gen_LCs,gen_IdCs]).transpose()
    column_names = ['art_style', 'painting', 'captions_predicted', 'LC_predicted', 'IdC_predicted']
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

