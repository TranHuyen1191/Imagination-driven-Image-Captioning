"""
#### Code adapted from the source code of ArtEmis dataset paper
"""
"""
Training a neural-speaker.

The MIT License (MIT)
Originally created at 6/16/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
   
import torch
import argparse
import json
import pprint
import pathlib
import os
import os.path as osp
from datetime import datetime
import random
import numpy as np

import pdb 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tupleType(s):
    print(s)
    s = eval(s)
    if isinstance(s, (tuple, list)):
        x, y, z, t = s
        return x, y, z, t
    else:
        return s

def nullable_string(val):
    if val == 'None':
        return None
    return int(val)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_train_speaker_arguments(notebook_options=None, save_args=True):
    parser = argparse.ArgumentParser(description='training figurative affective speaker')

    #### Non-optional arguments
    parser.add_argument('-output_dir', type=str, required=True)
    parser.add_argument('-data_dir', type=str, required=True, help='path to data directory')
    parser.add_argument('-img_dir', type=str, required=True, help='path to  image directory')


    #### Model parameters
    parser.add_argument('--modeltype', type=str, default='full', help='')
    parser.add_argument('--no_transform',type=str2bool, default=False, help='')
    parser.add_argument('--backboneVisEnc', type=str, default='ViTB32', help='')
    parser.add_argument('--image_resolution', type=int, default=224, help='')
    ## for caption generation
    parser.add_argument('--context_length', type=int, default=65,help='')#77
    parser.add_argument('--vocab_size', type=int, default=13444,help='')#49408
    parser.add_argument('--transformer_heads', type=int, default=8,help='')
    parser.add_argument('--transformer_layers', type=int, default=12,help='')
    parser.add_argument('--use_vocabFAM', type=str2bool, default=True, help='')


    #### Training parameters
    parser.add_argument('--droprate', type=float, default=1,help='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr_visEnc', type=float, default=1e-4)
    parser.add_argument('--lr_textEnc', type=float, default=1e-4)
    parser.add_argument('--lr_others', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)########### default: 50
    parser.add_argument('--train_patience', type=int, default=5, help='')
    parser.add_argument('--lr_patience', type=int, default=2, help='')
    parser.add_argument('--save_each_epoch',type=str2bool, default=False, help='')
    parser.add_argument('--FreezeCase', type=int, default=True, help='') #default=224
    parser.add_argument('--accum_iter', type=int, default=1, help='') 
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--use_timestamp', default=False, type=str2bool)

    #print(notebook_options)
    # Parse arguments
    if notebook_options is not None:  # Pass options directly
        #print([str(arg) for arg in notebook_options])
        args = parser.parse_args([str(arg) for arg in notebook_options])
    else:
        args = parser.parse_args() # Read from command line.

    if args.use_timestamp:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        output_dir = osp.join(args.output_dir, timestamp)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
            args.output_dir = output_dir

    # pprint them
    args_string = pprint.pformat(vars(args))
    print(args_string)
    #pdb.set_trace()
    if save_args:
        out = osp.join(args.output_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args


def parse_test_speaker_arguments(notebook_options=None):
    """ Parameters for testing (sampling) a neural-speaker.
    :param notebook_options: list, if you are using this via a jupyter notebook
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='testing-a-neural-speaker')

    ## Basic required arguments
    parser.add_argument('-speaker-saved-args', type=str, required=True, help='config.json.txt file for saved speaker model (output of train_speaker.py)')
    parser.add_argument('-speaker-checkpoint', type=str, required=True, help='saved model checkpoint ("best_model.pt" (output of train_speaker.py)')
    parser.add_argument('-img-dir', type=str, required=True, help='path to top image dir (typically that\'s the WikiArt top-dir)')
    parser.add_argument('-out-file', type=str, required=True, help='file to save the sampled utterances, their attention etc. as a pkl')


    ## Basic optional arguments
    parser.add_argument('--out-file-full', type=str, help='file to save all sampled utterances, their attention etc. as a pkl')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val', 'rest'], help='set the split of the dataset you want to annotate '
                                                                                                            'the code will load the dataset based on the dir-location marked '
                                                                                                            'in the input config.json.txt file. ' 
                                                                                                            'this param has no effect if a custom-data-csv is passed.')

    

    ## Optional arguments controlling the generation/sampling process
    
    parser.add_argument('--drop-bigrams', type=str2bool, default=True, help='if True, prevent the same bigram to occur '
                                                                            'twice in a sampled utterance')
    parser.add_argument('--drop-unk', type=str2bool, default=True, help='')
    parser.add_argument('--max-utterance-len', type=int, help='maximum allowed lenght for any sampled utterances. If not given '
                                                              'the maximum found in the underlying dataset split will be used.'
                                                              'Fot the official ArtEmis split for deep-nets that is 30 tokens.')

    

    ## To enable the pass of multiple configurations for the sampler at once! i.e., so you can try many
    ## sampling temperatures, methods to sample (beam-search vs. topk), beam-size (or more)
    ## You can provide a simple .json that specifies these values you want to try.
    ## See  >> data/speaker_sampling_configs << for examples
    ## Note. if you pass nothing the >> data/speaker_sampling_configs/selected_hyper_params.json.txt << will be used
    ##       these are parameters used in the the paper.
    parser.add_argument('--sampling-config-file', type=str, help='Note. if max-len, drop-unk '
                                                                 'and drop-bigrams are not specified in the json'
                                                                 'the directly provided values of these parameters '
                                                                 'will be used.')


    parser.add_argument('--random-seed', type=int, default=2021, help='if -1 it won\'t have an effect; else the sampler '
                                                                      'becomes deterministic')

    
    parser.add_argument('--gpu', type=str, default='0')

    


    # Parse arguments
    if notebook_options is not None:  # Pass options directly
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args() # Read from command line.

    # load "default"
    if args.sampling_config_file is None:
        up_dir = osp.split(pathlib.Path(__file__).parent.absolute())[0]
        args.sampling_config_file = osp.join(up_dir, 'data/speaker_sampling_configs/selected_hyper_params.json.txt')

    # pprint them
    print('\nParameters Specified:')
    args_string = pprint.pformat(vars(args))
    print(args_string)
    print('\n')

    return args

 
def parse_train_CLIPmetric_arguments(notebook_options=None):
    parser = argparse.ArgumentParser(description='training CLIP')
    parser.add_argument('-output_dir', type=str, required=True, help='path to data directory')
    parser.add_argument('-img_dir', type=str, required=True, help='path to  image directory')
    parser.add_argument('--backbone', type=str, default='ViTB32', help='')
    
    #### Training parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_CLIP', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)########### default: 50
    parser.add_argument('--train_patience', type=int, default=5, help='')
    parser.add_argument('--lr_patience', type=int, default=2, help='')
    parser.add_argument('--FreezeCase', type=int, default=True, help='') #default=224
    parser.add_argument('--accum_iter', type=int, default=1, help='') 
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--debug', default=False, type=str2bool)

    #print(notebook_options)
    # Parse arguments
    if notebook_options is not None:  # Pass options directly
        #print([str(arg) for arg in notebook_options])
        args = parser.parse_args([str(arg) for arg in notebook_options])
    else:
        args = parser.parse_args() # Read from command line.

   
    # pprint them
    args_string = pprint.pformat(vars(args))
    print(args_string)
    #pdb.set_trace()
    out = osp.join(args.output_dir, 'config.json.txt')
    with open(out, 'w') as f_out:
        json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args