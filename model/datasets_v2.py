"""
#### Code adapted from the source code of ArtEmis dataset paper
"""
"""
The MIT License (MIT)
Originally in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from os import path as osp
import pdb
from torchvision.transforms import Compose, ToTensor
from .argument import set_seed

class ICDataset(Dataset):
    def __init__(self, image_files,tokens_encoded,subjects_encoded,predicates_encoded,img_transform=None,context_length=65):
        super(ICDataset, self).__init__()
        self.image_files = image_files
        self.tokens_encoded = tokens_encoded
        self.subjects_encoded = subjects_encoded
        self.predicates_encoded = predicates_encoded
        self.img_transform = img_transform
        self.context_length = context_length

    def __getitem__(self, index):
        tokens = np.array(self.tokens_encoded[index]).astype(dtype=np.long)
        if self.subjects_encoded[index]:
            IdCflag = 1
            subjects = np.array(self.subjects_encoded[index]).astype(dtype=np.long)
            predicates = np.array(self.predicates_encoded[index]).astype(dtype=np.long)
        else:
            IdCflag = 0
            subjects = np.zeros([self.context_length,]).astype(dtype=np.long)
            predicates = np.zeros([self.context_length,]).astype(dtype=np.long)
        
        if self.img_transform is not None:
            if self.image_files is not None:
                img = Image.open(self.image_files[index]+ '.jpg')

                if img.mode is not 'RGB':
                    img = img.convert('RGB')

                img = self.img_transform(img)
            else:
                img = []
        else: # load .pt
            img =  torch.load(self.image_files[index]+ '.pt')
        
        item = {'image': img, 'tokens_encoded': tokens, 'subjects_encoded': subjects,
                'predicates_encoded': predicates,'index': index,'IdCflags':IdCflag,'image_file':self.image_files[index]}
        return item

    def __len__(self):
        return len(self.tokens_encoded)

def preprocess_dataset(df, args,img_transform):
    img_transforms = None
    img_transforms = dict()
    img_transforms['train'] = img_transform
    img_transforms['val'] = img_transform
    img_transforms['test'] = img_transform
    print("img_transforms:",img_transforms)
    set_seed(args.random_seed)
    datasets = dict()
    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True) 
        img_files = None

        img_files = g.apply(lambda x : osp.join(args.img_dir, x.art_style,  x.painting ), axis=1)
        img_files.name = 'image_files'
        dataset = ICDataset(img_files, g.tokens_encoded, g.subject_encoded,g.predicate_encoded,img_transform=img_transforms[split],context_length=args.context_length)

        datasets[split] = dataset

    dataloaders = dict()
    for split in datasets:
        b_size = args.batch_size if split=='train' else args.batch_size * 2
        dataloaders[split] = torch.utils.data.DataLoader(dataset=datasets[split],
                                                         batch_size=b_size,
                                                         shuffle=split=='train')
    return dataloaders, datasets

