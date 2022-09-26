import torch
import pandas as pd
import os 
import os.path as osp
import numpy as np
from ast import literal_eval
from clip import clip 
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from random import seed
from random import choice
import pdb
from clip.model import convert_weights
from torch import nn
from model.argument import parse_train_CLIPmetric_arguments
from model.func_train_v2 import save_state_dicts
from model.argument import set_seed
from model.AdapterCLIPMetric_3MLP import train_adapterCLIP,val_adapterCLIP
from model.AdapterCLIPMetric_3MLP import adapterCLIP as adapterCLIP_model

torch.backends.cudnn.benchmark = True

FreezeCase = 1 # 1: Frozen, 2 Finetune
runRawImg = False
CLIP_name = 'RN50x16' #RN50, RN101, or RN50x4
initPrompt = 'A photo depicts'


no_ErrorTypes = 3
output_dir = f"output/adapterCLIP_3MLP/{CLIP_name}_F{FreezeCase}"

debug = False
print(f"########## DEBUG =  {debug}")

# Load dataset
df = pd.read_csv(f'../Dataset/ArtEmis/ArtEmis_IdC/ArtEmis_IdCII_3ErrType.csv')
df['captSet_CLIP_tokens'] = df['captSet_CLIP_tokens'].apply(literal_eval)
df['captSet_CLIP_tokens'] = df['captSet_CLIP_tokens'].apply(lambda x: x[:no_ErrorTypes+1])
print('Total number of loaded captions:', len(df))
print('Number of captions per image:', len(df.captSet_CLIP_tokens.iloc[0]))

if debug:
    df = df.sample(50) ## CHECK HERE

# Settings
batchsize = 10
accum_iter = 4
random_seed = 2021
lr_CLIP =  1e-7
lr =  1e-3
no_epochs = 200
lr_patience = 2
train_patience = 3
device = "cuda" if torch.cuda.is_available() else "cpu"## CHECK HERE


# Get hyper-parameters and weights of original CLIP
origCLIP,CLIPtransform,CLIPsettings = clip.load(CLIP_name,"cpu")
embed_dim,image_resolution, vision_layers, vision_width, vision_patch_size,context_length_CLIP, vocab_size_CLIP, transformer_width, transformer_heads, transformer_layers = CLIPsettings

os.makedirs('output', exist_ok=True)
os.makedirs("output/adapterCLIP_3MLP", exist_ok=True)
os.mkdir(output_dir) ## CHECK HERE


if runRawImg:
    img_dir = '../Dataset/ArtEmis/OriginalArtEmis/wikiart'
    img_transform = Compose([ 
                        Resize(image_resolution, interpolation=BICUBIC),
                        CenterCrop(image_resolution),
                        lambda image: image.convert("RGB"),
                        ToTensor(),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
else:
    img_transform = None
    img_dir = f'../Dataset/ArtEmis/OriginalArtEmis/Images/CLIP_{image_resolution}'

df['img_files'] = [osp.join(img_dir,s,p+'.jpg') for s,p in zip(df.art_style,df.painting)]


args = parse_train_CLIPmetric_arguments(['-output_dir',output_dir,'-img_dir',img_dir,
    '--backbone',CLIP_name,'--batch_size',batchsize,'--device',device,'--lr',lr,'--lr_CLIP',lr_CLIP,
    '--max_epochs',no_epochs,'--train_patience',train_patience,
    '--lr_patience',lr_patience,'--FreezeCase',FreezeCase,'--accum_iter',accum_iter,'--random_seed',2021])
set_seed(args.random_seed)

adapterCLIP = adapterCLIP_model(embed_dim,image_resolution,vision_layers,vision_width,
            vision_patch_size,context_length_CLIP,
            vocab_size_CLIP,transformer_width,transformer_heads,transformer_layers)

#Copy weights of original CLIP 
def copy_params(model_init,model):
    state_dict_init = model_init.state_dict()
    model.load_state_dict(state_dict_init,strict=False)
    return True

copy_params(origCLIP,adapterCLIP)
adapterCLIP.to(device)

class IdCIIDataset(Dataset):
    def __init__(self, image_files,captsets,img_transform=None):
        super().__init__()
        self.image_files = image_files
        self.captsets = captsets
        self.img_transform = img_transform
        print(img_transform)

    def __getitem__(self, index):
        captset = np.array(self.captsets[index]).astype(dtype=np.long)
        if self.img_transform is not None:
            if self.image_files is not None:
                img = Image.open(self.image_files[index])

                if img.mode is not 'RGB':
                    img = img.convert('RGB')

                img = self.img_transform(img)
            else:
                img = []
        else: # load .pt
            img =  torch.load(self.image_files[index][:-4]+ '.pt')

        item = {'image_file': self.image_files[index],'image': img, 'captset': captset, 'index': index}
        return item

    def __len__(self):
        return len(self.image_files)

def preprocess_Dataset(df, img_transform):
    set_seed(args.random_seed)
    datasets = dict()
    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True) 
        img_files = None
        img_trans = None

        img_files = g.img_files
        img_files.name = 'image_files'

        dataset = IdCIIDataset(img_files, g.captSet_CLIP_tokens,img_transform=img_transform)

        datasets[split] = dataset

    dataloaders = dict()
    for split in datasets:
        if split=='train' or split == 'val':
            b_size = args.batch_size  
        else:
            b_size = 1
        dataloaders[split] = torch.utils.data.DataLoader(dataset=datasets[split],
                                                         batch_size=b_size,
                                                         shuffle=split=='train')
    return dataloaders, datasets
dataloaders, datasets = preprocess_Dataset(df,img_transform)


if FreezeCase == 0 or FreezeCase == 2:
    for name, param in adapterCLIP.named_parameters():
        param.requires_grad=True 

    print("Trained parameters:")
    for name, param in adapterCLIP.named_parameters():
        if param.requires_grad == True: 
            print(name)
    print("Frozen parameters:")
    for name, param in adapterCLIP.named_parameters():
        if param.requires_grad == False: 
            print(name)
elif FreezeCase == 1:
    for name, param in adapterCLIP.named_parameters():
        if 'MLP_g_score' in name  or 'MLP_ita1Text' in name  or 'MLP_ita2Text' in name:
            param.requires_grad=True 
        else:
            param.requires_grad=False 
    print("Frozen parameters:")
    for name, param in adapterCLIP.named_parameters():
        if param.requires_grad == False: 
            print(name)
    print("Trained parameters:")
    for name, param in adapterCLIP.named_parameters():
        if param.requires_grad == True: 
            print(name)
else:
    raise ValueError("Do not support!!!")



if FreezeCase == 2:
    optimizer = torch.optim.Adam([
           {'params': filter(lambda p: p.requires_grad, adapterCLIP.MLP_g_score.parameters()), 'lr': args.lr},
           {'params': filter(lambda p: p.requires_grad, adapterCLIP.MLP_ita1Text.parameters()), 'lr': args.lr},
           {'params': filter(lambda p: p.requires_grad, adapterCLIP.MLP_ita2Text.parameters()), 'lr': args.lr},
            {'params': filter(lambda p: p.requires_grad, [adapterCLIP.positional_embedding]), 'lr': args.lr_CLIP},
            {'params': filter(lambda p: p.requires_grad,[ adapterCLIP.text_projection]), 'lr': args.lr_CLIP},
            {'params': filter(lambda p: p.requires_grad, [adapterCLIP.logit_scale]), 'lr': args.lr_CLIP},
            {'params': filter(lambda p: p.requires_grad, adapterCLIP.visual.parameters()), 'lr': args.lr_CLIP},
            {'params': filter(lambda p: p.requires_grad, adapterCLIP.transformer.parameters()), 'lr': args.lr_CLIP},
            {'params': filter(lambda p: p.requires_grad, adapterCLIP.token_embedding.parameters()), 'lr': args.lr_CLIP},
            {'params': filter(lambda p: p.requires_grad, adapterCLIP.ln_final.parameters()), 'lr': args.lr_CLIP}])
elif FreezeCase == 1 or FreezeCase == 0:
    optimizer = torch.optim.Adam([
            {'params': filter(lambda p: p.requires_grad, adapterCLIP.parameters()), 'lr': args.lr}])
else:
    raise ValueError("Do not support!!!")


lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=args.lr_patience,
                                                          verbose=True,min_lr=1e-9)
best_epoch = -1
best_val_loss = np.Inf
no_impv = 0
model_dir = osp.join(args.output_dir, 'checkpoints')
os.mkdir(model_dir) ## CHECK HERE

## Train.
for epoch in range(1, args.max_epochs + 1):
    train_loss = train_adapterCLIP(dataloaders['train'], accum_iter, adapterCLIP, optimizer, epoch, device)
    val_loss = val_adapterCLIP(dataloaders['val'], adapterCLIP, device) ## CHECK HERE
    lr_scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        out_name = osp.join(model_dir,  'best_model.pt')
        print(f"************************ Save best model at epoch = {best_epoch}, best_val_loss = {best_val_loss} ******************************")
        save_state_dicts(out_name, epoch, model=adapterCLIP, optimizer=optimizer, lr_scheduler=lr_scheduler)
        no_impv = 0
    else:
        no_impv += 1
    if no_impv == args.train_patience:
        break












