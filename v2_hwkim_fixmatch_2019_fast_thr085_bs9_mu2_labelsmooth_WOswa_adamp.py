#!/usr/bin/env python
# coding: utf-8

# # TO-DO LIST 
# - Label Smoothing 
#     - https://www.kaggle.com/chocozzz/train-cassava-starter-using-label-smoothing
#     - https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
#     
# - Class Imbalance 
# 
# - SWA / SWAG 
# 
# - Augmentation 
#     - https://www.kaggle.com/sachinprabhu/pytorch-resnet50-snapmix-train-pipeline

# In[1]:


import os
print(os.listdir("./input/"))


# In[2]:


package_paths = [
    './input/pytorch-image-models/pytorch-image-models-master', #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
    './input/pytorch-gradual-warmup-lr-master'
]
import sys; 

for pth in package_paths:
    sys.path.append(pth)
    
# from warmup_scheduler import GradualWarmupScheduler


# In[3]:


from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm
from adamp import AdamP

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom


# In[4]:


CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 9,
    'valid_bs': 1,
    'T_0': 5,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'target_size' : 5,
    'smoothing' : 0.2,
    'swa_start_epoch' : 5,
    ## Following four are related to FixMatch
    'mu' : 2,
    'T' : 1,          # temperature
    'lambda_u' : 1.,
    'threshold' : 0.85,
    ##
    'debug' : False 
}


# In[5]:


train = pd.read_csv('./input/cassava-leaf-disease-classification/train.csv')
delete_id = ['2947932468.jpg', '2252529694.jpg', '2278017076.jpg']
train = train[~train['image_id'].isin(delete_id)].reset_index(drop=True)
train.head()


# > We could do stratified validation split in each fold to make each fold's train and validation set looks like the whole train set in target distributions.

# In[6]:


submission = pd.read_csv('./input/cassava-leaf-disease-classification/sample_submission.csv')
submission.head()


# # Helper Functions

# In[7]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb


# # Dataset

# In[8]:


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        
        self.output_label = output_label
        self.labels = self.df['label'].values

            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label == True:
            return img, target
        else:
            return img


# # Define Train\Validation Image Augmentations

# In[9]:


from albumentations.core.transforms_interface import DualTransform
# from albumentations.augmentations import functional as F
class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


# In[10]:


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return Compose([
            OneOf([
                Resize(CFG['img_size'], CFG['img_size'], p=1.),
                CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
                RandomResizedCrop(CFG['img_size'], CFG['img_size'], p=1.)
            ], p=1.), 
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            GridMask(num_grid=3, p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            OneOf([
                Resize(CFG['img_size'], CFG['img_size'], p=1.),
                CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
                RandomResizedCrop(CFG['img_size'], CFG['img_size'], p=1.)
            ], p=1.), 
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


# # Model

# In[11]:


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


# # For FixMatch Unlabeled DataLoader

# In[12]:


#######
o = os.listdir('./input/cassava-disease/all/')
o = np.array([o]).T
label_col = np.ones_like(o)
o = np.concatenate((o,label_col),axis=1)
unlabeled = pd.DataFrame(o,columns=['image_id','label'])
unlabeled.head()
# unlabeled = train


# In[13]:


import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


# In[14]:


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(CFG['img_size']*0.5))
        return img

def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=CFG['img_size'],
                                  padding=int(CFG['img_size']*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=CFG['img_size'],
                                  padding=int(CFG['img_size']*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CassavaDataset_ul(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        
        self.output_label = output_label
        self.labels = self.df['label'].values

            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = Image.open("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(img)
        
        if self.output_label == True:
            return img, target
        else:
            return img


# In[15]:


from torch.utils.data import RandomSampler

######################## 바꿔주자!!! 2019 데이터셋으로
# unlabeled_dataset = CassavaDataset_ul(unlabeled, './input/cassava-disease/all', transforms=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
unlabeled_dataset = CassavaDataset_ul(unlabeled, './input/cassava-disease/all/', transforms=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

train_loader_ul = torch.utils.data.DataLoader(
    unlabeled_dataset,
    sampler = RandomSampler(unlabeled_dataset),
    batch_size=CFG['train_bs'] * CFG['mu'],
    pin_memory=False,
    drop_last=True,     
    num_workers=CFG['num_workers'],
)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# train_loader_ul = iter(train_loader_ul)
# (inputs_u_w, inputs_u_s), _ = train_loader_ul.next()
# print(len(inputs_u_s), len(inputs_u_w))


# # Training APIs

# In[16]:


def prepare_dataloader(df, trn_idx, val_idx, data_root='./input/cassava-leaf-disease-classification/train_images/'):
    
    # from catalyst.data.sampler import BalanceClassSampler
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=True,###
        shuffle=True,        
        num_workers=CFG['num_workers'],
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, unlabeled_trainloader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in enumerate(train_loader):
        imgs = imgs.float()
        image_labels = image_labels.to(device).long()

        try:
            (inputs_u_s, inputs_u_w), _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_s, inputs_u_w), _ = unlabeled_iter.next()

        inputs = interleave(
                torch.cat((imgs, inputs_u_w, inputs_u_s)), 2*CFG['mu']+1).contiguous().to(device)

        with autocast():
            image_preds = model(inputs)   #output = model(input)
            logits = de_interleave(image_preds, 2*CFG['mu']+1)
            logits_x = logits[:CFG['train_bs']]
            logits_u_w, logits_u_s = logits[CFG['train_bs']:].chunk(2)
            del logits

            Lx = loss_fn(logits_x, image_labels)

            pseudo_label = torch.softmax(logits_u_w.detach()/CFG['T'], dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(CFG['threshold']).float()

#             Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            Lu = (loss_fn(logits_u_s, targets_u, reduction='none')*mask).mean()
            
            loss = Lx + CFG['lambda_u'] * Lu
            
            
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            # if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            #     description = f'epoch {epoch} loss: {running_loss:.4f}'
            #     print(description)
                # pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    # pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in enumerate(val_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        # if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
        #     description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
        #     pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('epoch = {}'.format(epoch+1), 'validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()
            
def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    # pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, (imgs, image_labels) in enumerate(data_loader):
            imgs = imgs.to(device).float()

            image_preds = model(imgs)   #output = model(input)
            image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


# In[17]:


# reference: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
class MyCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


# In[18]:


# ====================================================
# Label Smoothing
# ====================================================
class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
        
    def forward(self, pred, target, reduction = 'mean'): 
        pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        if reduction == 'mean':
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            return torch.sum(-true_dist * pred, dim=self.dim)


# # Main Loop

# In[19]:


from torchcontrib.optim import SWA
from sklearn.metrics import accuracy_score


# In[20]:


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # specify GPUs locally


# In[21]:


# #debug
# train = pd.read_csv('./input/cassava-leaf-disease-classification/train_debug.csv')
# CFG['epochs']=7
# CFG['model_path'] = 'temporary'
# !mkdir -p temporary


# In[1]:


# model_path='v2_hwkim_fixmatch_2019_fast_thr085_bs9_mu2_labelsmooth_WOswa_adamp'
# !mkdir -p v2_hwkim_fixmatch_2019_fast_thr085_bs9_mu2_labelsmooth_WOswa_adamp


# In[23]:


if __name__ == '__main__':
    for c in range(5): 
        train[c] = 0
        
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        print('Training with {} started'.format(fold))
        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root='./input/cassava-leaf-disease-classification/train_images/')
        unlabeled_trainloader = train_loader_ul
        
        device = torch.device(CFG['device'])

        model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)

        scaler = GradScaler()   
        optimizer = AdamP(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)

        loss_tr = LabelSmoothingLoss(classes=CFG['target_size'], smoothing=CFG['smoothing']).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, unlabeled_trainloader, device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
            # torch.save(model.state_dict(),'./model9_2/{}_fold_{}_{}_{}'.format(CFG['model_arch'], fold, epoch, seed)) 
        del train_loader, unlabeled_trainloader
        
        with torch.no_grad():
            valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
            torch.save(model.state_dict(),'./v2_hwkim_fixmatch_2019_fast_thr085_bs9_mu2_labelsmooth_WOswa_adamp/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)) 
        
        tst_preds = []
        for tta in range(5):
            tst_preds += [inference_one_epoch(model, val_loader, device)]
        
        train.loc[val_idx, [0, 1, 2, 3, 4]] = np.mean(tst_preds, axis=0)
        
        del model, optimizer, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
    
    train['pred'] = np.array(train[[0, 1, 2, 3, 4]]).argmax(axis=1)
    print(accuracy_score(train['label'].values, train['pred'].values))


# In[ ]:




