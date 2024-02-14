# python native
import os
import json
import random
import datetime
from functools import partial
import wandb

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import albumentations as A
from Unet_3Plus import UNet_3Plus

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import pickle

# visualization
import matplotlib.pyplot as plt

# 데이터 경로를 입력하세요
WAND_NAME = '18_unet_3plus_BC_fp16_ignore_RLROP_sgkf_hardaugV2_4_2_512_pkl'
SAVE_PT_NAME = '_18_unet_3plus_BC_fp16_ignore_RLROP_sgkf_hardaugV2_4_2_512_pkl.pt'

IMAGE_ROOT = "../../../data/train/DCM"
LABEL_ROOT = "../../../data/train/outputs_json"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
# CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
# IND2CLASS = {v: k for k, v in CLASS2IND.items()}

BATCH_SIZE_T = 4
BATCH_SIZE_V = 2
LR = 1e-4
RANDOM_SEED = 21

NUM_EPOCHS = 100
VAL_EVERY = 10

SAVED_DIR = "save_dir"

if not os.path.exists(SAVED_DIR):                                                           
    os.makedirs(SAVED_DIR)


class PklDataset(Dataset):
    def __init__(self, pickle_file = 'None'):
        with open(pickle_file, 'rb') as f:
            self.data_list = pickle.load(f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        image = self.data_list[item][0]
        label = self.data_list[item][1]

        return image, label

concept = 'hardaug'
train_pkl_path = os.path.join('../../../data/save_pkl', concept, 'train.pkl')
valid_pkl_path = os.path.join('../../../data/save_pkl', concept, 'valid.pkl')

train_dataset = PklDataset(train_pkl_path)
valid_dataset = PklDataset(valid_pkl_path)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE_T,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

# 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=BATCH_SIZE_V,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name=SAVE_PT_NAME):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            # outputs = model(images)['out']
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    scaler = GradScaler()
    wandb.init(entity='level2-cv-10-detection', project='yumin', name=WAND_NAME)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_steps = len(data_loader)
        
        for step, (images, masks) in enumerate(data_loader):   
                     
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            with torch.cuda.amp.autocast(): #fp16 연산
                outputs = model(images)
                loss = criterion(outputs, masks)

            # loss를 계산합니다.
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}, '
                    f'lr: {scheduler.optimizer.param_groups[0]["lr"]}'
                )
                wandb.log({'Train Loss': loss.item(),
                           'learning rate' : scheduler.optimizer.param_groups[0]['lr']})
        avg_loss = total_loss / total_steps
        scheduler.step(avg_loss)

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            print(f"current valid Dice: {dice:.4f}")
            wandb.log({'Validation Dice': dice})

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)
                
                



model = UNet_3Plus(n_classes=len(CLASSES))

# Loss function을 정의합니다.
criterion = nn.BCEWithLogitsLoss()

# Optimizer를 정의합니다.
optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)

# scheduler 주기
# T_max = 5

# 스케줄러 설정
# scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min = 1e-7)
# scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 시드를 설정합니다.
set_seed()

train(model, train_loader, valid_loader, criterion, optimizer)

wandb.finish()