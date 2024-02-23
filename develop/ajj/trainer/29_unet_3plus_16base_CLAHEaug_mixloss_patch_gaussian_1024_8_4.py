# python native
import os
import json
import random
import datetime
from functools import partial
import wandb
from patchify import patchify

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import albumentations as A
# from UNet_Version.models.UNet_3Plus import UNet_3Plus 
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

# visualization
import matplotlib.pyplot as plt
import wandb

# 데이터 경로를 입력하세요
WAND_NAME = '29_unet_3plus_16base_CLAHEaug_mixloss_patch_gaussian_1024_8_4_2'
SAVE_PT_NAME = '_29_unet_3plus_16base_CLAHEaug_mixloss_patch_gaussian_1024_8_4_2.pt'

BATCH_SIZE_T = 4
BATCH_SIZE_V = 2

IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


LR = 1e-3
RANDOM_SEED = 21

NUM_EPOCHS = 100
VAL_EVERY = 5

SAVED_DIR = "save_dir"

if not os.path.exists(SAVED_DIR):                                                           
    os.makedirs(SAVED_DIR)

ignore_list = ['ID073','ID288','ID363','ID387','ID430','ID487','ID519','ID523','ID543']

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    if not any(ignore_folder in root for ignore_folder in ignore_list)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}


jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    if not any(ignore_folder in root for ignore_folder in ignore_list)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        wrist_pa_oblique = [f'ID{str(fname).zfill(3)}' for fname in range(274,320)]
        wrist_pa_oblique.append('ID321')
        y = [ 0 if os.path.dirname(fname) in wrist_pa_oblique else 1 for fname in _filenames]    

        sgkf = StratifiedGroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(sgkf.split(_filenames, y, groups)):
            if is_train:
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames) * 9
    
    def __getitem__(self, item):
        # image_name = self.filenames[item]
        # image_path = os.path.join(IMAGE_ROOT, image_name)

        image_index = item // 9
        patch_index = item % 9
        
        image_name = self.filenames[image_index]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path)
        
        label_name = self.labelnames[image_index]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        image_patches = patchify(image, (1024, 1024, 3), step=512)
        label_patches = patchify(label, (1024, 1024, len(CLASSES)), step=512)

        patch_row, patch_col = divmod(patch_index, 3)
        image_patch = image_patches[patch_row, patch_col, 0, :, :, :]
        label_patch = label_patches[patch_row, patch_col, 0, :, :, :]        

        if self.transforms is not None:
            transformed = self.transforms(image=image_patch, mask=label_patch)
            image_patch = transformed["image"]
            label_patch = transformed["mask"]

        image_patch = image_patch / 255.
        
        image_patch = image_patch.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label_patch = label_patch.transpose(2, 0, 1)
        
        image_patch = torch.from_numpy(image_patch).float()
        label_patch = torch.from_numpy(label_patch).float()
            
        return image_patch, label_patch
    
# 시각화를 위한 팔레트를 설정합니다.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

tf_1 = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.CLAHE(p=1.0),
                A.RandomBrightnessContrast(brightness_limit = 0.05, contrast_limit = 0.3, p=0.5),
                A.Rotate(10),
                ])
tf_2 = A.Compose([A.Resize(512, 512),
                  A.CLAHE(p=1.0)
                ])

train_dataset = XRayDataset(is_train=True, transforms=tf_1)
valid_dataset = XRayDataset(is_train=False, transforms=tf_2)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE_T,
    shuffle=True,
    num_workers=16,
    drop_last=True,
)

# 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=BATCH_SIZE_V,
    shuffle=False,
    num_workers=4,
    drop_last=False
)

# def focal_loss(inputs, targets, alpha=0.25, gamma=2):
#     BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#     # targets = targets.type(inputs.type())  # Ensuring same data type
#     PT = torch.exp(-BCE)  # Prevents computing the exponential of a large number which could result in inf or nan
#     focal_loss = alpha * (1-PT)**gamma * BCE
#     return focal_loss.mean()

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)  # Apply sigmoid to predict probabilities
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

def mix_loss(inputs, targets, weight=0.5):
    loss_of_dice = dice_loss(inputs, targets)
    BCE = nn.BCEWithLogitsLoss()
    # loss_of_focal = focal_loss(inputs, targets)
    loss_of_bce = BCE(inputs, targets)
    loss = loss_of_dice * weight + loss_of_bce * (1 - weight)
    return loss

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

def validation(epoch, model, data_loader, thr=0.5):
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
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            # loss = criterion(outputs, masks)
            # loss = dice_loss(outputs, masks)
            # total_loss += loss
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
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def train(model, data_loader, val_loader,  optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    scaler = GradScaler()
    wandb.init(entity='level2-cv-10-detection', project='yumin', name=WAND_NAME)
    
    for epoch in range(NUM_EPOCHS):
        if epoch % VAL_EVERY == 0:
            start_time = datetime.datetime.now()
        model.train()
        total_loss = 0.0
        total_steps = len(data_loader)
        
        for step, (images, masks) in enumerate(data_loader):   
                     
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            with torch.cuda.amp.autocast(): #fp16 연산
                outputs = model(images)
                # loss = criterion(outputs, masks)
                loss = mix_loss(outputs, masks)

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
        print(avg_loss)
        scheduler.step(avg_loss)
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader)
            print(f"current valid Dice: {dice:.4f}")

            time_gap = datetime.datetime.now() - start_time
            total_seconds = time_gap.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            print(f'{VAL_EVERY} epoch당 시간: {minutes}분 {seconds}초')

            wandb.log({'Validation Dice': dice})

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)
                
                


# model = models.segmentation.fcn_resnet50(pretrained=True)
model = UNet_3Plus(n_classes=len(CLASSES))


# output class 개수를 dataset에 맞도록 수정합니다.
# model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# Loss function을 정의합니다.
# criterion = nn.BCEWithLogitsLoss()

# Optimizer를 정의합니다.
optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 시드를 설정합니다.
set_seed()

train(model, train_loader, valid_loader, optimizer)

wandb.finish()