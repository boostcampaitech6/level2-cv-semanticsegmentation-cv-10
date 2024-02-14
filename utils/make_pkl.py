import json
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import psutil
# aug 컨셉 입력 [ 하위 폴더 명 설정 ]
concept = 'hardaug'
save_dir_pkl = os.path.join("../data/save_pkl", concept)

#원본 이미지 주소 입력
IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"

if not os.path.exists(save_dir_pkl):
    os.makedirs(save_dir_pkl)

# aug 정의
tf_1 = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                # A.CenterCrop(480, 480),
                # A.Resize(512, 512),
                # A.Resize(1024, 1024),
                # A.CenterCrop(980, 980),
                # A.Resize(1024, 1024),
                A.OneOf([A.OneOf([A.Blur(blur_limit = 4, always_apply = True),
                                    A.GlassBlur(sigma = 0.4, max_delta = 1, iterations = 2, always_apply = True),
                                    A.MedianBlur(blur_limit = 5, always_apply = True)], p=1),
                         A.RandomBrightnessContrast(brightness_limit = 0.05, contrast_limit = 0.3,always_apply = True),
                         A.CLAHE(p=1.0)], 
                         p=0.5),
                A.Rotate(10),
                ])
tf_2 = A.Resize(512, 512)
                


# ignore 할 건지 결정
# ignore_list = ['ID073','ID288','ID363','ID387','ID430','ID487','ID519','ID523','ID543']

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    # if not any(ignore_folder in root for ignore_folder in ignore_list) # 원본 쓸 거면 이 line 주석처리
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    # if not any(ignore_folder in root for ignore_folder in ignore_list) # 원본 쓸 거면 이 line 주석처리
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}


##################  여기까지가 기본 설정  #######################

def save_dataset_as_single_pickle(dataset, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    data_list = []

    for image, label in tqdm(dataset):
        data_list.append((image, label))

    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(data_list, f)

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

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    total_memory = psutil.virtual_memory().total
    used_memory_percent = (mem_info.rss / total_memory) * 100
    print(f'Current memory usage: {used_memory_percent:.2f}%')

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # ys = [0 for fname in _filenames]
        wrist_pa_oblique = [f'ID{str(fname).zfill(3)}' for fname in range(274,320)]
        wrist_pa_oblique.append('ID321')
        y = [ 0 if os.path.dirname(fname) in wrist_pa_oblique else 1 for fname in _filenames]    

        # gkf = GroupKFold(n_splits=5)
        sgkf = StratifiedGroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        # for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        for i, (x, y) in enumerate(sgkf.split(_filenames, y, groups)):
            if is_train:
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        print(f'{item} : 번째, {image_name} 이미지 처리 시작' )
        # print_memory_usage()
        image = cv2.imread(image_path)
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image / 255.
        
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        # print(f'{item} : 번째, {image_name} 이미지 처리 완료' )
        # print_memory_usage()
        return image, label

train_dataset = XRayDataset(is_train=True, transforms=tf_1)
valid_dataset = XRayDataset(is_train=False, transforms=tf_2)

# save_dataset_as_single_pickle(train_dataset, save_dir_pkl, 'train.pkl')
save_dataset_as_single_pickle(valid_dataset, save_dir_pkl, 'valid.pkl')