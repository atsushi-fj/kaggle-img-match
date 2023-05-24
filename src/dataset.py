import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import os 
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, img_path, rotation_label, translation_label, transforms=None):
        self.root = root
        self.img_path = img_path
        self.rotation_label = rotation_label
        self.translation_label = translation_label
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path[index]
        img_path = self.root + img_path
        image = cv2.imread(img_path)
        print(image)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
          
        rotation_label = self.rotation_label[index].split(";")
        rotation_label = list(map(float, rotation_label))
        
        translation_label = self.translation_label[index].split(";")
        translation_label = list(map(float, translation_label))
        
        return image, np.array(rotation_label), np.array(translation_label)
        
    def __len__(self):
        return len(self.img_path)

def get_datasets(root, path):
    file_path = os.path.join(root,path)
    df = pd.read_csv(file_path)
    return df

def get_train_validation_set(df):
    validtion_data = df.sample(frac=0.3)
    training_data =  df[~df["image_path"].isin(validtion_data["image_path"])]
    return training_data,validtion_data


def get_translation_rotation(df):
    df["rotation_matrix_split"] = df.apply(lambda x:list(map(float, x["rotation_matrix"].split(";"))), axis=1)
    df["translation_vector_split"] = df.apply(lambda x:list(map(float, x["translation_vector"].split(";"))), axis=1)
    rotation_value = np.array(df["rotation_matrix_split"].tolist())
    translation_value = np.array(df["translation_vector_split"].tolist())
    
    return translation_value,rotation_value

train_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
   ])

validation_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
    ])

def get_train_dataset(train_path,training_data):
    train_dataset = ImageDataset(train_path, 
                        training_data["image_path"].tolist(), 
                        training_data["rotation_matrix"].tolist(), 
                        training_data["translation_vector"].tolist(), 
                        transforms=train_transform)
    return train_dataset


def get_validation_dataset(train_path,validation_data):    
    validation_dataset = ImageDataset(train_path, 
                        validation_data["image_path"].tolist(), 
                        validation_data["rotation_matrix"].tolist(), 
                        validation_data["translation_vector"].tolist(), 
                        transforms=validation_transform)
    return validation_dataset

def get_train_dataloader(train_dataset):
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=os.cpu_count())
    return train_loader
    
def get_validation_dataloader(validation_dataset):
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=4,
                                   shuffle=False)
    return validation_loader


