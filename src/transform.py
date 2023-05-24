import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from dataset import ImageDataset

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

def get_train_dataset(train_path, training_data):
    train_dataset = ImageDataset(train_path, 
                        training_data["image_path"].tolist(), 
                        training_data["rotation_matrix"].tolist(), 
                        training_data["translation_vector"].tolist(), 
                        transforms=train_transform)
    return train_dataset


def get_validation_dataset(train_path, validation_data):    
    validation_dataset = ImageDataset(train_path, 
                        validation_data["image_path"].tolist(), 
                        validation_data["rotation_matrix"].tolist(), 
                        validation_data["translation_vector"].tolist(), 
                        transforms=validation_transform)
    return validation_dataset

def get_train_dataloader(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle=True, num_workers=2)
    return train_loader
    
def get_validation_dataloader(validation_dataset):
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
    return validation_loader