import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.color import gray2rgb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.transforms as transforms


#######################################################
############# CVC ClinicDB Dataset Loader #############
#######################################################

# Image Preprocessing Parameters
IMG_SIZE = (224, 224)

# Albumentations Transformations
train_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    ToTensorV2()
])

# Custom Dataset with skimage
class SegmentationDatasetCVC(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = imread(img_path).astype('float32') / 255.0  # shape: (H, W, 3) or (H, W)
        if image.ndim == 2:  # grayscale
            image = gray2rgb(image)  # make it 3 channels

        mask = imread(mask_path, as_gray=True).astype('float32') / 255.0  # shape: (H, W)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # (1, H, W)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return image, mask

def dataset_cvc(batch_size=4):
    """
    Load the CVC-ClinicDB dataset for segmentation tasks.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
    """
    # Define paths
    dataset_path = "datasets/CVC-ClinicDB"
    image_folder = os.path.join(dataset_path, "Original")
    mask_folder = os.path.join(dataset_path, "Ground Truth")

    # File listing
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.tif')])

    # Split dataset
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42)

    train_dataset = SegmentationDatasetCVC(
        [os.path.join(image_folder, f) for f in train_images],
        [os.path.join(mask_folder, f) for f in train_masks],
        transform=train_transform
    )

    test_dataset = SegmentationDatasetCVC(
        [os.path.join(image_folder, f) for f in test_images],
        [os.path.join(mask_folder, f) for f in test_masks],
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Check a sample
    x, y = next(iter(train_loader))
    print(f"Train batch shape: {x.shape}, {y.shape}")

    return train_loader, test_loader

#######################################################
################ Kvasir Dataset Loader ################
#######################################################

# Function to read and preprocess images
def load_image(img_path, mask_path):
    # Read image and mask
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale
    
    # Resize images and normalize
    image = cv2.resize(image, IMG_SIZE) / 255.0  # Normalize image
    mask = cv2.resize(mask, IMG_SIZE) / 255.0    # Normalize mask
    
    return image, mask


# Define PyTorch Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image, mask = load_image(img_path, mask_path)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        
        return image, mask

def dataset_kvasir(batch_size=4):
    """
    Load the Kvasir-SEG dataset for segmentation tasks.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
    """
    dataset_path = "datasets/Kvasir-SEG"
    image_folder = os.path.join(dataset_path, "images")
    mask_folder = os.path.join(dataset_path, "masks")
    # List all image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])

    # Split dataset into train (80%) and test (20%)
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42)

    train_dataset = SegmentationDataset(
        [os.path.join(image_folder, f) for f in train_images],
        [os.path.join(mask_folder, f) for f in train_masks]
    )

    test_dataset = SegmentationDataset(
        [os.path.join(image_folder, f) for f in test_images],
        [os.path.join(mask_folder, f) for f in test_masks]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_dataset_shape = next(iter(train_loader))[0].shape
    print(f"Shape of training dataset: {train_dataset_shape}")

    return train_loader, test_loader

#######################################################
######### Retina Blood Vessel Dataset Loader ##########
#######################################################

def dataset_retina(batch_size=4):
    """
    Load the Retina Blood Vessel dataset for segmentation tasks.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
    """
    train_dataset_path = "datasets/Retina-Blood-Vessel/train"
    train_image_folder = os.path.join(train_dataset_path, "image")
    train_mask_folder = os.path.join(train_dataset_path, "mask")
    test_dataset_path = "datasets/Retina-Blood-Vessel/test"
    test_image_folder = os.path.join(test_dataset_path, "image")
    test_mask_folder = os.path.join(test_dataset_path, "mask")

    # List all image files
    train_image_files = sorted([f for f in os.listdir(train_image_folder) if f.endswith('.png')])
    train_mask_files = sorted([f for f in os.listdir(train_mask_folder) if f.endswith('.png')])
    train_image_files = sorted([f for f in os.listdir(test_image_folder) if f.endswith('.png')])
    train_mask_files = sorted([f for f in os.listdir(test_mask_folder) if f.endswith('.png')])

    # Create datasets
    train_dataset = SegmentationDataset(
        [os.path.join(train_image_folder, f) for f in train_image_files],
        [os.path.join(train_mask_folder, f) for f in train_mask_files]
    )

    test_dataset = SegmentationDataset(
        [os.path.join(test_image_folder, f) for f in train_image_files],
        [os.path.join(test_mask_folder, f) for f in train_mask_files]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#######################################################
############## Brain Tumor Dataset Loader #############
#######################################################

def dataset_brain_tumor(batch_size=4):
    """
    Load the Brain Tumor dataset for segmentation tasks.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
    """
    dataset_path = "datasets/Brain-Tumor"
    image_folder = os.path.join(dataset_path, "images")
    mask_folder = os.path.join(dataset_path, "masks")
    
    # List all image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
    
    # Split dataset into train (90%) and test (10%)
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42)

    train_dataset = SegmentationDataset(
        [os.path.join(image_folder, f) for f in train_images],
        [os.path.join(mask_folder, f) for f in train_masks]
    )

    test_dataset = SegmentationDataset(
        [os.path.join(image_folder, f) for f in test_images],
        [os.path.join(mask_folder, f) for f in test_masks]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#######################################################
############## Skin Lesion Dataset Loader #############
#######################################################

def dataset_skin_lesion(batch_size=4):
    """
    Load the Skin Lesion dataset for segmentation tasks.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
    """
    dataset_path = "datasets/Skin-Lesion"
    image_folder = os.path.join(dataset_path, "images")
    mask_folder = os.path.join(dataset_path, "masks")
    
    # List all image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
    
    # Split dataset into train (90%) and test (10%)
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42)

    train_dataset = SegmentationDataset(
        [os.path.join(image_folder, f) for f in train_images],
        [os.path.join(mask_folder, f) for f in train_masks]
    )

    test_dataset = SegmentationDataset(
        [os.path.join(image_folder, f) for f in test_images],
        [os.path.join(mask_folder, f) for f in test_masks]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#######################################################
############## Breast Cancer Dataset Loader ###########
#######################################################

def dataset_breast_cancer(batch_size=4):
    """
    Load the Breast Cancer dataset for segmentation tasks.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
    """
    dataset_path = "datasets/Breast-Ultrasound-Images"
    benign_image_folder = os.path.join(dataset_path, "benign")
    malignant_image_folder = os.path.join(dataset_path, "malignant")

    # Collect (image, mask) pairs
    def collect_pairs(folder):
        image_files = [f for f in os.listdir(folder) if f.endswith('.png') and not f.endswith('_mask.png')]
        pairs = []
        for img in image_files:
            mask = img.replace('.png', '_mask.png')
            mask_path = os.path.join(folder, mask)
            img_path = os.path.join(folder, img)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
        return pairs

    benign_pairs = collect_pairs(benign_image_folder)
    malignant_pairs = collect_pairs(malignant_image_folder)
    all_pairs = benign_pairs + malignant_pairs

    # Split into train/test
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.1, random_state=42)

    train_dataset = SegmentationDataset(
        [img for img, _ in train_pairs],
        [mask for _, mask in train_pairs]
    )
    test_dataset = SegmentationDataset(
        [img for img, _ in test_pairs],
        [mask for _, mask in test_pairs]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
