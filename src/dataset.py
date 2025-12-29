from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import src.config as config

def get_loaders():
    # Augmentations for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Deterministic transform for calibration/validation
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset twice with same class ordering
    full_dataset_for_idxs = datasets.ImageFolder(root=config.TRAIN_DIR, transform=None)
    class_names = full_dataset_for_idxs.classes

    indices = np.arange(len(full_dataset_for_idxs))

    # Split indices into train / calib
    train_idx, cal_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=full_dataset_for_idxs.targets
    )

    # create two datasets with different transforms but same root/class mapping
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=transform_train)
    cal_dataset   = datasets.ImageFolder(root=config.TRAIN_DIR, transform=transform_eval)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    cal_loader = DataLoader(Subset(cal_dataset, cal_idx), batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, cal_loader, class_names