from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import src.config as config

def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=transform)
    class_names = full_dataset.classes

    # Chia Train (80%) và Calibration (20%)
    train_idx, cal_idx = train_test_split(
        np.arange(len(full_dataset)), 
        test_size=0.2, 
        random_state=42, 
        stratify=full_dataset.targets
    )

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    cal_loader = DataLoader(Subset(full_dataset, cal_idx), batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, cal_loader, class_names