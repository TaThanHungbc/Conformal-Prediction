from torchvision import models
import torch.nn as nn
import src.config as config

def get_model(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(config.DEVICE)