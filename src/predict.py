import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import src.config as config

def run_test(model, qhat, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_images = [f for f in os.listdir(config.TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    
    model.eval()
    print(f"[PROGRESS] Predicting on {len(test_images)} testing images...")
    
    with torch.no_grad():
        for img_name in tqdm(test_images):
            img_path = os.path.join(config.TEST_DIR, img_name)
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
            
            probs = F.softmax(model(img_tensor), dim=1).cpu().numpy()[0]
            pred_set_idx = np.where(probs >= (1 - qhat))[0]
            pred_set_names = [class_names[i] for i in pred_set_idx]
            
            results.append({
                "id": img_name,
                "label": class_names[np.argmax(probs)],
                "prediction_set": "|".join(pred_set_names),
                "set_size": len(pred_set_names)
            })
            
    df = pd.DataFrame(results)
    df[['id', 'label']].to_csv('outputs/submission.csv', index=False)
    df.to_csv('outputs/conformal_results.csv', index=False)
    return df