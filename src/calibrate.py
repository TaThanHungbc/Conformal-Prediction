import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import src.config as config

def get_qhat(model, cal_loader):
    model.eval()
    cal_scores = []
    print("[PROGRESS] Đang thực hiện Calibration...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(cal_loader, desc="Calculating Scores"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            probs = F.softmax(model(inputs), dim=1)
            # Lấy xác suất của nhãn đúng
            conf_correct = probs[torch.arange(len(labels)), labels].cpu().numpy()
            cal_scores.extend(1 - conf_correct)

    n = len(cal_loader.dataset)
    q_level = np.ceil((n + 1) * (1 - config.ALPHA)) / n
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    return qhat