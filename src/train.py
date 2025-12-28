import torch
import torch.nn.functional as F
from tqdm import tqdm
import src.config as config
import os

# Smoothing (0.05-0.2)
SMOOTHING = 0.1

def smooth_one_hot(labels, n_classes, smoothing=0.0, device='cpu'):
    """
    Trả về tensor shape (B, n_classes) chứa smoothed target distribution.
    smoothing: tổng lượng được phân phối sang các lớp khác (vd. 0.1)
    """
    assert 0.0 <= smoothing < 1.0
    with torch.no_grad():
        off_value = smoothing / float(n_classes - 1)
        on_value = 1.0 - smoothing
        labels_onehot = torch.full((labels.size(0), n_classes), off_value, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), on_value)
    return labels_onehot

def train_model(model, train_loader):
    # Try native label_smoothing first (newer PyTorch)
    try:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTHING)
        native_smoothing = True
    except TypeError:
        # older PyTorch: fallback to manual soft-label loss
        criterion = None
        native_smoothing = False

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    print(f"[PROGRESS] Bắt đầu Training...")
    for epoch in range(config.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)  # (B, K)

            if native_smoothing:
                loss = criterion(outputs, labels)
            else:
                # manual smoothed cross-entropy: - sum(p_smooth * log_softmax(outputs))
                n_classes = outputs.size(1)
                smoothed_targets = smooth_one_hot(labels, n_classes, smoothing=SMOOTHING, device=config.DEVICE)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = -(smoothed_targets * log_probs).sum(dim=1).mean()

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
    
    model_save_path = config.MODEL_SAVE_PATH + "_1"
    k = 1
    while(os.path.exists(model_save_path + ".pth")):
        k += 1
        model_save_path = config.MODEL_SAVE_PATH + "_" + str(k)
    
    model_save_path = model_save_path + ".pth"
    
    torch.save(model.state_dict(), model_save_path)
    print(f"[SUCCESS] Đã lưu model tại {model_save_path}")
