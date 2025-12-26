import torch
from tqdm import tqdm
import src.config as config
import os

def train_model(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    print(f"[PROGRESS] Bắt đầu Training...")
    for epoch in range(config.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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