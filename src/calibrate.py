import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import src.config as config

def optimize_temperature(logits_tensor, labels_tensor, init_T=1.0, max_iter=50):
    """
    logits_tensor: torch.Tensor shape (n, K) on CPU
    labels_tensor: torch.LongTensor shape (n,)
    Returns optimal scalar T > 0
    """
    T = torch.tensor([init_T], requires_grad=True, dtype=torch.float64)
    logits = logits_tensor.double()
    labels = labels_tensor.long()

    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def loss_closure():
        optimizer.zero_grad()
        scaled = logits / T
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(loss_closure)
        T_opt = float(T.detach().cpu().numpy()[0])
    except Exception:
        T_opt = float(T.detach().cpu().numpy()[0])
    # ensure T >= 0.05
    return max(0.05, T_opt)

def get_qhat(model, cal_loader):
    """
    Calibrate using calibration loader.
    Steps:
     - collect logits and labels on calibration set
     - optimize temperature T on logits/labels (post-hoc calibration)
     - compute scaled probabilities = softmax(logits / T)
     - compute calibration scores s = 1 - p_true (using scaled probs)
     - compute qhat as before
    """
    model.eval()
    logits_list = []
    labels_list = []
    print("[PROGRESS] Đang thu thập logits & nhãn từ Calibration set...")

    with torch.no_grad():
        for inputs, labels in tqdm(cal_loader, desc="Collecting logits"):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            logits = model(inputs)  # shape (B, K)
            # move to CPU for safer numeric ops and to feed optimizer (which expects CPU tensors)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    if len(logits_list) == 0:
        raise RuntimeError("Calibration loader empty or model produced no outputs.")

    # concat along sample dimension
    logits_all = torch.cat(logits_list, dim=0)  # shape (n_cal, K)
    labels_all = torch.cat(labels_list, dim=0)  # shape (n_cal,)

    # Optional: run temperature scaling to improve calibration (minimal change)
    try:
        print("[PROGRESS] Tối ưu temperature bằng LBFGS (post-hoc calibration)...")
        T_opt = optimize_temperature(logits_all, labels_all, init_T=1.0, max_iter=50)
        print(f"[INFO] Found temperature T = {T_opt:.4f}")
    except Exception as e:
        print(f"[WARN] Temperature optimization failed: {e} - sẽ dùng T=1.0")
        T_opt = 1.0

    # compute scaled probabilities (numpy)
    with torch.no_grad():
        scaled_logits = (logits_all.double() / float(T_opt)).numpy()  # shape (n, K)
        # stable softmax (numpy)
        ex = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs_all = ex / np.sum(ex, axis=1, keepdims=True)  # (n, K)

    # compute calibration scores s = 1 - p_true (scaled)
    true_probs = probs_all[np.arange(len(labels_all)), labels_all.numpy()]
    cal_scores = (1.0 - true_probs).tolist()

    # compute qhat using existing formula
    n = len(cal_scores)
    if n == 0:
        raise RuntimeError("No calibration samples found.")
    q_level = np.ceil((n + 1) * (1 - config.ALPHA)) / n
    # use 'higher' interpolation (same as before)
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    print(f"[INFO] q_level={q_level:.6f}, qhat={qhat:.6g}")
    return float(qhat)
