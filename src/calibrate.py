import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import src.config as config

# Minimal: set SOFTEN to 1 (no aggressive multiplication).
SOFTEN = 1.0

def optimize_temperature(logits_tensor, labels_tensor, init_logT=0.0, max_iter=200):
    """
    Optimize log(T) using LBFGS so that T = exp(logT) > 0.
    logits_tensor: torch.Tensor (n, K) on CPU
    labels_tensor: torch.LongTensor (n,)
    Returns scalar T (float)
    """
    logits = logits_tensor.double()
    labels = labels_tensor.long()

    # optimize logT for positivity
    logT = torch.tensor([init_logT], requires_grad=True, dtype=torch.float64)

    optimizer = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT)
        loss = F.cross_entropy(logits / (T + 1e-12), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        # if optimizer fails, fall back to initial T
        pass

    T_opt = float(torch.exp(logT).detach().cpu().numpy()[0])
    # clamp T to reasonable range to avoid extreme flattening or sharpening
    T_opt = max(1e-2, min(T_opt, 100.0))
    return T_opt


def get_qhat(model, cal_loader):
    """
    Calibrate using calibration loader.
    Returns qhat (float)
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
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    if len(logits_list) == 0:
        raise RuntimeError("Calibration loader empty or model produced no outputs.")

    logits_all = torch.cat(logits_list, dim=0)  # (n_cal, K)
    labels_all = torch.cat(labels_list, dim=0)  # (n_cal,)

    # Temperature scaling
    try:
        print("[PROGRESS] Tối ưu temperature bằng LBFGS (post-hoc calibration)...")
        T_opt = optimize_temperature(logits_all, labels_all, init_logT=0.0, max_iter=200)
        print(f"[INFO] Found temperature T = {T_opt:.4f}")
    except Exception as e:
        print(f"[WARN] Temperature optimization failed: {e} - sẽ dùng T=1.0")
        T_opt = 1.0

    # Apply small SOFTEN factor if you want, but default = 1.0 (no aggressive multiply)
    T_opt = float(T_opt * SOFTEN)
    print(f"[INFO] Final temperature T = {T_opt:.4f}")

    # compute scaled probabilities (numpy) using stable softmax
    with torch.no_grad():
        scaled_logits = (logits_all.double() / (float(T_opt) + 1e-12)).numpy()  # (n, K)
        ex = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs_all = ex / np.sum(ex, axis=1, keepdims=True)  # (n, K)

    # compute calibration scores s = 1 - p_true
    true_probs = probs_all[np.arange(len(labels_all)), labels_all.numpy()]
    cal_scores = (1.0 - true_probs).tolist()

    n = len(cal_scores)
    if n == 0:
        raise RuntimeError("No calibration samples found.")

    # standard conformal quantile index: k = ceil((n+1)*(1-alpha))
    q_level = np.ceil((n + 1) * (1 - config.ALPHA)) / n
    q_level = min(float(q_level), 1.0)  # clamp to [0,1]
    # numpy version differences: prefer method='higher' if available
    try:
        qhat = np.quantile(cal_scores, q_level, method='higher')
    except TypeError:
        # older numpy: fallback to interpolation param
        qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    print(f"[INFO] q_level={q_level:.6f}, qhat={qhat:.6g}")
    return float(qhat)
