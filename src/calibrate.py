# calibrate.py   (Thay thế toàn bộ file bằng nội dung này)
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import src.config as config

# giữ mềm nhẹ: default = 1.0 (không nhân mạnh)
SOFTEN = 1.0

def optimize_temperature(logits_tensor, labels_tensor, init_s=0.0, max_iter=200):
    """
    Optimize s where T = 1 + softplus(s) so that T >= 1.
    logits_tensor: torch.Tensor (n, K) on CPU
    labels_tensor: torch.LongTensor (n,)
    Returns scalar T (float)
    """
    logits = logits_tensor.double()
    labels = labels_tensor.long()

    # parameter s (unconstrained). T = 1 + softplus(s) ensures T >= 1.
    s = torch.tensor([init_s], requires_grad=True, dtype=torch.float64)

    optimizer = torch.optim.LBFGS([s], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = 1.0 + F.softplus(s)   # guaranteed >= 1
        loss = F.cross_entropy(logits / (T + 1e-12), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        # fallback: do nothing, keep s initial
        pass

    with torch.no_grad():
        T_opt = float((1.0 + F.softplus(s)).detach().cpu().numpy()[0])

    # clamp to a reasonable upper bound to avoid extreme flattening
    T_opt = max(1.0, min(T_opt, 100.0))
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

    # Temperature scaling (T >= 1 enforced)
    try:
        print("[PROGRESS] Tối ưu temperature bằng LBFGS (post-hoc calibration) với ràng buộc T>=1...")
        T_opt = optimize_temperature(logits_all, labels_all, init_s=0.0, max_iter=200)
        print(f"[INFO] Found temperature T = {T_opt:.4f}")
    except Exception as e:
        print(f"[WARN] Temperature optimization failed: {e} - sẽ dùng T=1.0")
        T_opt = 1.0

    # Apply optional small SOFTEN (default 1.0)
    T_opt = float(T_opt * SOFTEN)
    print(f"[INFO] Final temperature T = {T_opt:.4f}")

    # compute scaled probabilities (numpy) using stable softmax and clip to avoid exact 0/1
    with torch.no_grad():
        scaled_logits = (logits_all.double() / (float(T_opt) + 1e-12)).numpy()  # (n, K)
        ex = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs_all = ex / np.sum(ex, axis=1, keepdims=True)  # (n, K)

    # clip tiny numeric extremes
    eps = 1e-12
    probs_all = np.clip(probs_all, eps, 1.0 - eps)

    # compute calibration scores s = 1 - p_true
    true_probs = probs_all[np.arange(len(labels_all)), labels_all.numpy()]
    cal_scores = (1.0 - true_probs).tolist()

    n = len(cal_scores)
    if n == 0:
        raise RuntimeError("No calibration samples found.")

    # compute q_level and quantile robustly
    q_level = np.ceil((n + 1) * (1 - config.ALPHA)) / n
    q_level = min(float(q_level), 1.0)
    try:
        # numpy >= 1.22 uses 'method', older uses 'interpolation'
        qhat = np.quantile(cal_scores, q_level, method='higher')
    except TypeError:
        qhat = np.quantile(cal_scores, q_level, interpolation='higher')

    # avoid exact zero qhat (numerical safety): set small floor
    qhat = float(max(qhat, 1e-9))
    print(f"[INFO] q_level={q_level:.6f}, qhat={qhat:.6g}")
    return float(qhat)
