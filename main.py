import os
from src.config import *
from src.dataset import get_loaders
from src.model import get_model
from src.train import train_model
from src.calibrate import get_qhat
from src.predict import run_test
from src.evaluate import evaluate_coverage

def main():
    # Tạo folder output
    if(not os.path.exists("outputs")):
        os.makedirs("outputs")

    # 1. Load Data
    train_loader, cal_loader, class_names = get_loaders()

    # 2. Build & Train Model
    model = get_model(len(class_names))
    train_model(model, train_loader)

    # 3. Conformal Prediction Calibration
    qhat = get_qhat(model, cal_loader)
    with open("outputs/qhat.txt", "w") as f:
        f.write(str(qhat))
    print(f"[INFO] Ngưỡng qhat xác định: {qhat:.4f}")

    # 4. Predict & Evaluate
    df_results = run_test(model, qhat, class_names)
    evaluate_coverage(df_results)

if __name__ == "__main__":
    main()