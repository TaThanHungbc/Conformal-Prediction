# Conformal-Prediction

This repository implements **Conformal Prediction for Classification** applied to deep learning models using PyTorch. The project was developed as a CS115 course project and focuses on uncertainty-aware classification using statistically valid prediction sets.

Conformal Prediction (CP) is a distribution-free framework that wraps around any classifier to produce prediction sets with coverage guarantees, allowing models to express uncertainty instead of returning only a single label.

---

## 📦 Project Overview

The project combines:

- A CNN-based image classifier (ResNet18).
- A calibration phase to compute conformal threshold `qhat`.
- A FastAPI backend for real-time prediction.
- Conformal Prediction logic to output prediction sets.
- Optional background removal using `rembg`.

---

## 📁 Project Structure

```
Conformal-Prediction/
│
├── src/                    # Core conformal prediction utilities
├── process.py              # FastAPI inference + CP service
├── main.py                 # Training / evaluation scripts
├── calibrate.py            # Compute qhat from calibration set
├── config.py               # Project configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Features

- CNN image classification with PyTorch.
- Conformal Prediction for multi-class classification.
- REST API powered by FastAPI.
- JSON outputs including:
  - Argmax prediction.
  - Full probability distribution.
  - Conformal prediction set.
  - qhat and threshold probability.
- Background removal for uploaded images.

---

## 🧠 Conformal Prediction

Conformal Prediction provides a statistically valid way to quantify uncertainty. Instead of outputting only one label, the model outputs a **set of labels** that is guaranteed (with probability ≥ 1 − α) to contain the true label under mild assumptions.

This project uses **Inductive Conformal Prediction (ICP)** with softmax probabilities from a CNN classifier.

---

## ⚙️ Installation

```bash
git clone https://github.com/TaThanHungbc/Conformal-Prediction
cd Conformal-Prediction
pip install -r requirements.txt
```

---

## 🏃 Running the Backend

```bash
uvicorn process:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```
http://localhost:8000/docs
```

---

## 📤 Example API Output

```json
{
  "argmax": {
    "label": "Pear",
    "confidence": 0.99
  },
  "conformal_set": [],
  "all_probs": [...],
  "qhat": 0.00007975,
  "threshold_prob": 0.99992025
}
```

- `all_probs` shows the sorted probability distribution.
- `conformal_set` contains labels satisfying the conformal threshold.
- An empty conformal set reflects a highly confident calibration regime, not an implementation error.

---

## References

- Wikipedia: Conformal Prediction  
  https://en.wikipedia.org/wiki/Conformal_prediction

---

## Author

Developed by **Bach Chan Hung** for academic study and experimentation with uncertainty-aware machine learning models.

---

## License

This project is for educational and research purposes.
