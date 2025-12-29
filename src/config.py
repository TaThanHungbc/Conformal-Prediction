import torch

DATA_DIR = "data"
TRAIN_DIR = "data/train/train"
TEST_DIR = "data/test/test"
SAMPLE_SUB = "data/sampleSubmission.csv"

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0005
ALPHA = 0.05  # Reliability 95%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "outputs/fruit_resnet18"

MODEL_FILE_NAME = "fruit_resnet18_1.pth"

MODEL_DIR = "models"
QHAT_FILE = "qhats1.txt"
