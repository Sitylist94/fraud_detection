from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH  = ROOT / "data" / "raw" / "test.csv"
TARGET = "is_fraud"
