import pandas as pd
from fraud_detection.config import TRAIN_PATH, TEST_PATH, TARGET
from loguru import logger
from tqdm import tqdm


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    logger.success(f"Données chargées : train {train.shape} | test {test.shape}")
    return train, test

def split_X_y(df):
    X = df.drop([TARGET], axis=1)
    y= df[TARGET]

    return X, y