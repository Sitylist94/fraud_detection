from sklearn.pipeline import Pipeline
from fraud_detection.features import build_preprocessor
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from loguru import logger

def build_pipeline():
    preprocessor = build_preprocessor()

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(class_weight="balanced"))
    ])

    return model

def train_model(X_train, y_train):
    model = build_pipeline()

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
    logger.info(f"CV AUC : {scores.mean():.4f}")

    model.fit(X_train, y_train)

    return model