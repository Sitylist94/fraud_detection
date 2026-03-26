from fraud_detection.modeling.train import build_pipeline
from sklearn.metrics import classification_report
from loguru import logger

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)
    logger.info(f"Classification report : {classification_report(y_test, y_pred)}")

    return y_pred
