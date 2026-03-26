from fraud_detection.dataset import load_data, split_X_y
from fraud_detection.modeling.train import train_model
from fraud_detection.modeling.predict import evaluate

train, test = load_data()

X_train, y_train = split_X_y(train)
X_test,  y_test  = split_X_y(test)

model = train_model(X_train, y_train)

evaluate(model, X_test, y_test)