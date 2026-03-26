from autogluon.tabular import TabularPredictor
from fraud_detection.dataset import load_data

train, test = load_data()

predictor = TabularPredictor.load("AutogluonModels/ag-20260325_185909")

# 1. Modèles entraînés
print("=== MODÈLES ===")
print(predictor.model_names())

# 2. Hyperparamètres de chaque modèle
print("\n=== HYPERPARAMÈTRES ===")
for model in predictor.model_names():
    print(f"\n{model}:")
    print(predictor.model_hyperparameters(model))

# 3. Feature importance
print("\n=== FEATURE IMPORTANCE ===")
importance = predictor.feature_importance(test)
print(importance)

# 4. Evaluation complète
print("\n=== EVALUATION ===")
print(predictor.evaluate(test))

# 5. Infos générales
print("\n=== INFO COMPLÈTE ===")
print(predictor.info())
