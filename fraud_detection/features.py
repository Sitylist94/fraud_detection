from loguru import logger
from tqdm import tqdm
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb


def build_preprocessor():
    preprocessor = ColumnTransformer([
        ("num", "passthrough", make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object))
    ])
    return preprocessor