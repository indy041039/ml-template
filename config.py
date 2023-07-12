from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class Config:
    cat_cols = ['category']
    num_cols = ['amt','lat','long','merch_lat','merch_long']
    features_cols = cat_cols + num_cols
    target_col = ['is_fraud']

    num_pipeline= Pipeline(
        steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())

        ]
    )

    cat_pipeline=Pipeline(
        steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", OneHotEncoder()),
        ("scaler", StandardScaler(with_mean=False))
        ]

    )

    preprocessor=ColumnTransformer(
        [
        ("num_pipeline",num_pipeline, num_cols),
        ("cat_pipelines",cat_pipeline, cat_cols)
        ]
    )

    seed = 42
    save_path = 'artifact'
    test_size = 0.2
    models = {
        'random_forest': RandomForestClassifier(class_weight='balanced' ,random_state=42)
    }

