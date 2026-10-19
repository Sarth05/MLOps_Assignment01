import pandas as pd

from heart_ml.data import normalize_frame
from heart_ml.features import FEATURES, build_preprocessor, split_xy


def test_normalize_maps_multiclass_target_to_binary():
    raw = pd.DataFrame([[
        63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 2
    ], [
        55, 0, 2, 130, 250, 0, 1, 160, 0, 1.0, 2, 0, 3, 0
    ]])
    clean = normalize_frame(raw)
    assert clean["target"].tolist() == [1, 0]
    assert clean.shape[1] == 14


def test_missing_values_are_preserved_for_pipeline_imputation():
    raw = pd.DataFrame([[
        63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, "?", 0, 6, 2
    ]])
    clean = normalize_frame(raw)
    assert pd.isna(clean.loc[0, "slope"])


def test_split_xy_validates_schema():
    frame = pd.DataFrame([{**{feature: 1 for feature in FEATURES}, "target": 0}])
    x, y = split_xy(frame)
    assert list(x.columns) == FEATURES
    assert y.iloc[0] == 0


def test_preprocessor_fits_small_frame():
    frame = pd.DataFrame([
        {"age": 55, "trestbps": 130, "chol": 220, "thalach": 150, "oldpeak": 1.0,
         "sex": 1, "cp": 2, "fbs": 0, "restecg": 1, "exang": 0, "slope": 2, "ca": 0, "thal": 3},
        {"age": 60, "trestbps": 140, "chol": 240, "thalach": 140, "oldpeak": 2.0,
         "sex": 0, "cp": 3, "fbs": 0, "restecg": 0, "exang": 1, "slope": 1, "ca": 1, "thal": 6},
    ])
    transformed = build_preprocessor().fit_transform(frame)
    assert transformed.shape[0] == 2
