"""Load pretrained demo model and run inference with parameter tuning."""

import os

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'biomedical_classifier.joblib')

_bundle_cache = None


def get_model_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        if not os.path.isfile(MODEL_PATH):
            from data_generator import build_sample_dataset_and_model
            build_sample_dataset_and_model(os.path.dirname(__file__))
        _bundle_cache = joblib.load(MODEL_PATH)
    return _bundle_cache


def get_glossary():
    bundle = get_model_bundle()
    return {
        'diagnosis': bundle.get('diagnosis_glossary', {}),
        'behavior': bundle.get('behavior_glossary', {}),
        'features': {
            'eeg_alpha_power': 'EEG alpha band power — relaxation / idle cortex',
            'eeg_theta_beta_ratio': 'EEG theta/beta ratio — attention regulation marker',
            'hrv_rmssd_ms': 'Heart-rate variability (RMSSD) — autonomic balance',
            'sleep_efficiency_pct': 'Sleep efficiency from overnight monitoring',
            'gait_variability_index': 'Wearable gait variability during walking',
        },
    }


def _build_imputer(strategy):
    if strategy == 'zero':
        return SimpleImputer(strategy='constant', fill_value=0)
    if strategy in {'mean', 'median', 'most_frequent'}:
        return SimpleImputer(strategy=strategy)
    return SimpleImputer(strategy='median')


def predict_diagnosis(df, feature_cols, params):
    """
    Predict diagnosis using:
    - Pretrained Random Forest when all canonical features are selected.
    - Retuned Logistic Regression when the user selects a feature subset or custom upload.
    """
    if 'diagnosis' not in df.columns or not feature_cols:
        return None, None

    labels = df['diagnosis'].dropna().unique()
    if len(labels) < 2:
        return None, None

    bundle = get_model_bundle()
    encoder = LabelEncoder()
    encoder.fit(sorted(labels, key=str))

    canonical = bundle['feature_columns']
    use_pretrained = set(feature_cols) == set(canonical) and all(c in df.columns for c in canonical)

    y_series = df['diagnosis']
    valid = y_series.notna()

    if use_pretrained:
        pipeline = bundle['pipeline']
        X = df[canonical]
        valid = valid & np.isfinite(X.fillna(-999)).all(axis=1)  # allow imputer to handle NaN
        valid = y_series.notna()
        proba_all = pipeline.predict_proba(X)[:, 1]
        model_mode = f'Pretrained {bundle["model_type"]} (5-fold CV acc {bundle["cv_accuracy_mean"]:.0%})'
    else:
        imputer = _build_imputer(params.get('imputation', 'median'))
        X = imputer.fit_transform(df[feature_cols])
        valid = valid & np.isfinite(X).all(axis=1)
        if valid.sum() < 4:
            return None, None
        clf = LogisticRegression(C=params['regularization_c'], max_iter=2000)
        clf.fit(X[valid], encoder.transform(y_series[valid]))
        proba_all = clf.predict_proba(X)[:, 1]
        model_mode = f'Retrained LogisticRegression ({len(feature_cols)} sensors)'

    if valid.sum() < 4:
        return None, None

    threshold = params['decision_threshold']
    pred_idx = (proba_all >= threshold).astype(int)
    pred_labels = encoder.inverse_transform(pred_idx)

    out = df.copy()
    out['predicted_diagnosis'] = pred_labels
    out['prediction_probability'] = np.round(proba_all, 3)
    out['model_correct'] = out['diagnosis'] == out['predicted_diagnosis']
    out['failure_case'] = (~out['model_correct'].fillna(True)) | (
        (proba_all > threshold - 0.1) & (proba_all < threshold + 0.1)
    )

    model_info = {
        'y_true': encoder.transform(y_series[valid]),
        'y_pred': pred_idx[valid],
        'y_proba': proba_all[valid],
        'class_names': encoder.classes_.tolist(),
        'feature_cols': feature_cols,
        'model_mode': model_mode,
    }
    return out, model_info
