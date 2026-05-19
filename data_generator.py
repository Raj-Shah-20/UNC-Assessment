"""
Build the demo dataset and train the saved ML model.

Run manually:  python data_generator.py

This script is NOT part of the web request path. It:
  1. Writes sample_data.csv (30 subjects, 2 visits, multimodal sensors)
  2. Trains models/biomedical_classifier.joblib (Random Forest)

The Flask app calls the same logic on first startup if those files are missing.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Multimodal sensor / embedding-style measurements (meaningful names).
MODEL_FEATURES = [
    'eeg_alpha_power',           # EEG alpha band power (µV²)
    'eeg_theta_beta_ratio',      # Theta/beta ratio (attention regulation)
    'hrv_rmssd_ms',              # Heart-rate variability (RMSSD, ms)
    'sleep_efficiency_pct',      # Sleep efficiency (%)
    'gait_variability_index',    # Wearable gait variability (unitless index)
]

DIAGNOSIS_LABELS = {
    'Undiagnosed': 'No study diagnosis — comparison group',
    'Diagnosed': 'Meets study criteria for the target condition',
}

BEHAVIOR_LABELS = {
    'Stable': 'Behavior within expected range for group',
    'Watchlist': 'Mild behavioral shift — monitor across sessions',
    'Elevated': 'Marked behavioral change — review clinically',
}

SESSIONS = ['baseline', 'follow_up']
N_SUBJECTS = 30


def _subject_profile(diagnosis, rng):
    """Latent profile per subject (controls vs clinical cases)."""
    if diagnosis == 'Undiagnosed':
        return {
            'eeg_alpha_power': rng.normal(12.0, 1.2),
            'eeg_theta_beta_ratio': rng.normal(1.1, 0.15),
            'hrv_rmssd_ms': rng.normal(48.0, 6.0),
            'sleep_efficiency_pct': rng.normal(88.0, 4.0),
            'gait_variability_index': rng.normal(0.35, 0.06),
            'behavior_base': 0,
        }
    return {
        'eeg_alpha_power': rng.normal(9.5, 1.4),
        'eeg_theta_beta_ratio': rng.normal(1.65, 0.22),
        'hrv_rmssd_ms': rng.normal(32.0, 7.0),
        'sleep_efficiency_pct': rng.normal(74.0, 8.0),
        'gait_variability_index': rng.normal(0.58, 0.1),
        'behavior_base': rng.integers(1, 3),
    }


def generate_sample_dataframe(seed=42):
    """30 subjects × 2 sessions, diagnosis labels, realistic missingness."""
    rng = np.random.default_rng(seed)
    rows = []
    half = N_SUBJECTS // 2

    for subject_idx in range(1, N_SUBJECTS + 1):
        diagnosis = 'Undiagnosed' if subject_idx <= half else 'Diagnosed'
        profile = _subject_profile(diagnosis, rng)
        behavior_levels = ['Stable', 'Watchlist', 'Elevated']
        behavior_label = behavior_levels[min(profile['behavior_base'], 2)]

        for session in SESSIONS:
            session_shift = 0.08 if session == 'follow_up' else 0.0
            row = {
                'subject_id': f'SUBJ-{subject_idx:03d}',
                'session': session,
                'diagnosis': diagnosis,
                'behavior_label': behavior_label,
            }
            for feat in MODEL_FEATURES:
                value = profile[feat] + rng.normal(0, 0.12 * abs(profile[feat]) + 0.05)
                if feat == 'eeg_theta_beta_ratio':
                    value += session_shift
                elif feat == 'hrv_rmssd_ms':
                    value -= session_shift * 40
                elif feat == 'sleep_efficiency_pct' and session == 'follow_up':
                    value -= rng.uniform(0, 3) if diagnosis == 'Diagnosed' else rng.uniform(0, 1)
                row[feat] = round(float(max(value, 0.01)), 3)

            # Missingness: more gaps on follow-up and for clinical cases (~12–18% cells).
            miss_prob = 0.10 if session == 'baseline' else 0.16
            if diagnosis == 'Diagnosed':
                miss_prob += 0.04
            for feat in MODEL_FEATURES:
                if rng.random() < miss_prob:
                    row[feat] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def train_and_save_classifier(df, model_dir):
    """Train Random Forest on full demo cohort; save for inference."""
    os.makedirs(model_dir, exist_ok=True)

    complete = df.dropna(subset=MODEL_FEATURES + ['diagnosis'])
    encoder = LabelEncoder()
    encoder.fit(sorted(complete['diagnosis'].unique(), key=str))
    y = encoder.transform(complete['diagnosis'])

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
        )),
    ])
    X = complete[MODEL_FEATURES]
    pipeline.fit(X, y)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    bundle = {
        'pipeline': pipeline,
        'feature_columns': MODEL_FEATURES,
        'class_names': encoder.classes_.tolist(),
        'label_encoder': encoder,
        'diagnosis_glossary': DIAGNOSIS_LABELS,
        'behavior_glossary': BEHAVIOR_LABELS,
        'cv_accuracy_mean': float(cv_scores.mean()),
        'model_type': 'RandomForestClassifier',
        'trained_rows': len(complete),
    }

    path = os.path.join(model_dir, 'biomedical_classifier.joblib')
    joblib.dump(bundle, path)
    return bundle, path


def build_sample_dataset_and_model(base_dir):
    df = generate_sample_dataframe()
    csv_path = os.path.join(base_dir, 'sample_data.csv')
    df.to_csv(csv_path, index=False)
    bundle, model_path = train_and_save_classifier(
        df, os.path.join(base_dir, 'models'),
    )
    return df, csv_path, bundle, model_path


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    df, csv_path, bundle, model_path = build_sample_dataset_and_model(base)
    print(f'Wrote {csv_path} ({len(df)} rows)')
    print(f'Model saved to {model_path}')
    print(f'CV accuracy (5-fold): {bundle["cv_accuracy_mean"]:.1%}')
