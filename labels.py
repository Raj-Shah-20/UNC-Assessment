"""Human-readable labels for sensors, diagnoses, and legacy CSV columns."""

# Old demo CSV column names → current sensor names
LEGACY_COLUMN_MAP = {
    'feature1': 'eeg_alpha_power',
    'feature2': 'eeg_theta_beta_ratio',
    'feature3': 'hrv_rmssd_ms',
    'feature4': 'sleep_efficiency_pct',
    'feature5': 'gait_variability_index',
}

LEGACY_VALUE_MAP = {
    'diagnosis': {
        'Control': 'Undiagnosed',
        'Patient': 'Diagnosed',
        'control': 'Undiagnosed',
        'patient': 'Diagnosed',
        'Healthy_Control': 'Undiagnosed',
        'Health_Control': 'Undiagnosed',
        'Clinical_Case': 'Diagnosed',
    },
    'behavior_label': {
        'Low-Risk': 'Stable',
        'Moderate-Risk': 'Watchlist',
        'High-Risk': 'Elevated',
        'low-risk': 'Stable',
        'moderate-risk': 'Watchlist',
        'high-risk': 'Elevated',
    },
    'session': {
        'A': 'baseline',
        'B': 'follow_up',
    },
}

SENSOR_DISPLAY = {
    'eeg_alpha_power': 'EEG alpha power',
    'eeg_theta_beta_ratio': 'EEG theta/beta ratio',
    'hrv_rmssd_ms': 'HRV (RMSSD)',
    'sleep_efficiency_pct': 'Sleep efficiency (%)',
    'gait_variability_index': 'Gait variability index',
}

DIAGNOSIS_DISPLAY = {
    'Undiagnosed': 'Undiagnosed',
    'Diagnosed': 'Diagnosed',
}


def sensor_display_name(column):
    return SENSOR_DISPLAY.get(column, column.replace('_', ' ').title())


def diagnosis_display_name(value):
    if value is None or (isinstance(value, float) and str(value) == 'nan'):
        return ''
    return DIAGNOSIS_DISPLAY.get(str(value), str(value).replace('_', ' '))


def apply_legacy_mappings(df):
    """Upgrade old uploads (feature1, Control/Patient, session A/B) to current schema."""
    out = df.copy()
    rename = {}
    for col in out.columns:
        key = col.strip().lower()
        if key in LEGACY_COLUMN_MAP and LEGACY_COLUMN_MAP[key] not in out.columns:
            rename[col] = LEGACY_COLUMN_MAP[key]
    out = out.rename(columns=rename)

    for col, mapping in LEGACY_VALUE_MAP.items():
        if col in out.columns:
            out[col] = out[col].replace(mapping)
    return out
