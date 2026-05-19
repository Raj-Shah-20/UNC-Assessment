import os
from io import StringIO

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, session
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from insights import CLINICAL_FALLBACKS, build_insight_payload, generate_chart_insights
from labels import (
    SENSOR_DISPLAY,
    apply_legacy_mappings,
    diagnosis_display_name,
    sensor_display_name,
)
from ml_model import get_glossary, predict_diagnosis

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'unc-research-prototype-key')

CORE_FIELDS = ['subject_id', 'session', 'diagnosis', 'behavior_label']

COLUMN_ALIASES = {
    'session_id': 'session',
    'sess': 'session',
    'subject': 'subject_id',
    'subjectid': 'subject_id',
    'subj_id': 'subject_id',
    'dx': 'diagnosis',
    'behavior': 'behavior_label',
    'behavioral_label': 'behavior_label',
    'behaviour_label': 'behavior_label',
    'behavioral': 'behavior_label',
}

DEFAULT_PARAMS = {
    'decision_threshold': 0.5,
    'regularization_c': 1.0,
    'imputation': 'mean',
}

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'sample_data.csv')


def normalize_dataframe(df):
    normalized = df.copy()
    rename_map = {}
    for col in normalized.columns:
        key = col.strip().lower().replace(' ', '_')
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    normalized = normalized.rename(columns=rename_map)

    if 'behavior_label' not in normalized.columns:
        for col in list(normalized.columns):
            name = col.lower()
            if col in CORE_FIELDS or col == 'diagnosis':
                continue
            if any(term in name for term in ['behavior', 'behaviour', 'behavioral']):
                normalized = normalized.rename(columns={col: 'behavior_label'})
                break

    return apply_legacy_mappings(normalized)


def get_core_hover_fields(df):
    return [field for field in CORE_FIELDS if field in df.columns]


def filter_hover_fields(plot_df, columns):
    return [col for col in columns if col in plot_df.columns]


def get_label_columns(df):
    candidates = []
    for col in df.columns:
        if col in CORE_FIELDS:
            continue
        name = col.lower()
        if name.endswith('_id') or name in {'id'}:
            continue
        n_unique = df[col].nunique(dropna=True)
        is_categorical = df[col].dtype == 'object' or str(df[col].dtype).startswith('category')
        is_label_like = any(term in name for term in ['diagnosis', 'label', 'behavior', 'group', 'class'])
        if is_label_like and 2 <= n_unique <= 20:
            candidates.append(col)
        elif is_categorical and 2 <= n_unique <= 12:
            candidates.append(col)
    return candidates


def get_numeric_feature_columns(df):
    excluded = set(CORE_FIELDS) | {'id'}
    return [
        col for col in df.select_dtypes(include='number').columns
        if col.lower() not in excluded and not col.lower().endswith('_id')
    ]


def get_selectable_columns(df):
    core = [c for c in CORE_FIELDS if c in df.columns]
    numeric = get_numeric_feature_columns(df)
    extra = [c for c in get_label_columns(df) if c not in core]
    return list(dict.fromkeys(core + numeric + extra))


def resolve_feature_columns(df, selected_columns):
    available = get_numeric_feature_columns(df)
    if not selected_columns:
        return available
    return [c for c in selected_columns if c in available]


def parse_params(form):
    return {
        'decision_threshold': float(form.get('decision_threshold', DEFAULT_PARAMS['decision_threshold'])),
        'regularization_c': float(form.get('regularization_c', DEFAULT_PARAMS['regularization_c'])),
        'imputation': form.get('imputation', DEFAULT_PARAMS['imputation']),
    }


def save_df_to_session(df):
    session['dataset'] = df.to_json(orient='split', date_format='iso')


def load_df_from_session():
    raw = session.get('dataset')
    if not raw:
        return None
    return normalize_dataframe(pd.read_json(StringIO(raw), orient='split'))


def _field_title(field_name):
    titles = {
        'diagnosis': 'Diagnosis group',
        'behavior_label': 'Behavioral status',
        'session': 'Visit',
    }
    return titles.get(field_name, field_name.replace('_', ' ').title())


BEHAVIOR_DISPLAY = {
    'Stable': 'Stable',
    'Watchlist': 'Watchlist',
    'Elevated': 'Elevated',
}

# Dashboard-wide palette - ONLY 4 colors
CHART_PRIMARY = '#2563eb'      # Blue - Primary/Undiagnosed/First visit/Stable
CHART_SECONDARY = '#dc2626'    # Red - Diagnosed/Elevated/Emphasis
CHART_TERTIARY = '#8b5cf6'     # Purple - Second visit/Watchlist
CHART_NEUTRAL = '#94a3b8'      # Gray - Reference/Muted

CHART_GRID = '#e2e8f0'
CHART_TEXT = '#334155'
CHART_TITLE = '#1e293b'

CHART_COLORWAY = [
    CHART_PRIMARY,
    CHART_SECONDARY,
    CHART_TERTIARY,
    CHART_NEUTRAL,
]

COLOR_MAPS = {
    'diagnosis': {'Undiagnosed': CHART_PRIMARY, 'Diagnosed': CHART_SECONDARY},
    'visit': {'First visit': CHART_PRIMARY, 'Second visit': CHART_TERTIARY},
    'behavior': {'Stable': CHART_PRIMARY, 'Watchlist': CHART_TERTIARY, 'Elevated': CHART_SECONDARY},
    'behavior_label': {'Stable': CHART_PRIMARY, 'Watchlist': CHART_TERTIARY, 'Elevated': CHART_SECONDARY},
}

CORRELATION_SCALE = [
    [0.0, CHART_PRIMARY],
    [0.5, '#f8fafc'],
    [1.0, CHART_SECONDARY],
]

CONFUSION_SCALE = [
    [0.0, '#eff6ff'],
    [0.5, '#93c5fd'],
    [1.0, CHART_PRIMARY],
]

DIAGNOSIS_ORDER = ['Undiagnosed', 'Diagnosed']


def _display_category(column, value):
    """Plain labels for axes, legends, and facet titles (no 'field=value')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    text = str(value).strip()
    if column in ('diagnosis', 'predicted_diagnosis'):
        return diagnosis_display_name(text) or text.replace('_', ' ')
    if column in ('behavior', 'behavior_label'):
        return BEHAVIOR_DISPLAY.get(text, text.replace('_', ' '))
    if column in ('session', 'visit'):
        return _visit_label(text)
    if column == 'sensor':
        if text in SENSOR_DISPLAY.values():
            return text
        return sensor_display_name(text)
    return text.replace('_', ' ')


def _format_plot_categories(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(lambda v, c=col: _display_category(c, v))
    return out


def _clean_facet_annotations(fig):
    """Plotly facets default to 'sensor=Name'; show only the value."""
    for ann in fig.layout.annotations or []:
        text = (ann.text or '').strip()
        if '=' in text:
            key, val = text.split('=', 1)
            ann.text = _display_category(key.strip(), val.strip())
    return fig


def _faceted_height(n_panels, wrap=3, row_px=340, pad=160):
    rows = max(1, (n_panels + wrap - 1) // wrap)
    return rows * row_px + pad


def _merge_layout_margin(fig, defaults):
    """Keep larger margins set by faceted charts (avoid clipping axis labels)."""
    cur = fig.layout.margin
    out = {}
    for side, default in defaults.items():
        existing = getattr(cur, side, None) if cur else None
        out[side] = max(default, int(existing) if existing is not None else 0)
    return out


def _fix_facet_xaxes(fig, *, bottom_margin=96):
    """Prevent overlapping numeric tick labels on faceted histograms/boxplots."""
    fig.update_xaxes(
        tickangle=-30,
        automargin=True,
        nticks=5,
        tickfont=dict(size=10),
    )
    cur = fig.layout.margin
    fig.update_layout(
        margin=dict(
            t=getattr(cur, 't', None) or 72,
            b=max(bottom_margin, getattr(cur, 'b', None) or 0),
            l=getattr(cur, 'l', None) or 52,
            r=getattr(cur, 'r', None) or 20,
        ),
    )
    return fig


def _apply_chart_style(fig, *, n_facets=0, facet_wrap=3, x_order=None):
    if n_facets:
        fig.update_layout(
            height=_faceted_height(n_facets, facet_wrap),
            margin=dict(t=80, b=100, l=60, r=30),
        )
        _fix_facet_xaxes(fig)
    else:
        fig.update_layout(margin=dict(t=70, b=60, l=60, r=30))
    if x_order:
        fig.update_xaxes(categoryorder='array', categoryarray=x_order)
    return _clean_facet_annotations(fig)


PLOTLY_CONFIG = {'displayModeBar': False, 'responsive': True, 'staticPlot': False}

LAYOUT_PRESETS = {
    'wide': {'min_height': 400},
    'medium': {'min_height': 380},
    'square': {'min_height': 400},
    'roc': {'min_height': 420},
}


def _apply_plot_theme(fig):
    """Shared fonts, grid, and default color cycle for every chart."""
    fig.update_layout(
        colorway=CHART_COLORWAY,
        font=dict(family='Arial, sans-serif', size=12, color=CHART_TEXT),
        title_font=dict(size=14, color=CHART_TITLE),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor=CHART_GRID,
            borderwidth=1,
            font=dict(color=CHART_TEXT),
        ),
    )
    fig.update_xaxes(
        gridcolor=CHART_GRID,
        linecolor=CHART_GRID,
        tickfont=dict(color=CHART_TEXT),
        title_font=dict(color=CHART_TITLE),
    )
    fig.update_yaxes(
        gridcolor=CHART_GRID,
        linecolor=CHART_GRID,
        tickfont=dict(color=CHART_TEXT),
        title_font=dict(color=CHART_TITLE),
    )
    return fig


def fig_to_html(fig, layout='wide'):
    """Export Plotly figure (page must load plotly.js once in dashboard.html)."""
    spec = LAYOUT_PRESETS.get(layout, LAYOUT_PRESETS['wide'])
    fig_h = fig.layout.height
    height = max(int(fig_h), spec['min_height']) if fig_h else spec['min_height']
    layout_kw = dict(
        height=height,
        autosize=True,
        margin=_merge_layout_margin(fig, dict(l=48, r=20, t=56, b=52)),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
    )
    fig.update_layout(**layout_kw)
    if layout == 'square':
        fig.update_yaxes(scaleanchor='x', scaleratio=1, constrain='domain')
        fig.update_xaxes(constrain='domain')
    _apply_plot_theme(fig)
    fig.update_traces(hoverlabel=dict(bgcolor='#ffffff', font_size=12))
    return fig.to_html(full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


def _color_column(df, label_columns=None):
    if 'diagnosis' in df.columns and df['diagnosis'].notna().any():
        return 'diagnosis'
    if 'behavior_label' in df.columns and df['behavior_label'].notna().any():
        return 'behavior_label'
    return None


def _melt_sensors(df, feature_cols, extra_id_vars=None):
    id_vars = [c for c in get_core_hover_fields(df) if c in df.columns]
    color_col = _color_column(df)
    if color_col and color_col not in id_vars:
        id_vars.append(color_col)
    if extra_id_vars:
        for col in extra_id_vars:
            if col in df.columns and col not in id_vars:
                id_vars.append(col)
    plot_df = df[id_vars + feature_cols].melt(
        id_vars=id_vars,
        value_vars=feature_cols,
        var_name='sensor',
        value_name='value',
    ).dropna(subset=['value'])
    if plot_df.empty:
        return None
    plot_df['sensor'] = plot_df['sensor'].map(sensor_display_name)
    return plot_df


def generate_histogram_plot(df, label_columns, feature_cols):
    if not feature_cols:
        return None
    plot_df = _melt_sensors(df, feature_cols)
    if plot_df is None:
        return None
    color_col = _color_column(df)
    hist_kwargs = dict(
        x='value',
        facet_col='sensor',
        facet_col_wrap=3,
        barmode='overlay',
        opacity=0.72,
        title='Distributions by diagnosis' if color_col == 'diagnosis' else 'Sensor distributions',
        hover_data=filter_hover_fields(plot_df, get_core_hover_fields(df)),
    )
    if color_col:
        hist_kwargs['color'] = color_col
        plot_df = _format_plot_categories(plot_df, [color_col])
        hist_kwargs['labels'] = {color_col: _field_title(color_col)}
        if color_col in COLOR_MAPS:
            hist_kwargs['color_discrete_map'] = COLOR_MAPS[color_col]
    n_facets = plot_df['sensor'].nunique()
    hist_kwargs['facet_col_spacing'] = 0.08
    hist_kwargs['facet_row_spacing'] = 0.20
    hist_kwargs['nbins'] = 16
    fig = px.histogram(plot_df, **hist_kwargs)
    fig.update_layout(bargap=0.06, legend_title_text='')
    fig.update_xaxes(matches=None, title_text='Reading')
    _apply_chart_style(fig, n_facets=n_facets, facet_wrap=3)
    return fig_to_html(fig, 'wide')


def generate_bell_curve_plot(df, label_columns, feature_cols):
    if not feature_cols:
        return None
    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    v_space = 0.24 if n_rows > 1 else 0.14
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[sensor_display_name(c) for c in feature_cols],
        vertical_spacing=v_space,
        horizontal_spacing=0.11,
    )
    for idx, sensor_col in enumerate(feature_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        values = df[sensor_col].dropna()
        if values.empty:
            continue
        fig.add_trace(
            go.Histogram(x=values, opacity=0.55, marker_color=CHART_PRIMARY, showlegend=False),
            row=row,
            col=col,
        )
        mean, std = values.mean(), values.std()
        if std > 0:
            x_range = np.linspace(values.min(), values.max(), 120)
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
            scale = len(values) * (values.max() - values.min()) / 20 if len(values) > 1 else len(values)
            fig.add_trace(
                go.Scatter(x=x_range, y=pdf * scale, mode='lines', line=dict(color=CHART_SECONDARY, width=2),
                           name='Normal fit', showlegend=idx == 0),
                row=row, col=col,
            )
    fig.update_layout(
        title_text='Sensor distributions with normal-curve reference',
        barmode='overlay',
        height=max(500, 340 * n_rows),
        margin=dict(t=80, b=70, l=60, r=30),
    )
    fig.update_annotations(font_size=11)
    return fig_to_html(fig, 'wide')


def _visit_label(session):
    if session == 'baseline':
        return 'First visit'
    if session == 'follow_up':
        return 'Second visit'
    return str(session).replace('_', ' ').title()


def generate_missing_by_visit_chart(df, feature_cols):
    """Bar chart: % missing per sensor, grouped by visit (easier than a subject grid)."""
    if not feature_cols:
        return None
    rows = []
    if 'session' in df.columns:
        for session in sorted(df['session'].dropna().unique(), key=str):
            sub = df[df['session'] == session]
            for col in feature_cols:
                rows.append({
                    'sensor': sensor_display_name(col),
                    'visit': _visit_label(session),
                    'missing_pct': round(float(sub[col].isna().mean() * 100), 1),
                })
        title = 'Missing readings by sensor and visit'
    else:
        for col in feature_cols:
            rows.append({
                'sensor': sensor_display_name(col),
                'visit': 'All visits',
                'missing_pct': round(float(df[col].isna().mean() * 100), 1),
            })
        title = 'Missing readings by sensor'
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None
    fig = px.bar(
        plot_df,
        x='sensor',
        y='missing_pct',
        color='visit',
        barmode='group',
        title=title,
        labels={'missing_pct': 'Missing (%)', 'sensor': 'Sensor', 'visit': 'Visit'},
        color_discrete_map=COLOR_MAPS['visit'],
    )
    fig.update_layout(xaxis_tickangle=-25, legend_title_text='')
    fig.update_traces(hovertemplate='Missing: %{y}%<br>%{x}<extra></extra>')
    _apply_chart_style(fig)
    return fig_to_html(fig, 'medium')


def generate_correlation_heatmap(df, feature_cols):
    if len(feature_cols) < 2:
        return None
    corr = df[feature_cols].corr()
    display = [sensor_display_name(c) for c in feature_cols]
    corr.index = display
    corr.columns = display
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale=CORRELATION_SCALE,
        zmin=-1,
        zmax=1,
        title='Sensor correlation matrix',
        labels=dict(color='Correlation'),
        aspect='auto',
    )
    fig.update_layout(margin=dict(t=56, b=80, l=100, r=24))
    fig.update_traces(hovertemplate='Correlation: %{z:.2f}<extra></extra>')
    return fig_to_html(fig, 'medium')


def generate_visit_comparison_chart(df, feature_cols):
    """Mean sensor levels by visit and diagnosis (longitudinal view)."""
    if not feature_cols or 'session' not in df.columns:
        return None
    rows = []
    for session in sorted(df['session'].dropna().unique(), key=str):
        sub = df[df['session'] == session]
        if 'diagnosis' in sub.columns:
            groups = sub.groupby('diagnosis')
        else:
            groups = [('All', sub)]
        for group_name, group_df in groups:
            for col in feature_cols:
                values = group_df[col].dropna()
                if values.empty:
                    continue
                rows.append({
                    'sensor': sensor_display_name(col),
                    'visit': _visit_label(session),
                    'diagnosis': str(group_name),
                    'mean_value': round(float(values.mean()), 2),
                })
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None
    plot_df = _format_plot_categories(plot_df, ['diagnosis', 'visit'])
    n_facets = plot_df['diagnosis'].nunique()
    fig = px.bar(
        plot_df,
        x='sensor',
        y='mean_value',
        color='visit',
        barmode='group',
        facet_col='diagnosis',
        facet_col_wrap=2,
        facet_col_spacing=0.10,
        facet_row_spacing=0.18,
        title='Average sensor levels: first vs second visit by diagnosis',
        labels={'mean_value': 'Average reading', 'sensor': 'Sensor', 'visit': 'Visit'},
        color_discrete_map=COLOR_MAPS['visit'],
    )
    fig.update_layout(xaxis_tickangle=-25, legend_title_text='')
    fig.update_traces(hovertemplate='Average: %{y}<br>%{x}<extra></extra>')
    _apply_chart_style(fig, n_facets=n_facets, facet_wrap=2)
    return fig_to_html(fig, 'wide')


def generate_sensor_boxplot(df, feature_cols):
    """Box plots of each sensor split by diagnosis."""
    if not feature_cols:
        return None
    plot_df = _melt_sensors(df, feature_cols)
    if plot_df is None:
        return None
    color_col = _color_column(df)
    if not color_col:
        return None
    plot_df = _format_plot_categories(plot_df, [color_col])
    n_facets = plot_df['sensor'].nunique()
    fig = px.box(
        plot_df,
        x=color_col,
        y='value',
        color=color_col,
        facet_col='sensor',
        facet_col_wrap=3,
        facet_col_spacing=0.08,
        facet_row_spacing=0.20,
        title=f'Sensor spread by {_field_title(color_col).lower()}',
        labels={'value': 'Reading', color_col: _field_title(color_col)},
        color_discrete_map=COLOR_MAPS.get(color_col),
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(hovertemplate='Reading: %{y}<extra></extra>')
    _apply_chart_style(
        fig,
        n_facets=n_facets,
        facet_wrap=3,
        x_order=DIAGNOSIS_ORDER if color_col == 'diagnosis' else None,
    )
    return fig_to_html(fig, 'wide')


def generate_behavior_sensor_chart(df, feature_cols):
    """Average sensor readings by behavioral monitoring tier."""
    if not feature_cols or 'behavior_label' not in df.columns:
        return None
    rows = []
    for behavior, group_df in df.groupby('behavior_label'):
        for col in feature_cols:
            values = group_df[col].dropna()
            if values.empty:
                continue
            rows.append({
                'sensor': sensor_display_name(col),
                'behavior': str(behavior),
                'mean_value': round(float(values.mean()), 2),
            })
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None
    plot_df = _format_plot_categories(plot_df, ['behavior'])
    fig = px.bar(
        plot_df,
        x='sensor',
        y='mean_value',
        color='behavior',
        barmode='group',
        title='Average sensor levels by behavioral status',
        labels={'mean_value': 'Average reading', 'sensor': 'Sensor', 'behavior': 'Behavior status'},
        color_discrete_map=COLOR_MAPS['behavior'],
    )
    fig.update_layout(xaxis_tickangle=-25, legend_title_text='')
    fig.update_traces(hovertemplate='Average: %{y}<br>%{x}<extra></extra>')
    _apply_chart_style(fig)
    return fig_to_html(fig, 'medium')


def generate_threshold_sensitivity_table(scored, model_info):
    """How agreement metrics change when the confidence cutoff moves."""
    if scored is None or model_info is None or 'prediction_probability' not in scored.columns:
        return None
    valid = scored['diagnosis'].notna()
    if valid.sum() < 4:
        return None

    y_labels = scored.loc[valid, 'diagnosis']
    proba = scored.loc[valid, 'prediction_probability'].astype(float)
    classes = sorted(y_labels.unique(), key=str)
    if len(classes) != 2:
        return None

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(classes)
    y_true = encoder.transform(y_labels)
    positive_idx = list(encoder.classes_).index('Diagnosed') if 'Diagnosed' in encoder.classes_ else 1

    rows = []
    for cutoff in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        y_pred = (proba >= cutoff).astype(int)
        rows.append({
            'Confidence cutoff': cutoff,
            'Agreement with chart (%)': round(float(accuracy_score(y_true, y_pred)) * 100, 1),
            'Diagnosed cases caught (%)': round(
                float(recall_score(y_true, y_pred, pos_label=positive_idx, zero_division=0)) * 100, 1,
            ),
        })

    highlight = min(rows, key=lambda r: abs(r['Confidence cutoff'] - 0.5))
    table_df = pd.DataFrame(rows)
    
    # Wrap in proper table structure to match other tables
    table_html = table_df.to_html(index=False, classes='dataframe', border=0)
    note = (
        f'<p class="threshold-note">Current cutoff is near {highlight["Confidence cutoff"]:.2f} '
        f'({highlight["Agreement with chart (%)"]}% agreement on this dataset).</p>'
    )
    
    return f'<div class="table-wrapper">{note}{table_html}</div>'


def generate_label_charts(df, label_columns):
    plots = []
    label_targets = list(dict.fromkeys(
        [c for c in ['diagnosis', 'behavior_label'] if c in df.columns]
        + [c for c in label_columns if c not in CORE_FIELDS]
    ))
    for label_col in label_targets:
        if label_col not in df.columns:
            continue
        if 'session' in df.columns:
            counts = df.groupby(['session', label_col], dropna=False).size().reset_index(name='records')
            counts = _format_plot_categories(counts, ['session', label_col])
            counts = counts.rename(columns={'session': 'visit'})
            fig = px.bar(
                counts, x='visit', y='records', color=label_col, barmode='group',
                title=f'{_field_title(label_col)} by visit',
                labels={'visit': 'Visit', label_col: _field_title(label_col), 'records': 'Visits'},
                color_discrete_map=COLOR_MAPS.get(label_col),
                hover_data=filter_hover_fields(counts, get_core_hover_fields(df)) or None,
            )
        else:
            counts = df[label_col].value_counts(dropna=False).reset_index()
            counts.columns = [label_col, 'records']
            counts = _format_plot_categories(counts, [label_col])
            fig = px.bar(
                counts, x=label_col, y='records', title=_field_title(label_col),
                labels={label_col: _field_title(label_col), 'records': 'Visits'},
                color_discrete_map=COLOR_MAPS.get(label_col),
            )
        fig.update_layout(legend_title_text='')
        _apply_chart_style(fig, x_order=DIAGNOSIS_ORDER if label_col == 'diagnosis' else None)
        plots.append(fig_to_html(fig, 'medium'))
    return '<br><br>'.join(plots) if plots else None


def run_model_inference(df, feature_cols, params):
    """Run pretrained RF (full feature set) or retrained logistic (custom subset)."""
    return predict_diagnosis(df, feature_cols, params)


def generate_classification_metrics_plots(model_info):
    """Confusion matrix, ROC/AUC (binary), and classification metrics table."""
    if not model_info:
        return None, None, None

    y_true = np.array(model_info['y_true'])
    y_pred = np.array(model_info['y_pred'])
    y_proba = np.array(model_info['y_proba'])
    classes = model_info['class_names']
    n_classes = len(classes)
    average = 'binary' if n_classes == 2 else 'macro'

    metric_rows = [
        ('Accuracy', accuracy_score(y_true, y_pred)),
        ('Precision', precision_score(y_true, y_pred, average=average, zero_division=0)),
        ('Recall', recall_score(y_true, y_pred, average=average, zero_division=0)),
        ('F1 Score', f1_score(y_true, y_pred, average=average, zero_division=0)),
    ]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    class_labels = [_display_category('diagnosis', c) for c in classes]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            text=cm,
            texttemplate='%{text}',
            colorscale=CONFUSION_SCALE,
            showscale=True,
            colorbar=dict(title='Count', len=0.6),
            xgap=2,
            ygap=2,
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        )
    )
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Model Prediction',
        yaxis_title='Chart Diagnosis',
        margin=dict(t=80, b=60, l=100, r=30),
    )

    roc_plot = None
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        metric_rows.append(('ROC AUC', roc_auc))
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color=CHART_PRIMARY, width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Chance',
            line=dict(dash='dash', color=CHART_NEUTRAL),
        ))
        fig_roc.update_traces(
            hovertemplate='%{fullData.name}<br>FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>',
        )
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
        )
        roc_plot = fig_to_html(fig_roc, 'square')

    metrics_table = pd.DataFrame([
        {
            'Metric': name,
            'Value': f'{value:.3f}' if isinstance(value, float) else str(value),
        }
        for name, value in metric_rows
    ]).to_html(index=False, border=0)

    return fig_to_html(fig_cm, 'square'), roc_plot, metrics_table


def generate_model_output_plots(df, feature_cols, params, scored=None, model_info=None):
    if scored is None or model_info is None:
        scored, model_info = run_model_inference(df, feature_cols, params)
    if scored is None:
        return None, None, None, None, None, None, None

    confidence_plot = None
    misclassification_plot = None

    scored_hist = _format_plot_categories(scored, ['diagnosis'])
    fig_prob = px.histogram(
        scored_hist,
        x='prediction_probability',
        color='diagnosis',
        barmode='overlay',
        opacity=0.7,
        nbins=20,
        title=f'Prediction confidence (cutoff {params["decision_threshold"]})',
        labels={'diagnosis': 'Chart diagnosis', 'prediction_probability': 'Confidence'},
        color_discrete_map=COLOR_MAPS['diagnosis'],
    )
    fig_prob.add_vline(x=params['decision_threshold'], line_dash='dash', line_color=CHART_SECONDARY)
    fig_prob.update_layout(legend_title_text='', bargap=0.04)
    fig_prob.update_traces(
        hovertemplate='Confidence: %{x:.2f}<br>Visits in bin: %{y}<extra></extra>',
    )
    _apply_chart_style(fig_prob)
    confidence_plot = fig_to_html(fig_prob, 'medium')

    wrong = scored[scored['model_correct'] == False].copy()  # noqa: E712
    if not wrong.empty and 'subject_id' in wrong.columns:
        wrong = wrong.sort_values('prediction_probability')
        wrong['participant'] = wrong['subject_id'].astype(str)
        visit_col = wrong['session'].map(_visit_label) if 'session' in wrong.columns else ''
        if 'session' in wrong.columns:
            wrong['participant'] = wrong['participant'] + ' · ' + visit_col.astype(str)
        hover_cols = ['subject_id', 'diagnosis', 'predicted_diagnosis']
        if 'session' in wrong.columns:
            hover_cols.insert(1, 'session')
        wrong = _format_plot_categories(wrong, ['diagnosis'])
        fig_fail = px.bar(
            wrong,
            y='participant',
            x='prediction_probability',
            orientation='h',
            color='diagnosis',
            title='Visits where the model disagreed with the chart diagnosis',
            labels={'diagnosis': 'Chart diagnosis', 'prediction_probability': 'Confidence'},
            color_discrete_map=COLOR_MAPS['diagnosis'],
        )
        custom = wrong[hover_cols].fillna('').astype(str).values
        if 'session' not in wrong.columns:
            hovertemplate = (
                'Participant: %{customdata[0]}<br>'
                'Chart diagnosis: %{customdata[1]}<br>'
                'Model guess: %{customdata[2]}<br>'
                'Confidence: %{x:.0%}<extra></extra>'
            )
        else:
            hovertemplate = (
                'Participant: %{customdata[0]}<br>'
                'Visit: %{customdata[1]}<br>'
                'Chart diagnosis: %{customdata[2]}<br>'
                'Model guess: %{customdata[3]}<br>'
                'Confidence: %{x:.0%}<extra></extra>'
            )
        fig_fail.update_traces(
            hovertemplate=hovertemplate,
            customdata=custom,
        )
        fig_fail.update_layout(
            yaxis_title='',
            xaxis_title='Model confidence',
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            legend_title_text='',
            height=max(320, 36 * len(wrong) + 80),
        )
        _apply_chart_style(fig_fail)
        misclassification_plot = fig_to_html(fig_fail, 'medium')

    confusion_plot, roc_plot, classification_metrics_table = generate_classification_metrics_plots(
        model_info,
    )

    return (
        confidence_plot,
        misclassification_plot,
        classification_metrics_table,
        confusion_plot,
        roc_plot,
        scored,
        model_info,
    )


def build_dashboard_context(df, selected_features, params):
    label_columns = get_label_columns(df)
    all_model_features = get_numeric_feature_columns(df)
    active_features = resolve_feature_columns(df, selected_features)
    if not active_features and all_model_features:
        active_features = all_model_features[:1]

    missing = df.isnull().mean() * 100
    field_labels = {
        'subject_id': 'Participant ID',
        'session': 'Visit',
        'diagnosis': 'Diagnosis',
        'behavior_label': 'Behavior status',
    }
    missing_rows = [
        {
            'Field': field_labels.get(col, sensor_display_name(col)),
            'Missing (%)': round(pct, 1),
        }
        for col, pct in missing.items()
    ]
    missingness_table = (
        pd.DataFrame(missing_rows)
        .sort_values(by='Missing (%)', ascending=False)
        .to_html(index=False)
    )

    confidence_plot = None
    misclassification_plot = None
    classification_metrics_table = None
    confusion_matrix_plot = None
    roc_plot = None
    model_message = None
    scored = None
    model_info = None

    if len(active_features) < 1:
        model_message = 'Enable at least one sensor modality to run predictions.'
    else:
        scored, model_info = run_model_inference(df, active_features, params)
        (
            confidence_plot,
            misclassification_plot,
            classification_metrics_table,
            confusion_matrix_plot,
            roc_plot,
            scored,
            model_info,
        ) = generate_model_output_plots(
            df, active_features, params, scored=scored, model_info=model_info,
        )

    histogram_plot = generate_histogram_plot(df, label_columns, active_features)
    bell_curve_plot = generate_bell_curve_plot(df, label_columns, active_features)
    visit_comparison_plot = generate_visit_comparison_chart(df, active_features)
    sensor_boxplot = generate_sensor_boxplot(df, active_features)
    behavior_sensor_plot = generate_behavior_sensor_chart(df, active_features)
    missing_by_visit_plot = generate_missing_by_visit_chart(df, active_features)
    correlation_heatmap = generate_correlation_heatmap(df, active_features)
    label_charts = generate_label_charts(df, label_columns)
    threshold_sensitivity_table = generate_threshold_sensitivity_table(scored, model_info)

    visible_charts = []
    if classification_metrics_table:
        visible_charts.append('classification_metrics')
    if confusion_matrix_plot:
        visible_charts.append('confusion_matrix')
    if roc_plot:
        visible_charts.append('roc_curve')
    if confidence_plot:
        visible_charts.append('model_outputs')
    if misclassification_plot:
        visible_charts.append('model_misclassifications')
    if histogram_plot:
        visible_charts.append('histogram')
    if bell_curve_plot:
        visible_charts.append('bell_curve')
    if visit_comparison_plot:
        visible_charts.append('visit_comparison')
    if sensor_boxplot:
        visible_charts.append('sensor_boxplot')
    if behavior_sensor_plot:
        visible_charts.append('behavior_sensors')
    if missing_by_visit_plot:
        visible_charts.append('missing_by_visit')
    if correlation_heatmap:
        visible_charts.append('correlation_heatmap')
    if label_charts:
        visible_charts.append('labels')
    if missingness_table:
        visible_charts.append('missingness_table')
    if threshold_sensitivity_table:
        visible_charts.append('threshold_sensitivity')
    insight_payload = build_insight_payload(
        df, active_features,
        [c for c in all_model_features if c not in active_features],
        params, scored, model_info,
    )
    enable_insights = os.environ.get('ENABLE_LLM_INSIGHTS', 'true').lower() in ('1', 'true', 'yes')
    if enable_insights:
        chart_insights, study_overview = generate_chart_insights(insight_payload, visible_charts)
    else:
        chart_insights = {k: CLINICAL_FALLBACKS.get(k, '') for k in visible_charts}
        study_overview = CLINICAL_FALLBACKS.get('study_overview', '')

    glossary = get_glossary()

    return dict(
        core_fields_present=[c for c in CORE_FIELDS if c in df.columns],
        model_feature_columns=all_model_features,
        active_features=active_features,
        inactive_features=[c for c in all_model_features if c not in active_features],
        sensor_display_names={c: sensor_display_name(c) for c in all_model_features},
        glossary=glossary,
        params=params,
        model_message=model_message,
        chart_insights=chart_insights,
        study_overview=study_overview,
        histogram_plot=histogram_plot,
        bell_curve_plot=bell_curve_plot,
        visit_comparison_plot=visit_comparison_plot,
        sensor_boxplot=sensor_boxplot,
        behavior_sensor_plot=behavior_sensor_plot,
        missing_by_visit_plot=missing_by_visit_plot,
        correlation_heatmap=correlation_heatmap,
        label_charts=label_charts,
        missingness_table=missingness_table,
        threshold_sensitivity_table=threshold_sensitivity_table,
        confidence_plot=confidence_plot,
        misclassification_plot=misclassification_plot,
        classification_metrics_table=classification_metrics_table,
        confusion_matrix_plot=confusion_matrix_plot,
        roc_plot=roc_plot,
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return {'status': 'ok', 'message': 'Research AI Explorer is running'}


@app.route('/demo')
def demo():
    """Load bundled sample data and open the dashboard (no upload required)."""
    if not os.path.isfile(SAMPLE_PATH):
        return redirect('/')
    df = normalize_dataframe(pd.read_csv(SAMPLE_PATH))
    save_df_to_session(df)
    session['selected_columns'] = get_numeric_feature_columns(df)
    session['params'] = DEFAULT_PARAMS.copy()
    return redirect('/dashboard')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    df = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            df = normalize_dataframe(pd.read_csv(file))
            save_df_to_session(df)
        else:
            df = load_df_from_session()
    else:
        df = load_df_from_session()

    if df is None:
        return redirect('/')

    if request.method == 'POST':
        selected_features = request.form.getlist('columns')
        params = parse_params(request.form)
    else:
        selected_features = session.get('selected_columns', get_numeric_feature_columns(df))
        params = {**DEFAULT_PARAMS, **session.get('params', {})}

    if not selected_features:
        selected_features = get_numeric_feature_columns(df)

    session['params'] = params
    session['selected_columns'] = selected_features

    ctx = build_dashboard_context(df, selected_features, params)
    numeric_cols = ctx.get('model_feature_columns') or get_numeric_feature_columns(df)
    ctx.setdefault(
        'sensor_display_names',
        {col: sensor_display_name(col) for col in numeric_cols},
    )
    return render_template('dashboard.html', **ctx)


def warm_start_pretrained_model():
    """Ensure sample CSV and saved Random Forest demo model exist."""
    from data_generator import build_sample_dataset_and_model

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'biomedical_classifier.joblib')
    if not os.path.isfile(SAMPLE_PATH) or not os.path.isfile(model_path):
        build_sample_dataset_and_model(os.path.dirname(__file__))


warm_start_pretrained_model()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Open http://127.0.0.1:{port}  —  or http://127.0.0.1:{port}/demo for sample data\n')
    app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)
