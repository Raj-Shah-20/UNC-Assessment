import json
import os
import re

from openai import OpenAI

from labels import sensor_display_name

# Written ONLY for non-technical readers (nurses, coordinators, clinicians).
# No stats/ML vocabulary. No advice or to-do lists.

CLINICAL_FALLBACKS = {
    'study_overview': (
        'This page brings together brain, heart, sleep, and walking information from two study visits. '
        'It helps you see how people with and without the study diagnosis compare, and where readings are missing.'
    ),
    'classification_metrics': (
        'The computer summary usually matches the diagnosis already recorded for each visit. '
        'The study condition is showing up clearly in the body signals for many diagnosed people, and only a few visits look unclear.'
    ),
    'confusion_matrix': (
        'Most visits stay in the correct diagnosis group, with a small number counted in the other group by the computer. '
        'A few diagnosed people may look more like the undiagnosed group on their signals—or the other way around—when symptoms are mild or mixed.'
    ),
    'roc_curve': (
        'The line shows how well the computer separates diagnosed and undiagnosed people. '
        'When the line bends toward the top-left corner, the two groups look more distinct in this study.'
    ),
    'model_outputs': (
        'Bars show how confident the model is for each visit, with a red line at the chosen cutoff. '
        'Visits to the right of the line are more likely to be called diagnosed on sensors alone.'
    ),
    'model_misclassifications': (
        'Each row is a visit where the model diagnosis did not match the chart. '
        'These are the people worth a closer clinical look because the body signals and chart label disagreed.'
    ),
    'histogram': (
        'Bars compare typical sensor levels for diagnosed and undiagnosed people. '
        'Diagnosed people often show more sleep and heart strain than the undiagnosed comparison group.'
    ),
    'bell_curve': (
        'Curves show how spread out readings are for each group. '
        'Diagnosed people more often sit in the stressful tail of the curve, which fits greater day-to-day burden from the condition.'
    ),
    'visit_comparison': (
        'Bars compare average sensor levels on the first and second visit for each diagnosis group. '
        'Diagnosed people often look worse on the second visit, which fits the condition weighing on sleep, heart, and movement over time.'
    ),
    'sensor_boxplot': (
        'Boxes show the spread of each sensor for diagnosed vs undiagnosed people. '
        'Diagnosed people often sit lower on heart and sleep sensors with more spread, which fits uneven day-to-day strain.'
    ),
    'behavior_sensors': (
        'Bars compare average sensors across Stable, Watchlist, and Elevated behavior tiers. '
        'People in Elevated tiers often show weaker sleep and heart signals, which matches closer monitoring in the study.'
    ),
    'missing_by_visit': (
        'Bars compare how often each sensor is missing on the first vs second visit. '
        'Diagnosed people often have more gaps on the second visit, so their true burden may be harder to see.'
    ),
    'correlation_heatmap': (
        'Colors show which health signals tend to rise or fall together. '
        'When sleep and heart signals move together, poor sleep and body stress may be linked in the same diagnosed people.'
    ),
    'labels': (
        'Bars count how many people are in each diagnosis and behavior group per visit. '
        'More diagnosed people sit in watchful or elevated behavior tiers, which matches closer monitoring in the study.'
    ),
    'missingness_table': (
        'The table lists which types of readings are missing most often. '
        'Diagnosed people are more often missing attention and walking readings, so their struggle with the condition may be under-represented.'
    ),
    'threshold_sensitivity': (
        'The table shows how results change when the confidence cutoff moves up or down. '
        'A stricter cutoff catches fewer diagnosed cases but makes fewer mistakes, while a looser cutoff flags more people as diagnosed.'
    ),
}

CHART_GUIDE = {
    'study_overview': 'Whole study at a glance.',
    'classification_metrics': 'How often the computer agrees with recorded diagnoses.',
    'confusion_matrix': 'Mix-ups between diagnosed and undiagnosed groups.',
    'roc_curve': 'Separation between the two diagnosis groups.',
    'model_outputs': 'How confident the model is across visits.',
    'model_misclassifications': 'Visits where model and chart diagnosis disagreed.',
    'histogram': 'Typical sensor levels by diagnosis group.',
    'bell_curve': 'Spread of readings by group.',
    'visit_comparison': 'Average sensors on visit 1 vs visit 2 by diagnosis.',
    'sensor_boxplot': 'Spread of each sensor by diagnosis group.',
    'behavior_sensors': 'Average sensors by behavior tier.',
    'missing_by_visit': 'Missing readings on first vs second visit.',
    'correlation_heatmap': 'Signals that move together.',
    'labels': 'Counts by diagnosis and behavior.',
    'missingness_table': 'Which readings are missing most.',
    'threshold_sensitivity': 'How results change when the confidence cutoff moves.',
}

# Words/phrases non-technical readers should never see
BANNED_TERMS = re.compile(
    r'\b('
    r'PCA|AUC|ROC|F1|ML|AI model|classifier|algorithm|dataset|cohort|matrix|'
    r'histogram|embedding|facet|imputation|regularization|precision|recall|'
    r'ablation|augmentation|refinement|retrain|off-diagonal|hyperparameter|'
    r'failure case|accuracy\s*0\.|support tool|predicted|true vs|'
    r'implication|next step|recommend|suggest|should|must|need to|'
    r'inspect|report|validate|prioritize|consider|implement|perform|'
    r'follow-up clinically|clinical review|collect again|tuning|'
    r'eeg_|hrv_|gait_|theta_beta|rmssd|inactive|subject_id'
    r')\b',
    re.I,
)

SNAKE_CASE = re.compile(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', re.I)

ACTION_LEAD = re.compile(
    r'^(Next|Then|Also|Consider|Inspect|Report|Run|Review|Check|'
    r'Prioritize|Validate|Ensure|Try|Use|Avoid|Focus on|Look at)\b',
    re.I,
)

MEANS_PREFIX = re.compile(r'^(In plain terms|That means|This means)[,:]?\s*', re.I)


def get_openai_client():
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _top_missing_sensors(df, columns, n=3):
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        pct = float(df[col].isna().mean() * 100)
        if pct > 0:
            rows.append((sensor_display_name(col), round(pct)))
    rows.sort(key=lambda x: -x[1])
    return rows[:n]


def _missingness_by_group(df, sensor_cols):
    if 'diagnosis' not in df.columns or not sensor_cols:
        return None
    out = {}
    for label in df['diagnosis'].dropna().unique():
        sub = df[df['diagnosis'] == label]
        avg_miss = round(float(sub[sensor_cols].isna().mean().mean() * 100), 1)
        out[str(label)] = avg_miss
    return out


def _sensor_comparison(df, sensor_cols):
    if 'diagnosis' not in df.columns or len(sensor_cols) < 1:
        return []
    groups = df.groupby('diagnosis')
    if 'Diagnosed' not in groups.groups or 'Undiagnosed' not in groups.groups:
        return []
    diag = groups.get_group('Diagnosed')
    ctrl = groups.get_group('Undiagnosed')
    facts = []
    for col in sensor_cols[:5]:
        d = diag[col].dropna()
        u = ctrl[col].dropna()
        if len(d) < 2 or len(u) < 2:
            continue
        d_mean, u_mean = float(d.mean()), float(u.mean())
        name = sensor_display_name(col)
        if d_mean < u_mean * 0.92:
            facts.append(f'{name} is usually lower in diagnosed people than in undiagnosed people.')
        elif d_mean > u_mean * 1.08:
            facts.append(f'{name} is usually higher in diagnosed people than in undiagnosed people.')
    return facts[:4]


def build_insight_payload(df, active_features, inactive_features, params, scored, model_info=None):
    sensor_cols = [c for c in active_features if c in df.columns]
    all_sensors = [c for c in dict.fromkeys(active_features + inactive_features) if c in df.columns]

    payload = {
        'people_in_study': int(df['subject_id'].nunique()) if 'subject_id' in df.columns else None,
        'visit_rows': len(df),
        'patterns_in_body_signals': _sensor_comparison(df, sensor_cols),
        'readings_often_missing': _top_missing_sensors(df, all_sensors),
        'missing_more_for_diagnosed': _missingness_by_group(df, all_sensors),
    }

    if 'diagnosis' in df.columns:
        payload['how_many_per_diagnosis'] = {
            str(k): int(v) for k, v in df['diagnosis'].value_counts().items()
        }

    if scored is not None and not scored.empty:
        agree = round(float(scored['model_correct'].mean()) * 100)
        payload['computer_agrees_with_chart'] = f'about {agree} percent of visits'

    return payload


def _reject_sentence(sentence):
    s = sentence.strip()
    if not s or len(s) < 12:
        return True
    if BANNED_TERMS.search(s):
        return True
    if SNAKE_CASE.search(s):
        return True
    if ACTION_LEAD.search(s):
        return True
    if re.search(r'=\s*\d|\b\d{2,}\b.*\b\d{2,}\b', s):
        return True
    return False


def _replace_technical_words(text):
    out = text
    replacements = {
        'eeg_alpha_power': 'EEG alpha power',
        'eeg_theta_beta_ratio': 'attention signals',
        'hrv_rmssd_ms': 'heart-rate variability',
        'sleep_efficiency_pct': 'sleep efficiency',
        'gait_variability_index': 'walking steadiness',
        'follow_up': 'second visit',
        'baseline': 'first visit',
        'support tool': 'computer summary',
        'undiagnosed comparison group': 'undiagnosed people',
        'diagnosed participants': 'diagnosed people',
    }
    for old, new in replacements.items():
        out = re.sub(re.escape(old), new, out, flags=re.I)
    return re.sub(r'\s+', ' ', out).strip()


def _clean_sentences(text, max_sentences=4):
    """Keep up to max_sentences valid sentences from a paragraph."""
    text = _replace_technical_words(text or '').strip()
    text = MEANS_PREFIX.sub('', text).strip()
    parts = [p.strip().rstrip('.') for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    kept = []
    for part in parts:
        sentence = part if part.endswith(('.', '!', '?')) else f'{part}.'
        if not _reject_sentence(sentence):
            kept.append(sentence.rstrip('.'))
        if len(kept) >= max_sentences:
            break
    return kept


def _assemble_insight(what_you_see, in_plain_terms):
    """Observation sentence(s) plus 3–4 plain-language sentences for the side panel."""
    see_parts = _clean_sentences(what_you_see, max_sentences=1)
    means_parts = _clean_sentences(in_plain_terms, max_sentences=4)

    see = see_parts[0] if see_parts else ''
    if not means_parts and not see:
        return ''
    if not see:
        return ' '.join(f'{s}.' if not s.endswith('.') else s for s in means_parts)
    if not means_parts:
        return f'{see}.'
    means_text = ' '.join(
        s if s.endswith(('.', '!', '?')) else f'{s}.'
        for s in means_parts
    )
    return f'{see}. {means_text}'


def _parse_chart_entry(raw_value):
    """Accept {"what_you_see","in_plain_terms"} or a legacy string."""
    if isinstance(raw_value, dict):
        return _assemble_insight(
            raw_value.get('what_you_see') or raw_value.get('see', ''),
            raw_value.get('in_plain_terms') or raw_value.get('means', ''),
        )
    if isinstance(raw_value, str):
        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', raw_value) if p.strip()]
        if len(parts) >= 2:
            return _assemble_insight(parts[0], ' '.join(parts[1:]))
        return _assemble_insight(raw_value, '')
    return ''


def generate_chart_insights(payload, visible_charts):
    visible = [c for c in visible_charts if c]
    if not visible:
        return {}, ''

    client = get_openai_client()
    if client is None:
        insights = {key: CLINICAL_FALLBACKS.get(key, '') for key in visible}
        return insights, CLINICAL_FALLBACKS['study_overview']

    api_keys = ['study_overview'] + visible

    schema_example = {
        'study_overview': {
            'what_you_see': 'Short plain description.',
            'in_plain_terms': 'How diagnosed vs undiagnosed people differ overall.',
        }
    }

    prompt = f"""AUDIENCE: You write ONLY for non-technical clinical staff (nurses, study coordinators, doctors).
They do NOT know statistics, machine learning, or programming. Write as if explaining to a colleague on a ward round.

FORBIDDEN (never write these ideas):
- Any advice, suggestion, next step, implication, or task ("inspect", "report", "recommend", "should", "consider", "next,")
- Any model tuning, data science, or research methods language
- Exact counts (e.g. "30 diagnosed"), decimals (0.867), column names, or sensor codes
- Words: accuracy, precision, recall, ROC, AUC, matrix, failure, predicted, cohort, dataset, algorithm

REQUIRED JSON SHAPE — each key maps to an object with exactly two string fields:
- "what_you_see": ONE short sentence (max 25 words) describing the chart visually in everyday language.
- "in_plain_terms": THREE or FOUR short sentences (about 60–100 words total) explaining what this
  suggests about PEOPLE — compare diagnosed vs undiagnosed, burden over visits, mild symptoms,
  or missing readings hiding struggle. Do NOT start with "Next" or give instructions.

Example value:
{json.dumps(schema_example, indent=2)}

Charts to write:
{json.dumps({k: CHART_GUIDE.get(k, k) for k in api_keys}, indent=2)}

Background (do not quote numbers or lists from this):
{json.dumps(payload, indent=2, default=str)}

Return JSON with exactly these top-level keys: {json.dumps(api_keys)}.
Each value must be {{"what_you_see": "...", "in_plain_terms": "..."}} only."""

    model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You write chart captions for non-technical hospital staff only. '
                        'You describe what charts show and what they mean for patients. '
                        'You never give tasks, advice, or research instructions. JSON only.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.2,
            response_format={'type': 'json_object'},
        )
        raw = json.loads(response.choices[0].message.content)

        overview = _parse_chart_entry(raw.get('study_overview'))
        if not overview:
            overview = CLINICAL_FALLBACKS['study_overview']

        insights = {}
        for key in visible:
            text = _parse_chart_entry(raw.get(key))
            insights[key] = text if text else CLINICAL_FALLBACKS.get(key, '')

        return insights, overview
    except Exception:
        return (
            {key: CLINICAL_FALLBACKS.get(key, '') for key in visible},
            CLINICAL_FALLBACKS['study_overview'],
        )
