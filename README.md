# Research AI Explorer (UNC Assessment Prototype)

Interactive Flask dashboard for exploring multimodal biomedical study data (30 subjects × 2 visits), tuning a diagnosis support model, and reviewing EDA plus validation metrics—with **insights** for non-technical users.

## Quick start

```bash
cd "UNC Assessment"          # project root
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional: OPENAI_API_KEY for live AI captions
python data_generator.py      # creates sample_data.csv + saved model if missing
python app.py
```

- **Demo cohort (no upload):** http://127.0.0.1:5000/demo  
- **Upload your CSV:** http://127.0.0.1:5000/    

Set `ENABLE_LLM_INSIGHTS=false` in `.env` for faster reloads using built-in clinical fallbacks.

## Dashboard layout

### Data Visualization
| Chart | Purpose |
|-------|---------|
| Histograms | Sensor distributions by diagnosis |
| Bell curves | Spread + normal reference per sensor |
| **Visit comparison** | Mean sensors: first vs second visit by diagnosis |
| **Box plots** | Spread of each sensor: diagnosed vs undiagnosed |
| **Behavior bars** | Mean sensors by Stable / Watchlist / Elevated |
| Missing by visit | % missing per sensor per visit |
| Correlation heatmap | Which sensors move together |
| Label counts | Diagnosis & behavior counts per visit |
| Missingness table | % missing per column |

### Model Results
| Chart | Purpose |
|-------|---------|
| Metrics table | Accuracy, precision, recall, F1 |
| Confusion matrix | True vs predicted diagnosis |
| ROC curve | Separation (binary) |
| **Threshold sensitivity** | Agreement vs cutoff (parameter tradeoffs) |
| Confidence histogram | Prediction strength + cutoff line |
| Failure bar chart | Misclassifications & low-confidence visits |

## Labels (demo data)

| Field | Values | Meaning |
|-------|--------|---------|
| **diagnosis** | `Undiagnosed` | Comparison group (no study condition) |
| | `Diagnosed` | Meets study condition criteria |
| **behavior_label** | `Stable`, `Watchlist`, `Elevated` | Behavioral monitoring tier |
| **session** | `baseline`, `follow_up` | First / second visit |

## Sensors

| Column | Description |
|--------|-------------|
| `eeg_alpha_power` | EEG alpha band power |
| `eeg_theta_beta_ratio` | Theta/beta ratio (attention) |
| `hrv_rmssd_ms` | Heart-rate variability (RMSSD) |
| `sleep_efficiency_pct` | Sleep efficiency (%) |
| `gait_variability_index` | Gait variability index |

Legacy CSV columns (`feature1`, `Control`/`Patient`, `Healthy_Control`/`Clinical_Case`, session `A`/`B`) are mapped automatically via `labels.py`.

## Model behavior

- **All five sensors selected** → pretrained Random Forest (`models/biomedical_classifier.joblib`)
- **Sensor subset or custom upload** → retrained logistic regression with your imputation / C settings
- Adjust **threshold** slider to see sensitivity table and predictions update

## What is `data_generator.py`?

One-time setup (not run on every request):

1. Writes `sample_data.csv` (30 subjects, 2 visits, missingness)
2. Trains and saves `models/biomedical_classifier.joblib`

Re-run after changing the synthetic recipe: `python data_generator.py`

## Project layout

| File | Role |
|------|------|
| `app.py` | Flask routes, charts, dashboard context |
| `ml_model.py` | Pretrained RF + logistic retrain |
| `insights.py` | Plain-language chart captions (OpenAI or fallbacks) |
| `labels.py` | Display names + legacy CSV mapping |
| `data_generator.py` | Regenerate sample data + model |
| `templates/` | `index.html`, `dashboard.html` |
| `static/style.css` | Layout + section cards |

## Environment variables (`.env`)

```bash
OPENAI_API_KEY=sk-...          # optional
OPENAI_MODEL=gpt-4o-mini       # optional
ENABLE_LLM_INSIGHTS=true       # set false for offline fallbacks
SECRET_KEY=change-me           # Flask session
PORT=5000                      # optional
```

## Interview tips

- Lead with **Data Visualization** + “In plain terms” for clinical framing.  
- Use **Model Results** + threshold table for “no single agreed metric.”  
- Mention tradeoffs: removed PCA/heatmaps for clarity; kept multi-metric validation for researchers.  
- Point reviewers to `SUBMISSION_SUMMARY.md` for written reasoning.
