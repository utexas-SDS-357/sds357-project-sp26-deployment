#!/usr/bin/env python3
"""Generate XGBoost PDP figure for slide."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Load & prep (mirrors notebook)
df = pd.read_csv('./data/sopp_svi_merged.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
time_parsed = pd.to_datetime(df['time'], errors='coerce')
df['hour'] = time_parsed.dt.hour

df['search_conducted'] = (df['search_conducted'] == True) | (df['search_conducted'] == 'True')
df['y'] = df['search_conducted'].astype(int)

K = 6
top_reasons = df['reason_for_stop'].value_counts(dropna=True).head(K).index.tolist()
df['reason_for_stop'] = df['reason_for_stop'].where(df['reason_for_stop'].isin(top_reasons), 'Other')

model_cols = ['subject_age', 'subject_race', 'subject_sex', 'reason_for_stop', 'service_area',
              'year', 'month', 'day_of_week', 'hour', 'svi_rpl_themes']
mask = df[model_cols + ['y']].notna().all(axis=1)
df_cc = df.loc[mask].copy()
# Use 50k sample for faster PDP computation
if len(df_cc) > 50000:
    df_cc = df_cc.sample(50000, random_state=42)

cat_cols = ['subject_race', 'subject_sex', 'reason_for_stop', 'service_area', 'day_of_week']
X = df_cc[model_cols].copy()
y = df_cc['y'].values
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
X = X.astype(float)

# Fit model (same params as notebook)
pos_rate = y.mean()
scale_pos_weight = (1 - pos_rate) / pos_rate
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss'
)
model.fit(X, y)

# PDP figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
features = ['subject_age', 'hour', 'svi_rpl_themes']
titles = ['Subject Age', 'Hour of Day', 'SVI (Vulnerability Index)']
for ax, feat, title in zip(axes, features, titles):
    idx = list(X.columns).index(feat)
    PartialDependenceDisplay.from_estimator(
        model, X, [idx], ax=ax,
        kind='average', grid_resolution=30
    )
    ax.set_ylabel('Avg predicted P(search)')
    ax.set_title(title)
    ax.set_xlabel(feat)
plt.suptitle('XGBoost: How Features Affect Predicted Search Probability', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('xgboost_pdp.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xgboost_pdp.png")
