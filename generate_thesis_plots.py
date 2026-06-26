"""
Thesis Plot Generator
Generates high-resolution academic figures (Figures 4, 5, 6, 7, and 8)
calibrated to the Caleb University research proposal.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Set academic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Poppins', 'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

# Ensure output directory exists
output_dir = 'results/plots'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# LOAD DATA AND PIPELINE
# ============================================================================

print("Loading dataset and pipeline...")
df = pd.read_csv('results/dataset.csv')

with open('models/trained_models.pkl', 'rb') as f:
    pipeline = pickle.load(f)

models_dict  = pipeline['models']
medians      = pipeline['medians']
modes        = pipeline['modes']
bounds       = pipeline['bounds']
encoder      = pipeline['encoder']
scaler       = pipeline['scaler']
nominal_cols = pipeline['nominal_cols']
numeric_cols = pipeline['numeric_cols']
feature_names = pipeline['feature_names']

# Re-run preprocessing pipeline to get the exact test set for plotting
from app import impute_missing_values, treat_outliers, engineer_features

df_imputed, _, _ = impute_missing_values(df, training_medians=medians, training_modes=modes)
df_treated, _ = treat_outliers(df_imputed, training_bounds=bounds)
df_features = engineer_features(df_treated)

y = df_features['delivery_outcome'].copy()
X_raw = df_features.drop(columns=['delivery_outcome'])

X_nominal_encoded = encoder.transform(X_raw[nominal_cols])
nominal_feature_names = encoder.get_feature_names_out(nominal_cols).tolist()
X_encoded_df = pd.DataFrame(X_nominal_encoded, columns=nominal_feature_names)
X_numeric_df = X_raw[numeric_cols].reset_index(drop=True)
X_final = pd.concat([X_numeric_df, X_encoded_df], axis=1)

X_scaled = scaler.transform(X_final)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Split 70/15/15 with stratification (using seed 42 to match training)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled_df, y, test_size=0.15, random_state=42, stratify=y)


# ============================================================================
# FIGURE 4: EXPLORATORY DATA ANALYSIS (2x2 Panel)
# ============================================================================
print("Generating Figure 4 (Exploratory Data Analysis)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4(a): Target outcome distribution
outcome_counts = df['delivery_outcome'].value_counts()
outcome_pcts = df['delivery_outcome'].value_counts(normalize=True) * 100
sns.barplot(x=outcome_counts.index, y=outcome_counts.values, ax=axes[0, 0], palette='Blues_r')
axes[0, 0].set_title("4(a): Target Outcome Distribution", fontweight='bold')
axes[0, 0].set_ylabel("Count (n)")
axes[0, 0].set_xlabel("Delivery Outcome")
for idx, p in enumerate(axes[0, 0].patches):
    val = outcome_counts.values[idx]
    pct = outcome_pcts.values[idx]
    axes[0, 0].annotate(f"{val}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                ha='center', va='center', color='white', fontweight='bold')

# 4(b): Distance distribution
sns.histplot(df['delivery_distance_km'], ax=axes[0, 1], kde=True, color='teal', bins=30)
axes[0, 1].set_title("4(b): Delivery Distance Distribution (Gamma)", fontweight='bold')
axes[0, 1].set_xlabel("Distance (km)")
axes[0, 1].set_ylabel("Density")

# 4(c): Success rates by time of day
time_order = ['early_morning', 'morning', 'afternoon', 'evening', 'night']
time_outcome = pd.crosstab(df['time_of_day'], df['delivery_outcome'], normalize='index') * 100
time_outcome = time_outcome.reindex(time_order)
time_outcome.plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#ff9999', '#ffcc99', '#99ff99'])
axes[1, 0].set_title("4(c): Outcome Rates by Time of Day", fontweight='bold')
axes[1, 0].set_xlabel("Time of Day")
axes[1, 0].set_ylabel("Percentage (%)")
axes[1, 0].legend(title="Outcome", loc='lower left')
axes[1, 0].tick_params(axis='x', rotation=30)

# 4(d): Success rates by location type
loc_order = ['island', 'mainland', 'suburban', 'rural']
loc_outcome = pd.crosstab(df['location_type'], df['delivery_outcome'], normalize='index') * 100
loc_outcome = loc_outcome.reindex(loc_order)
loc_outcome.plot(kind='bar', stacked=True, ax=axes[1, 1], color=['#ff9999', '#ffcc99', '#99ff99'])
axes[1, 1].set_title("4(d): Outcome Rates by Location Type", fontweight='bold')
axes[1, 1].set_xlabel("Location Type")
axes[1, 1].set_ylabel("Percentage (%)")
axes[1, 1].legend(title="Outcome", loc='lower left')
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{output_dir}/figure4_eda.png", dpi=300)
plt.close()


# ============================================================================
# FIGURE 5: CORRELATION MATRIX
# ============================================================================
print("Generating Figure 5 (Feature Correlation Matrix)...")
plt.figure(figsize=(10, 8))
# Compute correlation on numeric and ordinal features
corr_cols = [
    'delivery_distance_km', 'traffic_level', 'order_size_kg', 
    'rider_experience_months', 'address_quality_score',
    'distance_traffic_interaction', 'time_risk_category', 
    'zone_risk_score', 'seasonal_disruption_indicator'
]
corr_matrix = df_features[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Figure 5: Feature Correlation Matrix", fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/figure5_correlation.png", dpi=300)
plt.close()


# ============================================================================
# FIGURE 6: CONFUSION MATRICES (1x3 Panel)
# ============================================================================
print("Generating Figure 6 (Confusion Matrices)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ['success', 'delayed', 'failed']

for idx, (name, model) in enumerate(models_dict.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14})
    
    title_name = name.replace('_', ' ').title()
    axes[idx].set_title(f"({chr(97+idx)}) {title_name}", fontweight='bold', fontsize=14)
    axes[idx].set_ylabel("Actual Outcome", fontsize=12)
    axes[idx].set_xlabel("Predicted Outcome", fontsize=12)

plt.suptitle("Figure 6: Comparative Model Confusion Matrices", fontweight='bold', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig(f"{output_dir}/figure6_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# FIGURE 7: ROC CURVES
# ============================================================================
print("Generating Figure 7 (ROC Curves)...")
plt.figure(figsize=(10, 8))

# Binarize labels for multi-class ROC calculation
classes = ['success', 'delayed', 'failed']
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = len(classes)

colors = {'logistic_regression': 'blue', 'decision_tree': 'green', 'random_forest': 'red'}
linestyles = ['-', '--', ':']

for name, model in models_dict.items():
    y_score = model.predict_proba(X_test)
    
    # Compute macro-average ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Aggregate false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    macro_auc = auc(all_fpr, mean_tpr)
    
    model_title = name.replace('_', ' ').title()
    plt.plot(all_fpr, mean_tpr, color=colors[name], linewidth=2,
             label=f"{model_title} (Macro-AUC = {macro_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 7: Multi-Class ROC Curves (Macro-Average One-vs-Rest)', fontweight='bold', pad=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{output_dir}/figure7_roc_curves.png", dpi=300)
plt.close()


# ============================================================================
# FIGURE 8: FEATURE IMPORTANCE AND PERFORMANCE (1x2 Panel)
# ============================================================================
print("Generating Figure 8 (Feature Importance & Performance)...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 8(a): Feature Importances (Top 10)
rf_model = models_dict['random_forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
top_features = [feature_names[i].replace('_', ' ').title() for i in indices]
top_importances = importances[indices]

sns.barplot(x=top_importances, y=top_features, ax=axes[0], palette='viridis')
axes[0].set_title("8(a): Random Forest Top 10 Feature Importances", fontweight='bold')
axes[0].set_xlabel("Mean Decrease in Gini Impurity")
axes[0].set_ylabel("Features")

# 8(b): Performance Comparison
with open('results/training_results.json', 'r') as f:
    results_json = json.load(f)

metrics_data = []
for model_name, metrics in results_json['models'].items():
    metrics_data.append({
        'Model': model_name.replace('_', ' ').title(),
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1']
    })

df_metrics = pd.DataFrame(metrics_data)
df_melted = pd.melt(df_metrics, id_vars=['Model'], var_name='Metric', value_name='Score')

sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', ax=axes[1], palette='muted')
axes[1].set_title("8(b): Overall Model Performance Comparison", fontweight='bold')
axes[1].set_ylabel("Score (0 - 1)")
axes[1].set_ylim(0, 1.05)
axes[1].legend(loc='lower right')
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.savefig(f"{output_dir}/figure8_performance.png", dpi=300)
plt.close()

print("\nSUCCESS: All five academic figures have been generated and saved to results/plots/!")
