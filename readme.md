# Delivery Outcome Prediction System for Nigerian E-Commerce Logistics

## Overview

This software implements the machine learning research from the thesis: **"Modeling Delivery Outcomes in Nigerian E-Commerce Using Data-Driven Methods"**

The system trains three supervised classification models to predict delivery outcomes (Success, Delayed, Failed) based on operational and environmental factors in Nigerian e-commerce logistics.

## Project Purpose

- **Research Implementation**: Converts thesis methodology into working software
- **Model Training**: Trains Logistic Regression, Decision Tree, and Random Forest classifiers
- **Performance Evaluation**: Compares models using accuracy, precision, recall, F1-score, ROC-AUC
- **Feature Analysis**: Ranks variables by predictive importance
- **Operational Prediction**: Predicts delivery outcomes for new delivery records

## Dataset Characteristics

- **Size**: ~10,000 synthetic delivery records
- **Calibration**: Parameters based on 2020-2026 Nigerian logistics literature
- **Class Distribution**: 68.5% success, 21.5% delayed, 10% failed
- **Features**: 10 operational variables (distance, traffic, time, location, etc.)

## Features

### 1. Data Generation
- Generates synthetic delivery data calibrated to Nigerian conditions
- Realistic distributions for:
  - Distance (gamma distribution, median ~7.2km)
  - Traffic levels (1-5 scale)
  - Time of day (early morning, morning, afternoon, evening, night)
  - Location type (island, mainland, suburban, rural)
  - Order size, weather, rider experience, address quality, season

### 2. Data Preprocessing
- Missing value treatment (median for numeric, mode for categorical)
- Outlier detection (IQR and Z-score methods)
- Categorical encoding (one-hot for nominal, label encoding for ordinal)
- Feature standardization (StandardScaler)
- Train/validation/test split: 70%/15%/15% with stratification

### 3. Model Training
Three algorithms with hyperparameter tuning via GridSearchCV and 5-fold stratified cross-validation:

**Logistic Regression (Multinomial)**
- Baseline model, interpretable
- Hyperparameters: C (regularization), max_iter, solver

**Decision Tree**
- Captures non-linear relationships
- Hyperparameters: max_depth, min_samples_split, min_samples_leaf

**Random Forest**
- Ensemble method, best expected performance
- Hyperparameters: n_estimators, max_depth, max_features, bootstrap

### 4. Model Evaluation
Metrics computed for each model:
- **Accuracy**: Overall correctness
- **Precision (macro)**: Average precision across classes
- **Recall (macro)**: Average recall across classes
- **F1-Score (macro)**: Balanced precision-recall metric
- **ROC-AUC (macro)**: Area under ROC curve (One-vs-Rest)
- **Confusion Matrix**: Per-class classification breakdown
- **Per-class metrics**: Precision, recall, F1 for each outcome class

### 5. Feature Importance
- Extracts feature importance rankings from Random Forest
- Identifies top predictive variables
- Supports operational decision-making

### 6. Prediction
- API endpoint accepts new delivery record
- Returns predictions from all three models
- Includes confidence scores and class probabilities

## Data Features

| Feature | Type | Description | Expected Impact |
|---------|------|-------------|-----------------|
| delivery_distance_km | Numeric | Distance from depot to delivery address (km) | Positive with delay/failure |
| traffic_level | Ordinal | Traffic congestion level (1-5: light to severe) | Positive with delay/failure |
| time_of_day | Categorical | Delivery window (early morning, morning, afternoon, evening, night) | Non-linear; peak hours increase risk |
| location_type | Categorical | Area type (island, mainland, suburban, rural) | Rural increases failure risk |
| order_size_kg | Numeric | Package weight (kg) | Weak positive with delay |
| weather_condition | Categorical | Weather (clear, rainy, humid) | Rainy increases delay/failure |
| rider_experience_months | Numeric | Rider experience (months) | Negative with failure |
| address_quality_score | Ordinal | Address clarity (1: formal to 5: landmark-only) | Higher score increases failure risk |
| day_of_week | Categorical | Day (Monday-Sunday) | Weekends may differ from weekdays |
| season | Categorical | Season (dry or rainy) | Rainy increases delay/failure |

## Target Variable

**delivery_outcome** (3 classes):
- **success**: Order delivered within agreed time window
- **delayed**: Order delivered outside agreed time window
- **failed**: Order cannot be completed (address error, unreachable, returned, refused)

## File Structure

```
delivery-prediction-system/
├── app.py                    # Flask application (training, evaluation, prediction)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── scalers.pkl
│   └── label_encoders.pkl
├── results/                  # Training outputs
│   ├── training_results.json # Complete training metrics
│   ├── confusion_matrices.json
│   ├── feature_importance.json
│   └── dataset.csv          # Generated dataset
├── templates/               # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── training.html
│   ├── results.html
│   └── predict.html
└── static/                  # CSS and JavaScript
    ├── css/
    │   ├── style.css
    │   └── responsive.css
    └── js/
        ├── app.js
        ├── charts.js
        └── api.js
```

## API Endpoints

### Training
**POST /api/train**
- Generates dataset, trains all three models
- Returns: Training metrics, confusion matrices, feature importance
- Response: JSON with model performance comparison

### Prediction
**POST /api/predict**
- Input: New delivery record (JSON)
- Returns: Predictions from all three models with confidence scores
- Example input:
  ```json
  {
    "delivery_distance_km": 8.5,
    "traffic_level": 3,
    "time_of_day": "afternoon",
    "location_type": "mainland",
    "order_size_kg": 2.5,
    "weather_condition": "clear",
    "rider_experience_months": 24,
    "address_quality_score": 2,
    "day_of_week": "wednesday",
    "season": "dry"
  }
  ```

### Results
**GET /api/results**
- Fetches saved training results and metrics
- Returns: Complete evaluation data for all models

## Expected Results (Based on Thesis)

- **Logistic Regression**: ~72% accuracy, 0.66 F1-score
- **Decision Tree**: ~84% accuracy, 0.80 F1-score
- **Random Forest**: ~91% accuracy, 0.88 F1-score

Top 3 predictive features:
1. delivery_distance_km (22%)
2. traffic_level (18%)
3. time_of_day (15%)

## Technologies

- **Python 3.8+**
- **Flask**: Web framework
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **HTML/CSS/JavaScript**: Frontend

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Access web interface
# Open browser: http://localhost:5000
```

## Key Methodological Features

1. **Stratified Sampling**: Maintains class distribution across train/val/test sets
2. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
3. **Class Imbalance Handling**: Uses F1-macro as optimization metric, evaluates minority classes
4. **Feature Standardization**: StandardScaler for numeric features
5. **Categorical Encoding**: One-hot encoding for nominal, label encoding for ordinal variables
6. **Multi-metric Evaluation**: Not relying on accuracy alone; includes precision, recall, F1, AUC

## Limitations

- Uses simulated data calibrated to literature (not live operational data)
- Geographic focus: Lagos-centric conditions
- Temporal scope: 2020-2026 parameters
- Class imbalance: Failed deliveries (10%) underrepresented
- Three algorithms only (excludes XGBoost, neural networks)

## Research Alignment

This software directly implements the methodology described in:
- **Chapter 3**: Data preprocessing, feature engineering, model development, evaluation framework
- **Chapter 4**: Results presentation, model comparison, confusion matrices, feature importance
- **Thesis Scope**: Predictive modeling for Nigerian e-commerce last-mile delivery

## Author

Research-based implementation of thesis work on delivery outcome prediction in Nigerian e-commerce logistics.

## References

Primary literature cited in thesis (2020-2026):
- Küp et al. (2024): Real-time delivery delay prediction
- Sultana et al. (2025): ML models for delivery lead time forecasting
- Purnomo et al. (2025): Comparative ML for delivery delays
- Lesmana et al. (2025): Addressing class imbalance in delivery prediction
- Ampaw (2026): Last-mile logistics in Sub-Saharan Africa
- Ehimen et al. (2026): Last-mile delivery challenges in Lagos