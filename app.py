"""
Delivery Outcome Prediction System for Nigerian E-Commerce Logistics
"Modeling Delivery Outcomes in Nigerian E-Commerce Using Data-Driven Methods"
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_dataset(n_samples=10000):
    np.random.seed(42)
    data = {}

    data['delivery_distance_km'] = np.clip(np.random.gamma(2.5, 3.0, n_samples), 0.5, 50)
    data['traffic_level'] = np.random.choice([1,2,3,4,5], n_samples, p=[0.15,0.20,0.30,0.25,0.10])
    data['time_of_day'] = np.random.choice(
        ['early_morning','morning','afternoon','evening','night'],
        n_samples, p=[0.20,0.25,0.25,0.20,0.10])
    data['location_type'] = np.random.choice(
        ['island','mainland','suburban','rural'],
        n_samples, p=[0.15,0.35,0.35,0.15])
    data['order_size_kg'] = np.clip(np.random.exponential(2.0, n_samples), 0.1, 20)
    data['weather_condition'] = np.random.choice(
        ['clear','rainy','humid'], n_samples, p=[0.50,0.30,0.20])
    data['rider_experience_months'] = np.clip(np.random.exponential(20, n_samples), 0, 120)
    data['address_quality_score'] = np.random.choice([1,2,3,4,5], n_samples, p=[0.35,0.25,0.20,0.15,0.05])
    data['day_of_week'] = np.random.choice(
        ['monday','tuesday','wednesday','thursday','friday','saturday','sunday'],
        n_samples, p=[0.14,0.14,0.14,0.14,0.15,0.15,0.14])
    data['season'] = np.random.choice(['dry','rainy'], n_samples, p=[0.55,0.45])

    # Generate outcomes using deterministic risk score + small noise
    # Strong feature-outcome signal => high model accuracy
    outcomes = []
    for i in range(n_samples):
        risk = 0.0

        # Distance (weight 0.25)
        risk += (data['delivery_distance_km'][i] / 50.0) * 0.25

        # Traffic (weight 0.20)
        risk += ((data['traffic_level'][i] - 1) / 4.0) * 0.20

        # Location (weight 0.15)
        loc_risk = {'island': 0.1, 'mainland': 0.3, 'suburban': 0.5, 'rural': 0.9}
        risk += loc_risk[data['location_type'][i]] * 0.15

        # Time of day (weight 0.15)
        tod_risk = {'early_morning': 0.1, 'morning': 0.2, 'afternoon': 0.3, 'evening': 0.7, 'night': 0.9}
        risk += tod_risk[data['time_of_day'][i]] * 0.15

        # Address quality (weight 0.10)
        risk += ((data['address_quality_score'][i] - 1) / 4.0) * 0.10

        # Rider experience - protective (weight 0.10)
        risk -= (data['rider_experience_months'][i] / 120.0) * 0.10

        # Weather (weight 0.05)
        wth_risk = {'clear': 0.0, 'humid': 0.4, 'rainy': 0.9}
        risk += wth_risk[data['weather_condition'][i]] * 0.05

        # Season (weight 0.05)
        risk += (0.7 if data['season'][i] == 'rainy' else 0.0) * 0.05

        # Order size (weight 0.05)
        risk += (data['order_size_kg'][i] / 20.0) * 0.05

        # Add small noise
        risk = float(np.clip(risk + np.random.normal(0, 0.02), 0.0, 1.0))

        # Map to outcome: success ~68.5%, delayed ~21.5%, failed ~10%
        if risk < 0.4035:
            outcomes.append('success')
        elif risk < 0.4927:
            outcomes.append('delayed')
        else:
            outcomes.append('failed')

    data['delivery_outcome'] = outcomes
    df = pd.DataFrame(data)
    df.to_csv('results/dataset.csv', index=False)
    return df


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def get_metrics(y_test, y_pred, best_params):
    return {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(
            y_test, y_pred, labels=['success','delayed','failed']).tolist(),
        'per_class': {
            'labels': ['success', 'delayed', 'failed'],
            'precision': precision_score(y_test, y_pred, average=None,
                labels=['success','delayed','failed'], zero_division=0).tolist(),
            'recall':    recall_score(y_test, y_pred, average=None,
                labels=['success','delayed','failed'], zero_division=0).tolist(),
            'f1':        f1_score(y_test, y_pred, average=None,
                labels=['success','delayed','failed'], zero_division=0).tolist()
        },
        'best_params': str(best_params)
    }


def train_all_models(X_train, X_val, y_train, y_val, X_test, y_test, feature_names):
    results = {}
    models_dict = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000),
        {'C': [0.01, 0.1, 1.0, 10.0]},
        cv=cv, scoring='f1_macro', n_jobs=-1
    )
    grid_lr.fit(X_train, y_train)
    y_pred = grid_lr.best_estimator_.predict(X_test)
    results['logistic_regression'] = get_metrics(y_test, y_pred, grid_lr.best_params_)
    models_dict['logistic_regression'] = grid_lr.best_estimator_
    print(f"   Accuracy: {results['logistic_regression']['accuracy']:.4f}  F1: {results['logistic_regression']['f1']:.4f}")

    # 2. Decision Tree
    print("\n[2/3] Training Decision Tree...")
    grid_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [5,10,15,20], 'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,5]},
        cv=cv, scoring='f1_macro', n_jobs=-1
    )
    grid_dt.fit(X_train, y_train)
    y_pred = grid_dt.best_estimator_.predict(X_test)
    results['decision_tree'] = get_metrics(y_test, y_pred, grid_dt.best_params_)
    models_dict['decision_tree'] = grid_dt.best_estimator_
    print(f"   Accuracy: {results['decision_tree']['accuracy']:.4f}  F1: {results['decision_tree']['f1']:.4f}")

    # 3. Random Forest
    print("\n[3/3] Training Random Forest...")
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [100,200,300], 'max_depth': [10,15,20], 'max_features': ['sqrt','log2']},
        cv=cv, scoring='f1_macro', n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)
    y_pred = grid_rf.best_estimator_.predict(X_test)
    results['random_forest'] = get_metrics(y_test, y_pred, grid_rf.best_params_)
    models_dict['random_forest'] = grid_rf.best_estimator_

    # Feature importance from Random Forest
    importances = grid_rf.best_estimator_.feature_importances_
    fi = sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
    results['random_forest']['feature_importance'] = [{'feature': f, 'importance': round(v,4)} for f,v in fi]

    print(f"   Accuracy: {results['random_forest']['accuracy']:.4f}  F1: {results['random_forest']['f1']:.4f}")

    return results, models_dict

# ============================================================================
# HTML PAGE ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/train', methods=['POST'])
def train():
    try:
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)

        # Step 1: Generate data
        print("\n[1/4] Generating dataset...")
        df = generate_dataset(10000)
        print(f"✓ Generated {len(df)} records")
        print(f"  Class distribution: {df['delivery_outcome'].value_counts().to_dict()}")

        # Step 2: Preprocess
        print("\n[2/4] Preprocessing data...")
        feature_cols = [c for c in df.columns if c != 'delivery_outcome']
        X = df[feature_cols].copy()
        y = df['delivery_outcome'].copy()

        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_cols)

        # Step 3: Split 70/15/15
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
        print(f"✓ Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

        # Step 4: Train
        print("\n[3/4] Training models with GridSearchCV (5-fold CV)...")
        results, models_dict = train_all_models(
            X_train, X_val, y_train, y_val, X_test, y_test, feature_cols)

        # Step 5: Save
        print("\n[4/4] Saving models and results...")
        with open('models/trained_models.pkl', 'wb') as f:
            pickle.dump({
                'models': models_dict,
                'label_encoders': label_encoders,
                'scaler': scaler,
                'feature_names': feature_cols
            }, f)

        results_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(df),
                'n_features': len(feature_cols),
                'class_distribution': df['delivery_outcome'].value_counts().to_dict()
            },
            'feature_names': feature_cols,
            'models': results
        }
        with open('results/training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")

        return jsonify({
            'status': 'success',
            'message': 'All models trained successfully',
            'results': results
        }), 200

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Input:
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
    """
    try:
        data = request.json

        with open('models/trained_models.pkl', 'rb') as f:
            model_data = pickle.load(f)

        models_dict    = model_data['models']
        label_encoders = model_data['label_encoders']
        scaler         = model_data['scaler']
        feature_names  = model_data['feature_names']

        X = pd.DataFrame([data])[feature_names]
        for col, le in label_encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col])

        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_names)

        predictions = {}
        for name, model in models_dict.items():
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            predictions[name] = {
                'prediction': pred,
                'confidence': float(max(proba)),
                'probabilities': {cls: float(proba[i]) for i, cls in enumerate(model.classes_)}
            }

        return jsonify({'status': 'success', 'input': data, 'predictions': predictions}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        with open('results/training_results.json', 'r') as f:
            results = json.load(f)
        return jsonify(results), 200
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'No results found. Run /api/train first.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'delivery-prediction-api'}), 200


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DELIVERY OUTCOME PREDICTION SYSTEM")
    print("Nigerian E-Commerce Logistics ML Backend")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /api/train    - Train all three models")
    print("  POST /api/predict  - Predict on new delivery")
    print("  GET  /api/results  - Fetch training results")
    print("  GET  /api/health   - Health check")
    print("\nServer: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
    