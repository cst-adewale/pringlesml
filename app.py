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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Thread-safe training state
training_state = {
    'status': 'idle',      # idle, training, success, error
    'progress': 0,         # 0 to 100
    'message': 'System is ready for training.',
    'error_message': None,
    'results': None
}
training_lock = threading.Lock()



# ============================================================================
# DATA GENERATION (CALIBRATED TO THESIS SPECIFICATIONS)
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

    # Generate continuous risk score
    risks = []
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
        # Add noise
        risk = float(np.clip(risk + np.random.normal(0, 0.02), 0.0, 1.0))
        risks.append(risk)

    risks = np.array(risks)
    
    # Recalibrate thresholds using percentiles to achieve exact thesis distribution:
    # 68.5% success, 21.5% delayed, 10.0% failed
    p68_5 = np.percentile(risks, 68.5)
    p90 = np.percentile(risks, 90.0)

    outcomes = []
    for r in risks:
        if r < p68_5:
            outcomes.append('success')
        elif r < p90:
            outcomes.append('delayed')
        else:
            outcomes.append('failed')

    data['delivery_outcome'] = outcomes
    df = pd.DataFrame(data)

    # Artificially introduce a tiny amount of missing values (1%) to demonstrate pipeline compliance
    mask_rider = np.random.rand(n_samples) < 0.01
    mask_weather = np.random.rand(n_samples) < 0.01
    df.loc[mask_rider, 'rider_experience_months'] = np.nan
    df.loc[mask_weather, 'weather_condition'] = np.nan

    df.to_csv('results/dataset.csv', index=False)
    return df


# ============================================================================
# METHODOLOGICAL PIPELINE STEPS
# ============================================================================

def impute_missing_values(df, training_medians=None, training_modes=None):
    """
    Imputes missing values using Median for numeric features and Mode for categorical features.
    """
    df_clean = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = ['delivery_distance_km', 'order_size_kg', 'rider_experience_months']
    categorical_cols = ['time_of_day', 'location_type', 'weather_condition', 'day_of_week', 'season']

    # Dicts to store fitted imputation parameters during training
    medians = {}
    modes = {}

    for col in numeric_cols:
        if col in df_clean.columns:
            fill_val = training_medians[col] if training_medians is not None else df_clean[col].median()
            medians[col] = fill_val
            df_clean[col] = df_clean[col].fillna(fill_val)

    for col in categorical_cols:
        if col in df_clean.columns:
            fill_val = training_modes[col] if training_modes is not None else df_clean[col].mode()[0]
            modes[col] = fill_val
            df_clean[col] = df_clean[col].fillna(fill_val)

    return df_clean, medians, modes


def treat_outliers(df, training_bounds=None):
    """
    Detects outliers using the Interquartile Range (IQR) method and caps them (Winsorization).
    """
    df_clean = df.copy()
    outlier_cols = ['delivery_distance_km', 'order_size_kg']
    bounds = {}

    for col in outlier_cols:
        if col in df_clean.columns:
            if training_bounds is not None:
                lower_bound, upper_bound = training_bounds[col]
            else:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
            
            bounds[col] = (lower_bound, upper_bound)
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

    return df_clean, bounds


def engineer_features(df):
    """
    Creates the four derived features specified in Chapter Three.
    """
    df_eng = df.copy()
    
    # Ensure columns exist and fill NaNs safely before feature engineering
    dist = pd.to_numeric(df_eng.get('delivery_distance_km'), errors='coerce').fillna(10.0)
    traf = pd.to_numeric(df_eng.get('traffic_level'), errors='coerce').fillna(3.0)
    tod = df_eng.get('time_of_day', 'afternoon').astype(str).str.lower().str.strip()
    loc = df_eng.get('location_type', 'mainland').astype(str).str.lower().str.strip()
    addr = pd.to_numeric(df_eng.get('address_quality_score'), errors='coerce').fillna(2.0)
    seas = df_eng.get('season', 'dry').astype(str).str.lower().str.strip()

    # 1. Distance-traffic interaction
    df_eng['distance_traffic_interaction'] = dist * traf

    # 2. Time risk category (evening and night periods)
    df_eng['time_risk_category'] = tod.isin(['evening', 'night']).astype(int)

    # 3. Zone risk score (location mapping + address quality score)
    loc_mapping = {'island': 1, 'mainland': 2, 'suburban': 3, 'rural': 4}
    mapped_loc = loc.map(loc_mapping).fillna(2)
    df_eng['zone_risk_score'] = mapped_loc + addr

    # 4. Seasonal disruption indicator (rainy season flag)
    df_eng['seasonal_disruption_indicator'] = (seas == 'rainy').astype(int)

    return df_eng



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

# ============================================================================
# API ENDPOINTS (ASYNCHRONOUS BACKGROUND TRAINING)
# ============================================================================

def update_state(status=None, progress=None, message=None, error_message=None, results=None):
    with training_lock:
        if status is not None:
            training_state['status'] = status
        if progress is not None:
            training_state['progress'] = progress
        if message is not None:
            training_state['message'] = message
        if error_message is not None:
            training_state['error_message'] = error_message
        if results is not None:
            training_state['results'] = results


def validate_csv(df):
    required_cols = [
        'delivery_distance_km', 'traffic_level', 'time_of_day', 'location_type', 
        'order_size_kg', 'weather_condition', 'rider_experience_months', 
        'address_quality_score', 'day_of_week', 'season', 'delivery_outcome'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Check if target variable has the expected classes
    unique_outcomes = df['delivery_outcome'].dropna().unique().tolist()
    valid_outcomes = ['success', 'delayed', 'failed']
    invalid_outcomes = [o for o in unique_outcomes if o not in valid_outcomes]
    if len(unique_outcomes) == 0:
        raise ValueError("The 'delivery_outcome' column cannot be empty.")
    if invalid_outcomes:
        raise ValueError(f"Invalid outcomes: {invalid_outcomes}. Must be: {valid_outcomes}")


def run_training_thread(filepath=None):
    try:
        update_state(status='training', progress=5, message='Loading and validating dataset...', error_message=None, results=None)
        
        if filepath:
            print(f"[THREAD] Loading custom CSV: {filepath}")
            df = pd.read_csv(filepath)
            validate_csv(df)
            update_state(progress=15, message=f'Successfully validated CSV with {len(df)} records. Running imputation...')
        else:
            print("[THREAD] Generating synthetic calibrated dataset...")
            df = generate_dataset(10000)
            update_state(progress=15, message='Default calibrated dataset generated. Running imputation...')

        # Step 2: Imputation & Outlier Treatment
        df_imputed, medians, modes = impute_missing_values(df)
        df_treated, bounds = treat_outliers(df_imputed)
        update_state(progress=30, message='Imputation and outlier capping complete. Engineering derived features...')

        # Step 3: Feature Engineering
        df_features = engineer_features(df_treated)
        update_state(progress=40, message='Derived features engineered. Encoding categorical variables & scaling...')

        # Step 4: Encoding & Scaling
        y = df_features['delivery_outcome'].copy()
        X_raw = df_features.drop(columns=['delivery_outcome'])

        nominal_cols = ['time_of_day', 'location_type', 'weather_condition', 'day_of_week', 'season']
        numeric_cols = [
            'delivery_distance_km', 'traffic_level', 'order_size_kg', 
            'rider_experience_months', 'address_quality_score',
            'distance_traffic_interaction', 'time_risk_category', 
            'zone_risk_score', 'seasonal_disruption_indicator'
        ]

        # One-Hot Encode nominal variables
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_nominal_encoded = encoder.fit_transform(X_raw[nominal_cols])
        nominal_feature_names = encoder.get_feature_names_out(nominal_cols).tolist()
        
        X_encoded_df = pd.DataFrame(X_nominal_encoded, columns=nominal_feature_names)
        X_numeric_df = X_raw[numeric_cols].reset_index(drop=True)
        
        X_final = pd.concat([X_numeric_df, X_encoded_df], axis=1)
        final_feature_names = X_final.columns.tolist()

        # Standardize all features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_final)
        X = pd.DataFrame(X_scaled, columns=final_feature_names)

        # Split 70/15/15 with stratification
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

        # Step 5: Train Models
        update_state(progress=50, message='Training Logistic Regression baseline with 5-fold CV...')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        models_dict = {}

        # 1. Logistic Regression
        grid_lr = GridSearchCV(
            LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000),
            {'C': [0.01, 0.1, 1.0, 10.0]},
            cv=cv, scoring='f1_macro', n_jobs=-1
        )
        grid_lr.fit(X_train, y_train)
        y_pred = grid_lr.best_estimator_.predict(X_test)
        results['logistic_regression'] = get_metrics(y_test, y_pred, grid_lr.best_params_)
        models_dict['logistic_regression'] = grid_lr.best_estimator_

        update_state(progress=70, message='Training Decision Tree classifier with GridSearchCV hyperparameter tuning...')

        # 2. Decision Tree
        grid_dt = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            {'max_depth': [5,10,15,20], 'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,5]},
            cv=cv, scoring='f1_macro', n_jobs=-1
        )
        grid_dt.fit(X_train, y_train)
        y_pred = grid_dt.best_estimator_.predict(X_test)
        results['decision_tree'] = get_metrics(y_test, y_pred, grid_dt.best_params_)
        models_dict['decision_tree'] = grid_dt.best_estimator_

        update_state(progress=85, message='Training Random Forest ensemble (combining 100-300 estimators)...')

        # 3. Random Forest
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
        fi = sorted(zip(final_feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
        results['random_forest']['feature_importance'] = [{'feature': f, 'importance': round(v,4)} for f,v in fi]

        update_state(progress=95, message='Saving model binary & JSON performance metrics...')

        # Save all pipeline components
        with open('models/trained_models.pkl', 'wb') as f:
            pickle.dump({
                'models': models_dict,
                'medians': medians,
                'modes': modes,
                'bounds': bounds,
                'encoder': encoder,
                'scaler': scaler,
                'nominal_cols': nominal_cols,
                'numeric_cols': numeric_cols,
                'feature_names': final_feature_names
            }, f)

        results_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(df),
                'n_features': len(final_feature_names),
                'class_distribution': df['delivery_outcome'].value_counts().to_dict()
            },
            'feature_names': final_feature_names,
            'models': results
        }
        with open('results/training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        update_state(status='success', progress=100, message='All models trained successfully!', results=results)

    except Exception as e:
        print(f"\n[THREAD ERROR]: {str(e)}")
        import traceback; traceback.print_exc()
        update_state(status='error', progress=100, message=f'Error occurred: {str(e)}', error_message=str(e))


@app.route('/api/train', methods=['POST'])
def train():
    # Check if already training
    with training_lock:
        if training_state['status'] == 'training':
            return jsonify({
                'status': 'error',
                'message': 'Training is already in progress. Please wait for it to complete.'
            }), 400

    # Handle optional CSV file upload
    filepath = None
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '':
            if not file.filename.endswith('.csv'):
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid file format. Only CSV files are accepted.'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', f"uploaded_{int(datetime.now().timestamp())}_{filename}")
            file.save(filepath)

    # Launch background thread
    t = threading.Thread(target=run_training_thread, args=(filepath,))
    t.daemon = True
    t.start()

    return jsonify({
        'status': 'training',
        'message': 'Model training has started in the background.'
    }), 202


@app.route('/api/train/status', methods=['GET'])
def train_status():
    with training_lock:
        return jsonify(training_state.copy()), 200


@app.route('/api/train/reset', methods=['POST'])
def train_reset():
    with training_lock:
        if training_state['status'] == 'training':
            return jsonify({'status': 'error', 'message': 'Cannot reset while training is in progress.'}), 400
        training_state['status'] = 'idle'
        training_state['progress'] = 0
        training_state['message'] = 'System is ready for training.'
        training_state['error_message'] = None
        training_state['results'] = None
    return jsonify({'status': 'success', 'message': 'Training state reset successfully.'}), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    """
    try:
        data = request.json

        with open('models/trained_models.pkl', 'rb') as f:
            pipeline = pipeline = pickle.load(f)

        models_dict  = pipeline['models']
        medians      = pipeline['medians']
        modes        = pipeline['modes']
        bounds       = pipeline['bounds']
        encoder      = pipeline['encoder']
        scaler       = pipeline['scaler']
        nominal_cols = pipeline['nominal_cols']
        numeric_cols = pipeline['numeric_cols']
        feature_names = pipeline['feature_names']

        # Convert input dictionary to DataFrame
        X_input = pd.DataFrame([data])

        # 1. Imputation
        X_imputed, _, _ = impute_missing_values(X_input, training_medians=medians, training_modes=modes)

        # 2. Outlier Treatment
        X_treated, _ = treat_outliers(X_imputed, training_bounds=bounds)

        # 3. Feature Engineering
        X_features = engineer_features(X_treated)

        # 4. Encoding Nominal Columns
        X_nominal_encoded = encoder.transform(X_features[nominal_cols])
        nominal_feature_names = encoder.get_feature_names_out(nominal_cols).tolist()
        X_encoded_df = pd.DataFrame(X_nominal_encoded, columns=nominal_feature_names)
        
        # Combine numeric and encoded columns
        X_numeric_df = X_features[numeric_cols].reset_index(drop=True)
        X_final = pd.concat([X_numeric_df, X_encoded_df], axis=1)

        # 5. Scaling
        X_scaled = scaler.transform(X_final)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        # Generate predictions across models
        predictions = {}
        for name, model in models_dict.items():
            pred  = model.predict(X_scaled_df)[0]
            proba = model.predict_proba(X_scaled_df)[0]
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

    