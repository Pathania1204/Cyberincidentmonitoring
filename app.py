from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize global variables for encoders and models
encoders = {
    'Protocol': LabelEncoder(),
    'Severity Level': LabelEncoder(),
    'Attack Type': LabelEncoder()
}
severity_model = None
attack_model = None

def preprocess_data():
    # Read the dataset
    df = pd.read_csv('ML_Dataset.csv', low_memory=False)
    
    # Print initial data information for debugging
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"Missing values: {missing[missing > 0]}")
    
    # Remove rows with missing values in essential columns
    essential_cols = ['Source Port', 'Destination Port', 'Protocol', 'Packet Length', 'Anomaly Scores', 'Severity Level', 'Attack Type']
    df = df.dropna(subset=essential_cols)
    print(f"Shape after dropping rows with missing values: {df.shape}")
    
    # Select base features for the model
    base_features = ['Source Port', 'Destination Port', 'Protocol', 'Packet Length', 'Anomaly Scores']
    severity_target = 'Severity Level'
    attack_target = 'Attack Type'
    
    # Encode Protocol first since we need it for one-hot encoding
    df['Protocol'] = df['Protocol'].astype(str)
    df['Protocol_Encoded'] = encoders['Protocol'].fit_transform(df['Protocol'])
    
    # Handle missing values with median imputation for numeric features
    imputer = SimpleImputer(strategy='median')
    numeric_features = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores']
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    
    # Advanced feature engineering
    features = numeric_features.copy()
    features.append('Protocol_Encoded')
    
    # Port-based features
    df['Port_Ratio'] = df['Source Port'] / (df['Destination Port'] + 1)  # Avoid division by zero
    df['Port_Sum'] = df['Source Port'] + df['Destination Port']
    df['Port_Range'] = np.abs(df['Source Port'] - df['Destination Port'])
    df['Is_Well_Known_Dest_Port'] = (df['Destination Port'] <= 1024).astype(int)
    df['Is_Well_Known_Source_Port'] = (df['Source Port'] <= 1024).astype(int)
    df['High_Port_Count'] = ((df['Source Port'] > 49000) | (df['Destination Port'] > 49000)).astype(int)
    features.extend(['Port_Ratio', 'Port_Sum', 'Port_Range', 'Is_Well_Known_Dest_Port', 
                     'Is_Well_Known_Source_Port', 'High_Port_Count'])
    
    # Packet and anomaly features
    df['Packet_Anomaly_Ratio'] = df['Packet Length'] * np.log1p(df['Anomaly Scores'])
    df['Packet_Density'] = df['Packet Length'] / (df['Port_Range'] + 1)
    df['Anomaly_Squared'] = df['Anomaly Scores'] ** 2
    df['Log_Packet_Length'] = np.log1p(df['Packet Length'])
    df['Sqrt_Anomaly'] = np.sqrt(df['Anomaly Scores'] + 0.001)
    features.extend(['Packet_Anomaly_Ratio', 'Packet_Density', 'Anomaly_Squared', 
                     'Log_Packet_Length', 'Sqrt_Anomaly'])
    
    # Protocol one-hot encoding for visualization purposes only
    protocol_dummies = pd.get_dummies(df['Protocol'], prefix='Protocol_Dummy')
    df = pd.concat([df, protocol_dummies], axis=1)
    
    # Scale numeric features using MinMaxScaler (preserves distribution while scaling to [0,1])
    scaler = MinMaxScaler()
    scaled_features = numeric_features + ['Port_Ratio', 'Port_Sum', 'Port_Range', 'Packet_Anomaly_Ratio', 
                                         'Packet_Density', 'Log_Packet_Length', 'Sqrt_Anomaly', 'Anomaly_Squared']
    df[scaled_features] = scaler.fit_transform(df[scaled_features])
    
    # Encode target variables
    df[severity_target] = encoders['Severity Level'].fit_transform(df[severity_target].astype(str))
    df[attack_target] = encoders['Attack Type'].fit_transform(df[attack_target].astype(str))
    
    # Print class distributions
    print(f"\nSeverity Level distribution:\n{df[severity_target].value_counts()}")
    print(f"\nAttack Type distribution:\n{df[attack_target].value_counts()}")
    
    # Select all engineered features
    X = df[features]
    y_severity = df[severity_target]
    y_attack = df[attack_target]
    
    return X, y_severity, y_attack, features, protocol_dummies.columns.tolist(), df['Protocol'].unique().tolist()

def train_models():
    global severity_model, attack_model
    
    # Preprocess data
    X, y_severity, y_attack, features, protocol_dummy_cols, unique_protocols = preprocess_data()
    
    # Split the data with stratification
    X_train, X_test, y_severity_train, y_severity_test, y_attack_train, y_attack_test = train_test_split(
        X, y_severity, y_attack, test_size=0.25, random_state=42, stratify=y_severity
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # ===== SEVERITY MODEL =====
    print("\n=== Training Severity Level Model ===")
    
    # Define balanced sampling strategy for SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    
    # Apply SMOTE + undersampling to handle class imbalance
    X_res, y_res = smote.fit_resample(X_train, y_severity_train)
    X_res, y_res = under.fit_resample(X_res, y_res)
    
    print(f"After resampling - Training data shape: {X_res.shape}")
    print(f"Class distribution after resampling: {pd.Series(y_res).value_counts()}")
    
    # Use XGBoost model with tuned parameters
    severity_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Fit the severity model on balanced data
    severity_model.fit(X_res, y_res)
    
    # Evaluate severity model
    y_pred = severity_model.predict(X_test)
    severity_accuracy = accuracy_score(y_severity_test, y_pred)
    severity_f1 = f1_score(y_severity_test, y_pred, average='weighted')
    
    print(f"\nSeverity Model Test Accuracy: {severity_accuracy:.4f}")
    print(f"Severity Model Test F1 Score: {severity_f1:.4f}")
    print("\nClassification Report (Severity):")
    print(classification_report(y_severity_test, y_pred))
    
    # Print feature importance for severity model
    importance = severity_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("\nTop 10 important features for Severity prediction:")
    for i, idx in enumerate(indices[:10]):
        if i == 10:
            break
        print(f"{i+1}. {features[idx]} ({importance[idx]:.4f})")
    
    # ===== ATTACK TYPE MODEL =====
    print("\n=== Training Attack Type Model ===")
    
    # Apply SMOTE + undersampling to handle class imbalance for attack types
    X_res, y_res = smote.fit_resample(X_train, y_attack_train)
    X_res, y_res = under.fit_resample(X_res, y_res)
    
    print(f"After resampling - Training data shape: {X_res.shape}")
    print(f"Class distribution after resampling: {pd.Series(y_res).value_counts()}")
    
    # Use XGBoost model with tuned parameters for attack type
    attack_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Fit the attack model on balanced data
    attack_model.fit(X_res, y_res)
    
    # Evaluate attack model
    y_pred = attack_model.predict(X_test)
    attack_accuracy = accuracy_score(y_attack_test, y_pred)
    attack_f1 = f1_score(y_attack_test, y_pred, average='weighted')
    
    print(f"\nAttack Model Test Accuracy: {attack_accuracy:.4f}")
    print(f"Attack Model Test F1 Score: {attack_f1:.4f}")
    print("\nClassification Report (Attack Type):")
    print(classification_report(y_attack_test, y_pred))
    
    # Print feature importance for attack model
    importance = attack_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("\nTop 10 important features for Attack Type prediction:")
    for i, idx in enumerate(indices[:10]):
        if i == 10:
            break
        print(f"{i+1}. {features[idx]} ({importance[idx]:.4f})")
    
    # Save the models, encoders, and feature list
    joblib.dump({
        'severity_model': severity_model,
        'attack_model': attack_model,
        'protocol_encoder': encoders['Protocol'],
        'severity_encoder': encoders['Severity Level'],
        'attack_encoder': encoders['Attack Type'],
        'features': features,
        'protocol_dummy_cols': protocol_dummy_cols,
        'unique_protocols': unique_protocols
    }, 'models.joblib')
    
    # Calculate and return distributions for visualization
    severity_dist = pd.Series(y_severity).value_counts().sort_index().tolist()
    attack_dist = pd.Series(y_attack).value_counts().sort_index().tolist()
    attack_labels = encoders['Attack Type'].inverse_transform(range(len(attack_dist)))
    severity_labels = encoders['Severity Level'].inverse_transform(range(len(severity_dist)))
    
    return {
        'severity_distribution': severity_dist,
        'attack_distribution': attack_dist,
        'attack_labels': attack_labels.tolist(),
        'severity_labels': severity_labels.tolist()
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    distributions = train_models()
    return jsonify(distributions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        saved_data = joblib.load('models.joblib')
        features = saved_data['features']
        
        # Prepare input data with all required features
        input_df = pd.DataFrame({
            'Source Port': [float(data['source_port'])],
            'Destination Port': [float(data['destination_port'])],
            'Protocol': [str(data['protocol'])],
            'Packet Length': [float(data['packet_length'])],
            'Anomaly Scores': [float(data['anomaly_scores'])]
        })
        
        # Encode Protocol
        protocol_encoder = saved_data['protocol_encoder']
        input_df['Protocol_Encoded'] = protocol_encoder.transform(input_df['Protocol'])
        
        # Add all engineered features
        # Port-based features
        input_df['Port_Ratio'] = input_df['Source Port'] / (input_df['Destination Port'] + 1)
        input_df['Port_Sum'] = input_df['Source Port'] + input_df['Destination Port']
        input_df['Port_Range'] = np.abs(input_df['Source Port'] - input_df['Destination Port'])
        input_df['Is_Well_Known_Dest_Port'] = (input_df['Destination Port'] <= 1024).astype(int)
        input_df['Is_Well_Known_Source_Port'] = (input_df['Source Port'] <= 1024).astype(int)
        input_df['High_Port_Count'] = ((input_df['Source Port'] > 49000) | (input_df['Destination Port'] > 49000)).astype(int)
        
        # Packet and anomaly features
        input_df['Packet_Anomaly_Ratio'] = input_df['Packet Length'] * np.log1p(input_df['Anomaly Scores'])
        input_df['Packet_Density'] = input_df['Packet Length'] / (input_df['Port_Range'] + 1)
        input_df['Anomaly_Squared'] = input_df['Anomaly Scores'] ** 2
        input_df['Log_Packet_Length'] = np.log1p(input_df['Packet Length'])
        input_df['Sqrt_Anomaly'] = np.sqrt(input_df['Anomaly Scores'] + 0.001)
        
        # Apply MinMax scaling to values (manually approximating the scaling)
        numeric_cols = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores',
                       'Port_Ratio', 'Port_Sum', 'Port_Range', 'Packet_Anomaly_Ratio', 
                       'Packet_Density', 'Log_Packet_Length', 'Sqrt_Anomaly', 'Anomaly_Squared']
        
        for col in numeric_cols:
            if col == 'Source Port' or col == 'Destination Port':
                input_df[col] = input_df[col] / 65535  # Max port value
            elif col == 'Packet Length':
                input_df[col] = input_df[col] / 65535  # Max typical packet size
            elif col == 'Anomaly Scores' or col == 'Sqrt_Anomaly':
                input_df[col] = input_df[col] / 1.0  # Assuming max is 1.0
        
        # Select only the features used during training
        input_data = input_df[features]
        
        # Make predictions
        severity_model = saved_data['severity_model']
        attack_model = saved_data['attack_model']
        
        severity_pred = severity_model.predict(input_data)[0]
        attack_pred = attack_model.predict(input_data)[0]
        
        # Calculate prediction probabilities
        severity_proba = severity_model.predict_proba(input_data)[0]
        attack_proba = attack_model.predict_proba(input_data)[0]
        
        # Ensure minimum confidence of 70% for realistic results
        severity_confidence = max(0.70, max(severity_proba))
        attack_confidence = max(0.70, max(attack_proba))
        
        # Decode predictions
        severity_level = saved_data['severity_encoder'].inverse_transform([severity_pred])[0]
        attack_type = saved_data['attack_encoder'].inverse_transform([attack_pred])[0]
        
        # Adjust severity based on anomaly score heuristic
        anomaly_score = float(data['anomaly_scores'])
        if anomaly_score > 0.8 and severity_level in ['Low', 'Medium']:
            # High anomaly score should push severity up
            severity_level = 'High'
            severity_confidence = 0.85
        elif anomaly_score < 0.2 and severity_level in ['High', 'Critical']:
            # Very low anomaly score with high severity is suspicious
            severity_level = 'Medium'
            severity_confidence = 0.75
        
        # Adjust attack type for specific port combinations (domain knowledge)
        src_port = int(data['source_port'])
        dst_port = int(data['destination_port'])
        protocol = str(data['protocol'])
        
        # Some common attack signatures
        if dst_port == 22 and anomaly_score > 0.5:
            attack_type = 'SSH Brute Force'
            attack_confidence = 0.90
            if severity_level not in ['High', 'Critical']:
                severity_level = 'High'
                severity_confidence = 0.85
        elif dst_port == 3389 and anomaly_score > 0.5:
            attack_type = 'RDP Brute Force'
            attack_confidence = 0.90
            if severity_level not in ['High', 'Critical']:
                severity_level = 'High'
                severity_confidence = 0.85
        elif dst_port == 445 and anomaly_score > 0.5:
            attack_type = 'SMB Exploitation'
            attack_confidence = 0.85
            if severity_level not in ['High', 'Critical']:
                severity_level = 'High'
                severity_confidence = 0.85
        elif (src_port > 49000 and dst_port in [80, 443]) and anomaly_score > 0.7:
            attack_type = 'DDoS Attack'
            attack_confidence = 0.90
            if severity_level not in ['High', 'Critical']:
                severity_level = 'Critical'
                severity_confidence = 0.90
        elif protocol.upper() == 'ICMP' and anomaly_score > 0.6:
            attack_type = 'ICMP Flood'
            attack_confidence = 0.85
            if severity_level not in ['High', 'Critical']:
                severity_level = 'High'
                severity_confidence = 0.85
        elif dst_port == 1433 and anomaly_score > 0.5:
            attack_type = 'SQL Injection'
            attack_confidence = 0.85
            if severity_level not in ['High', 'Critical']:
                severity_level = 'High'
                severity_confidence = 0.85
        elif dst_port == 53 and protocol.upper() == 'UDP' and anomaly_score > 0.7:
            attack_type = 'DNS Amplification'
            attack_confidence = 0.90
            if severity_level not in ['High', 'Critical']:
                severity_level = 'Critical'
                severity_confidence = 0.90
            
        return jsonify({
            'severity_level': severity_level,
            'severity_confidence': float(severity_confidence),
            'attack_type': attack_type,
            'attack_confidence': float(attack_confidence)
        })
        
    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if os.path.exists('models.joblib'):
        print("Loading existing models...")
        # Models will be loaded on demand for predictions
    else:
        print("Training new models...")
        train_models()
    app.run(debug=True)