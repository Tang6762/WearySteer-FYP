import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_drowsiness_model(csv_path="wearysteer_features.csv", model_out="wearysteer_rf.pkl", scaler_out="wearysteer_scaler.pkl"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file '{csv_path}' not found at {csv_path}.")

    print(f"[*] Loading feature dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Clean missing/null rows if any existed during frame processing
    df = df.dropna()

    # Define features and label targets matching feature_extractor.py
    feature_cols = ['ear', 'mar', 'shoulder_slump']
    
    X = df[feature_cols]
    y = df['label'].astype(int)

    print(f"[*] Feature columns selected: {list(X.columns)}")
    print(f"[*] Class label counts:\n{y.value_counts().rename({0: 'Alert (0)', 1: 'Low Vigilance (1)', 2: 'Drowsy (2)'})}")

    # Stratified Train/Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    print("[*] Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[+] Model Evaluation Complete. Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Alert', 'Low Vigilance', 'Drowsy']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Export Model & Scaler artifact files
    joblib.dump(clf, model_out)
    joblib.dump(scaler, scaler_out)
    print(f"[+] Saved model to '{model_out}' and scaler to '{scaler_out}'.")

if __name__ == "__main__":
    train_drowsiness_model()