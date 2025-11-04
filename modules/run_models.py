import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from modules.fix_missing_values_for_clinical import fix_missing_values
from modules.train_model import train_model, train_model_ct, train_model_pt
from modules.test_model import test_model, test_model_ct_pt
from modules.train_model_knn import train_model_knn, train_model_ct_knn, train_model_pt_knn
from modules.train_ensemble_model import train_ensemble_model_clinical, train_ensemble_model_ct, train_ensemble_model_pt


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        #y_true: true labels
        #y_pred: predicted labels
        #y_pred_proba: predicted probabilities (optional, for ROC-AUC)
        #returns: dict with accuracy, precision, recall, f1_score, and optionally roc_auc
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            metrics['roc_auc'] = None
    
    return metrics


def run_clinical(df_train, df_test):
    """Train and test clinical MLP model, return metrics."""
    print("\n=== Clinical: Training (MLP) ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_model(df_train)
    print("Clinical training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    
    # Get predictions
    X_test = df_test[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 
                      'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    y_true = df_test['Outcome']
    y_pred = clf.predict(X_test)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Still call test_model for detailed output (confusion matrix, etc.)
    test_model(df_test, clf)
    
    return metrics


def run_ct(df_train, df_test, include_center_id=False):
    """Train and test CT MLP model, return metrics."""
    print("\n=== CT: Training (MLP) ===")
    clf_ct = train_model_ct(df_train, include_center_id=include_center_id)
    print("CT training complete.")

    print("\n=== CT: Testing ===")
    
    # Get predictions
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_test.columns:
        drop_cols.append("CenterID")
    
    feature_cols = clf_ct.feature_cols_
    X_test = df_test[feature_cols]
    y_true = df_test["Outcome"].astype(int)
    
    y_pred = clf_ct.predict(X_test)
    y_pred_proba = clf_ct.predict_proba(X_test)[:, 1] if hasattr(clf_ct, 'predict_proba') else None
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Still call test_model_ct_pt for detailed output
    test_model_ct_pt(df_test, clf_ct)
    
    return metrics


def run_pt(df_train, df_test, include_center_id=False):
    """Train and test PET MLP model, return metrics."""
    print("\n=== PET: Training (MLP) ===")
    clf_pt = train_model_pt(df_train, include_center_id=include_center_id)
    print("PET training complete.")

    print("\n=== PET: Testing ===")
    
    # Get predictions
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_test.columns:
        drop_cols.append("CenterID")
    
    feature_cols = clf_pt.feature_cols_
    X_test = df_test[feature_cols]
    y_true = df_test["Outcome"].astype(int)
    
    y_pred = clf_pt.predict(X_test)
    y_pred_proba = clf_pt.predict_proba(X_test)[:, 1] if hasattr(clf_pt, 'predict_proba') else None
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Still call test_model_ct_pt for detailed output
    test_model_ct_pt(df_test, clf_pt)
    
    return metrics


def run_clinical_knn(df_train, df_test):
    """Train and test clinical kNN model, return metrics."""
    print("\n=== Clinical: Training (kNN) ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_model_knn(df_train)
    print("Clinical kNN training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    
    # Get predictions
    X_test = df_test[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 
                      'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    y_true = df_test['Outcome']
    y_pred = clf.predict(X_test)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Still call test_model for detailed output
    test_model(df_test, clf)
    
    return metrics


def run_ct_knn(df_train, df_test, include_center_id=False):
    """Train and test CT kNN model, return metrics."""
    print("\n=== CT: Training (kNN) ===")
    clf_ct = train_model_ct_knn(df_train, include_center_id=include_center_id)
    print("CT kNN training complete.")

    print("\n=== CT: Testing ===")
    
    # Get predictions
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_test.columns:
        drop_cols.append("CenterID")
    
    feature_cols = clf_ct.feature_cols_
    X_test = df_test[feature_cols]
    y_true = df_test["Outcome"].astype(int)
    
    y_pred = clf_ct.predict(X_test)
    y_pred_proba = clf_ct.predict_proba(X_test)[:, 1] if hasattr(clf_ct, 'predict_proba') else None
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Still call test_model_ct_pt for detailed output
    test_model_ct_pt(df_test, clf_ct)
    
    return metrics


def run_pt_knn(df_train, df_test, include_center_id=False):
    """Train and test PET kNN model, return metrics."""
    print("\n=== PET: Training (kNN) ===")
    clf_pt = train_model_pt_knn(df_train, include_center_id=include_center_id)
    print("PET kNN training complete.")

    print("\n=== PET: Testing ===")
    
    # Get predictions
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_test.columns:
        drop_cols.append("CenterID")
    
    feature_cols = clf_pt.feature_cols_
    X_test = df_test[feature_cols]
    y_true = df_test["Outcome"].astype(int)
    
    y_pred = clf_pt.predict(X_test)
    y_pred_proba = clf_pt.predict_proba(X_test)[:, 1] if hasattr(clf_pt, 'predict_proba') else None
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Still call test_model_ct_pt for detailed output
    test_model_ct_pt(df_test, clf_pt)
    
    return metrics


def run_clinical_ensemble(df_train, df_test):
    """Train and test clinical ensemble model, return metrics."""
    print("\n=== Clinical: Training (Ensemble) ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_ensemble_model_clinical(df_train)
    print("Clinical ensemble training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    
    # Get predictions
    X_test = df_test[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 
                      'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    y_true = df_test['Outcome']
    y_pred = clf.predict(X_test)
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Still call test_model for detailed output
    test_model(df_test, clf)
    
    return metrics


def run_ct_ensemble(df_train, df_test, include_center_id=False):
    """Train and test CT ensemble model, return metrics."""
    print("\n=== CT: Training (Ensemble) ===")
    clf_ct = train_ensemble_model_ct(df_train, include_center_id=include_center_id)
    print("CT ensemble training complete.")

    print("\n=== CT: Testing ===")
    
    # Get predictions
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_test.columns:
        drop_cols.append("CenterID")
    
    feature_cols = clf_ct.feature_cols_
    X_test = df_test[feature_cols]
    y_true = df_test["Outcome"].astype(int)
    
    y_pred = clf_ct.predict(X_test)
    y_pred_proba = clf_ct.predict_proba(X_test)[:, 1] if hasattr(clf_ct, 'predict_proba') else None
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Still call test_model_ct_pt for detailed output
    test_model_ct_pt(df_test, clf_ct)
    
    return metrics


def run_pt_ensemble(df_train, df_test, include_center_id=False):
    """Train and test PET ensemble model, return metrics."""
    print("\n=== PET: Training (Ensemble) ===")
    clf_pt = train_ensemble_model_pt(df_train, include_center_id=include_center_id)
    print("PET ensemble training complete.")

    print("\n=== PET: Testing ===")
    
    # Get predictions
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_test.columns:
        drop_cols.append("CenterID")
    
    feature_cols = clf_pt.feature_cols_
    X_test = df_test[feature_cols]
    y_true = df_test["Outcome"].astype(int)
    
    y_pred = clf_pt.predict(X_test)
    y_pred_proba = clf_pt.predict_proba(X_test)[:, 1] if hasattr(clf_pt, 'predict_proba') else None
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Still call test_model_ct_pt for detailed output
    test_model_ct_pt(df_test, clf_pt)
    
    return metrics