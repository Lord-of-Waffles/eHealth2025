import pandas as pd
import numpy as np
from modules.fix_missing_values_for_clinical import fix_missing_values
from modules.train_model import train_model, train_model_ct, train_model_pt
from modules.test_model import test_model, test_model_ct_pt
from modules.train_model_knn import train_model_knn, train_model_ct_knn, train_model_pt_knn
from modules.train_ensemble_model import train_ensemble_model_clinical, train_ensemble_model_ct, train_ensemble_model_pt


def run_clinical(df_train, df_test):
    print("\n=== Clinical: Training ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_model(df_train)
    print("Clinical training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    test_model(df_test, clf)   # prints Accuracy, Report, Confusion Matrix


def run_ct(df_train, df_test, include_center_id=False):
    print("\n=== CT: Training ===")
    clf_ct = train_model_ct(df_train, include_center_id=include_center_id)
    print("CT training complete.")

    print("\n=== CT: Testing ===")
    test_model_ct_pt(df_test, clf_ct)  # prints Accuracy, ROC-AUC, Report, Confusion Matrix


def run_pt(df_train, df_test, include_center_id=False):
    print("\n=== PET: Training ===")
    clf_pt = train_model_pt(df_train, include_center_id=include_center_id)
    print("PET training complete.")

    print("\n=== PET: Testing ===")
    test_model_ct_pt(df_test, clf_pt)


def run_clinical_knn(df_train, df_test):
    print("\n=== Clinical: Training (kNN) ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_model_knn(df_train)
    print("Clinical training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    test_model(df_test, clf)   # prints Accuracy, Report, Confusion Matrix


def run_ct_knn(df_train, df_test, include_center_id=False):
    print("\n=== CT: Training (kNN) ===")
    clf_ct = train_model_ct_knn(df_train, include_center_id=include_center_id)
    print("CT training complete.")

    print("\n=== CT: Testing ===")
    test_model_ct_pt(df_test, clf_ct)  # prints Accuracy, ROC-AUC, Report, Confusion Matrix


def run_pt_knn(df_train, df_test, include_center_id=False):
    print("\n=== PET: Training (kNN) ===")
    clf_pt = train_model_pt_knn(df_train, include_center_id=include_center_id)
    print("PET training complete.")

    print("\n=== PET: Testing ===")
    test_model_ct_pt(df_test, clf_pt)

def run_clinical_ensemble(df_train, df_test):
    print("\n=== Clinical: Training ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_ensemble_model_clinical(df_train)
    print("Clinical training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    test_model(df_test, clf)   # prints Accuracy, Report, Confusion Matrix


def run_ct_ensemble(df_train, df_test, include_center_id=False):
    print("\n=== CT: Training ===")
    clf_ct = train_ensemble_model_ct(df_train, include_center_id=include_center_id)
    print("CT training complete.")

    print("\n=== CT: Testing ===")
    test_model_ct_pt(df_test, clf_ct)  # prints Accuracy, ROC-AUC, Report, Confusion Matrix


def run_pt_ensemble(df_train, df_test, include_center_id=False):
    print("\n=== PET: Training ===")
    clf_pt = train_ensemble_model_pt(df_train, include_center_id=include_center_id)
    print("PET training complete.")

    print("\n=== PET: Testing ===")
    test_model_ct_pt(df_test, clf_pt)
