from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Assuming you have a test dataframe called df_test
def test_model(df_test, clf):
    # Extract features (same columns as training, excluding ID and outcome)
    X_test = df_test[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 
                    'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    y_test = df_test['Outcome']
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred

# Usage:
# clf = train_model(df_train)
# predictions = test_model(clf, df_test)



def test_model_ct_pt(df_test, clf):
    """
    Test a radiomics model (CT or PET).
    Uses the SAME feature columns the model was trained on.
    """
    # pull the exact feature order used during training
    if not hasattr(clf, "feature_cols_"):
        raise ValueError("Model missing feature_cols_. Make sure you trained with train_model_ct().")

    feature_cols = clf.feature_cols_

    # Build X_test in the same column order
    X_test = df_test[feature_cols]
    y_test = df_test["Outcome"].astype(int)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # AUC (MLP supports predict_proba)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred