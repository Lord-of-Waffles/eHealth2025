from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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