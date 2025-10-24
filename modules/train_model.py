from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

def train_model(df):
    
    # okay, standardise data first, then fit
    scaler = StandardScaler()
    # so X should not include IDs
    X = df[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    #X_scaled = scaler.fit_transform(X)
    y = df['Outcome']

    numeric_features = ["Tobacco", "Alcohol", "Performance status", "Surgery", "Chemotherapy", "Age", "Weight"]
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer(
    transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = MLPClassifier()

    clf = Pipeline(steps=[("preprocess", preprocessor),
                        ("mlp", model)])

    # Fit
    clf.fit(X, y)

    return clf