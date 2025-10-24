from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
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




def train_model_ct(df_ct, include_center_id: bool = False, random_state: int = 42):
    """
    Train an MLP on CT radiomics.
    - Drops Outcome & PatientID (and CenterID by default to avoid site bias).
    - Scales ALL numeric radiomics features with RobustScaler (handles outliers better).
    - Returns a pipeline and remembers the feature columns used.
    """
    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_ct.columns:
        drop_cols.append("CenterID")

    # Use all remaining columns as features (radiomics are numeric)
    feature_cols = [c for c in df_ct.columns if c not in drop_cols]

    X = df_ct[feature_cols]
    y = df_ct["Outcome"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), feature_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=random_state
    )

    clf = Pipeline(steps=[("preprocess", preprocessor),
                         ("mlp", mlp)])
    clf.fit(X, y)

    # remember the raw feature columns so test uses the same order
    clf.feature_cols_ = feature_cols
    clf.include_center_id_ = include_center_id
    return clf