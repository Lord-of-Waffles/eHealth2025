from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


def train_model_knn(df):
    
    X = df[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    y = df['Outcome']

    numeric_features = ["Tobacco", "Alcohol", "Performance status", "Surgery", "Chemotherapy", "Age", "Weight"]
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # replacede mlp with knn - Ben
    knn = KNeighborsClassifier(
        n_neighbors=7,         # changing the value here can change results
        weights='distance',    
        n_jobs=-1              
    )

    clf = Pipeline(steps=[("preprocess", preprocessor),
                          ("knn", knn)])

    clf.fit(X, y)
    return clf


def train_model_ct_knn(df_ct, include_center_id: bool = False):

    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_ct.columns:
        drop_cols.append("CenterID")

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

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        n_jobs=-1
    )

    clf = Pipeline(steps=[("preprocess", preprocessor),
                          ("knn", knn)])
    clf.fit(X, y)

    clf.feature_cols_ = feature_cols
    clf.include_center_id_ = include_center_id
    return clf


def train_model_pt_knn(df_pt, include_center_id: bool = False):

    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_pt.columns:
        drop_cols.append("CenterID")

    feature_cols = [c for c in df_pt.columns if c not in drop_cols]
    X = df_pt[feature_cols]
    y = df_pt["Outcome"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), feature_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        n_jobs=-1
    )

    clf = Pipeline(steps=[("preprocess", preprocessor),
                          ("knn", knn)])
    clf.fit(X, y)

    clf.feature_cols_ = feature_cols
    clf.include_center_id_ = include_center_id
    return clf
