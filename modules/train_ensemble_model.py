from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import numpy as np

def train_ensemble_model_clinical(df, random_state: int = 42):

    X = df[['Gender', 'Tobacco', 'Alcohol', 'Performance status', 
            'Surgery', 'Chemotherapy', 'Age', 'Weight']]
    y = df['Outcome']

    numeric_features = ["Tobacco", "Alcohol", "Performance status", 
                       "Surgery", "Chemotherapy", "Age", "Weight"]
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Smaller networks for clinical data (fewer features)
    mlp1 = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        alpha=0.0001,
        random_state=random_state
    )
    
    mlp2 = MLPClassifier(
        hidden_layer_sizes=(20, 10),
        activation="tanh",
        solver="adam",
        max_iter=500,
        alpha=0.001,
        random_state=random_state + 1
    )
    
    mlp3 = MLPClassifier(
        hidden_layer_sizes=(40, 20, 10),
        activation="relu",
        solver="adam",
        max_iter=500,
        alpha=0.00001,
        random_state=random_state + 2
    )

    ensemble = VotingClassifier(
        estimators=[
            ('mlp1', mlp1),
            ('mlp2', mlp2),
            ('mlp3', mlp3)
        ],
        voting='soft',
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("ensemble", ensemble)
    ])
    
    clf.fit(X, y)
    return clf

def train_ensemble_model_ct(df_ct, include_center_id: bool = False, random_state: int = 42):

    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_ct.columns:
        drop_cols.append("CenterID")

    feature_cols = [c for c in df_ct.columns if c not in drop_cols]
    X = df_ct[feature_cols]
    y = df_ct["Outcome"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[("num", RobustScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # mpls with different features
    mlp1 = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        alpha=0.0001,  
        random_state=random_state
    )
    
    mlp2 = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="tanh",
        solver="adam",
        max_iter=500,
        alpha=0.001,  
        random_state=random_state + 1
    )
    
    mlp3 = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=500,
        alpha=0.00001, 
        learning_rate_init=0.001,
        random_state=random_state + 2
    )
    
    mlp4 = MLPClassifier(
        hidden_layer_sizes=(80, 40, 20),
        activation="logistic",  
        solver="adam",
        max_iter=500,
        alpha=0.0001,
        random_state=random_state + 3
    )

    # Voting ensemble - 'soft' uses predicted probabilities
    ensemble = VotingClassifier(
        estimators=[
            ('mlp_deep', mlp1),
            ('mlp_narrow', mlp2),
            ('mlp_wide', mlp3),
            ('mlp_sigmoid', mlp4)
        ],
        voting='soft',  
        n_jobs=-1  
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("ensemble", ensemble)
    ])
    
    clf.fit(X, y)

    clf.feature_cols_ = feature_cols
    clf.include_center_id_ = include_center_id
    return clf


def train_ensemble_model_pt(df_pt, include_center_id: bool = False, random_state: int = 42):

    drop_cols = ["Outcome", "PatientID"]
    if not include_center_id and "CenterID" in df_pt.columns:
        drop_cols.append("CenterID")

    feature_cols = [c for c in df_pt.columns if c not in drop_cols]
    X = df_pt[feature_cols]
    y = df_pt["Outcome"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[("num", RobustScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Same ensemble setup as CT
    mlp1 = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        alpha=0.0001,
        random_state=random_state
    )
    
    mlp2 = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="tanh",
        solver="adam",
        max_iter=500,
        alpha=0.001,
        random_state=random_state + 1
    )
    
    mlp3 = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=500,
        alpha=0.00001,
        learning_rate_init=0.001,
        random_state=random_state + 2
    )
    
    mlp4 = MLPClassifier(
        hidden_layer_sizes=(80, 40, 20),
        activation="logistic",
        solver="adam",
        max_iter=500,
        alpha=0.0001,
        random_state=random_state + 3
    )

    ensemble = VotingClassifier(
        estimators=[
            ('mlp_deep', mlp1),
            ('mlp_narrow', mlp2),
            ('mlp_wide', mlp3),
            ('mlp_sigmoid', mlp4)
        ],
        voting='soft',
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("ensemble", ensemble)
    ])
    
    clf.fit(X, y)

    clf.feature_cols_ = feature_cols
    clf.include_center_id_ = include_center_id
    return clf