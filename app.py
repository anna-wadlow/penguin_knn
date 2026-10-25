import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report)

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
)

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipped_length_mm",
    "body_mass_g"
]

CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "species"

@st.cache_data
def load_data():
    df = pd.read(DATA_URL)
    # Select columns that app needs, drop incomplete rows, and clean up the row numbers.
    return df[FEATURES + [TARGET]].dropna().reset_index(drop=True)

@st.cache_resource
def train_model():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, stratify=y, random_state=813
    )

    # Apply a different preprocessing step to different groups of columns. Then combine results into one array.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier())
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={"knn__n_neighbors": range(1,30)},
        cv=5,
        scoring="accuracy"
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_k = grid_search.best_params_["knn_n_neighbors"]
    cv_score = grid_search.best_score_
    test_acc = best_model.score(X_test, y_test)
    return best_model, best_k, cv_score, test_acc, X_train, y_train

df = load_data()
model, best_k, cv_score, test_acc, X_train, y_train = train_model()

# -----------------------------
# Page title
# -----------------------------

st.set_page_config(
    page_title="kNN Classifier",
    page_icon="🐧",
    layout="wide"
)

st.title("Palmer Penguins Species Classifier")
st.write(
    "This app trains a k-nearest neighbors (kNN) classifier on the Palmer Penguins dataset "
    "to predict a penguin's species based on their physical measurements, island, and sex. "
    "Rows with NA values are dropped, numerical values are standardized, and categorical variables are "
    "encoded prior to data classification."
)


# -----------------------------
# Model info
# -----------------------------

st.subheader("Model")
col1, col2, col3 = st.columns(3)
col1.metric("Best K selected by 5-fold cross-validation", best_k)
col2.metric("Best CV accuracy", f"{cv_score:.4f}")
col3.metric("Test Accuracy", f"{test_acc:.4f}")

# -----------------------------
# Inputs
# -----------------------------

st.subheader("Enter penguin characteristics")

def num_input(label, col):
    return st.slider(
        label,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].median()),
        step=0.05
    )

left, right = st.columns(2)
with left:
    bill_length = num_input("Bill length (mm)", "bill_length_mm")
    bill_depth = num_input("Bill depth (mm)", "bill_depth_mm")
    flipper_length = num_input("Flipper Length (mm)", "flipper_length_mm")
    body_mass = num_input("Body Mass (g)", "body_mass_g")

with right:
    island = st.selectbox("Island", sorted(df["island"].unique()))
    sex = st.selectbox("Sex", sorted(df["sex"].unique()))

new_penguin = pd.DataFrame(
    [
        {
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass,
            "island": island,
            "sex": sex
        }
    ]
)[FEATURES]


# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Species"):
    prediction = model.predict(new_penguin)[0]
    probabilities = model.predict_proba(new_penguin)[0]
    classes = model.classes_

    st.subheader("Prediction")
    st.success(f"Predicted species: **{prediction}**")

    st.subheader("Predicted probabilities")
    prob_df = pd.DataFrame({"Species": classes, "Probability": probabilities})
    st.dataframe(
        prob_df.style.format({"Probability":{:.4f}}),
        hide_index=True
    )
    st.bar_chart(prob_df.set_index("Species"))

    st.subheader("Prediction Explained")
    preprocessor = model.named_steps["preprocessor"]
    knn = model.named_steps["knn"]
    new_transformed = preprocessor.transform(new_penguin)
    _, neighbor_idx = knn.kneighbors(new_transformed)
    neighbor_labels = y_train.iloc[neighbor_idx[0]]
    counts = neighbor_labels.value_counts()
    count_text = ", ".join(f"{n} {s}" for s, n in counts.items())