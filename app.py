import io
import os
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Prediction")
st.write("Upload your car-sales CSV dataset, train the model, and estimate a car's price.")

TARGET = "Price"
DROP_COLUMNS = ["ID", "Name", "Variant", "Series", "Model"]


@st.cache_resource(show_spinner="Training the car price model...")
def train_model(csv_bytes: bytes):
    """Train a preprocessing + Random Forest pipeline from the uploaded CSV."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if TARGET not in df.columns:
        raise ValueError(f"The CSV must contain a '{TARGET}' column.")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    if len(df) < 10:
        raise ValueError("At least 10 valid rows are required to train and test the model.")

    features = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])
    if features.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing ID/name columns.")

    # Remove columns that contain no usable values.
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("The CSV has no usable feature values.")

    y = df.loc[features.index, TARGET]
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in features.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rows": len(df),
        "features": list(features.columns),
        "numeric": numeric_features,
        "categorical": categorical_features,
        "sample": features
    }
    return pipeline, metrics


uploaded_file = st.file_uploader("Upload cars_info.csv", type=["csv"])
if uploaded_file is not None:
    csv_bytes = uploaded_file.getvalue()
elif os.path.exists("cars_info.csv"):
    with open("cars_info.csv", "rb") as f:
        csv_bytes = f.read()
    st.info("Using cars_info.csv found in the project folder.")
else:
    csv_bytes = None
    st.warning("Please upload your original cars_info.csv dataset to continue.")

if csv_bytes:
    try:
        model, info = train_model(csv_bytes)
        st.success(f"Model trained using {info['rows']:,} dataset rows.")
        c1, c2 = st.columns(2)
        c1.metric("R² score", f"{info['r2']:.3f}")
        c2.metric("Mean Absolute Error (MAE)", f"{info['mae']:,.2f}")
        st.caption("A negative R² means the model performs worse than predicting the test-set average. Check the data and model features before relying on estimates.")

        st.subheader("Enter car details")
        sample = info["sample"]
        with st.form("prediction_form"):
            values = {}
            cols = st.columns(2)
            for i, feature in enumerate(info["features"]):
                with cols[i % 2]:
                    if feature in info["numeric"]:
                        series = pd.to_numeric(sample[feature], errors="coerce").dropna()
                        default = float(series.median()) if not series.empty else 0.0
                        values[feature] = st.number_input(
                            feature, value=default, format="%.2f"
                        )
                    else:
                        options = sorted(sample[feature].dropna().astype(str).unique().tolist())
                        if not options:
                            options = ["Unknown"]
                        values[feature] = st.selectbox(feature, options)

            submitted = st.form_submit_button("Predict car price", type="primary")

        if submitted:
            input_df = pd.DataFrame([values], columns=info["features"])
            prediction = float(model.predict(input_df)[0])
            st.subheader("Estimated price")
            st.metric("Predicted Price", f"{prediction:,.2f}")
            st.caption("This is an estimate based on the dataset you supplied, not a guaranteed market price.")

        with st.expander("Dataset preview"):
            st.dataframe(sample.head(10), use_container_width=True)

    except Exception as exc:
        st.error(f"Could not train the model: {exc}")
        st.info("Check that the CSV is valid, contains a numeric 'Price' column, and has usable feature columns.")
