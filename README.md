# Car Price Prediction App

A Streamlit application that trains a Random Forest regression pipeline from a car-sales CSV file and predicts an estimated car price.

## Project files

- `app.py` — Streamlit interface, preprocessing, model training, evaluation, and prediction.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container configuration for deployment platforms that build Docker images.
- `README.md` — project instructions.

## Dataset required

Upload your original `cars_info.csv` in the app, or place it beside `app.py` before running locally.

The CSV must contain a numeric `Price` column. The app uses the other available columns as model inputs, while excluding these identifier/high-cardinality columns when present: `ID`, `Name`, `Variant`, `Series`, and `Model`.

**Important:** The notebooks supplied with this project show a synthetic fallback dataset in one training workflow. That generated sample is not the original Kaggle dataset and its recorded R² score was `-0.1015`. The original `cars_info.csv` and saved `.pkl` model files were not included in the uploaded files. This app therefore trains a fresh pipeline from the real CSV you provide; it does not pretend that a missing pretrained model is available.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit and upload `cars_info.csv`.

## Upload to GitHub

1. Create a new repository on GitHub.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `Dockerfile`.
3. If your dataset is small enough and you are allowed to share it, you may add `cars_info.csv`. Otherwise, keep it private and upload it through the app when running.
4. Do not upload passwords, API keys, private data, or large files without checking repository limits and dataset permissions.

## Deploy

This project is a Streamlit app. A deployment platform must support Python/Streamlit or Docker containers.

- For a Streamlit-native hosting service, set the main file to `app.py` and install dependencies from `requirements.txt`.
- For a Docker-based host, build using the included `Dockerfile`.

DreamKit's exact GitHub import and deployment options depend on the DreamKit service/version you are using. If its interface asks for a start command, use:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Model evaluation

The app displays R² and Mean Absolute Error (MAE) on a held-out 20% test split. A low or negative R² indicates that the dataset/features may not support reliable price prediction. Evaluate results before using predictions in real decisions.
