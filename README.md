# PRODIGY_ML_01 - Bengaluru House Price Prediction

This project predicts Bengaluru house prices using a machine learning regression pipeline. It takes property details such as area type, availability, location, size, society, total square feet, bathrooms, and balconies, then returns an estimated price in lakhs through a Streamlit web interface.

The project was built as part of the Prodigy ML internship task, but it is structured like a small end-to-end ML application rather than only a notebook experiment. The code includes data ingestion, preprocessing, model training, saved artifacts, a prediction pipeline, and an interactive app.

## Project Overview

The goal is to estimate house prices from real estate features. The workflow starts from the raw house data, splits it into training and testing sets, transforms numeric and categorical columns, trains several regression models, saves the best-performing model, and uses that saved model inside the Streamlit app.

The current app is built with Streamlit, so there is no separate Flask server or HTML template required. Running `streamlit run app.py` launches the complete interface.

## What the App Does

- Loads the saved model from `artifacts/model.pkl`
- Loads the saved preprocessing object from `artifacts/preprocessor.pkl`
- Reads available dropdown/search values from `artifacts/data.csv`
- Lets the user enter property details in a clean Streamlit interface
- Converts the user input into the same format used during training
- Applies preprocessing automatically
- Predicts and displays the estimated house price in lakhs

## Input Features

The model uses these columns:

| Feature | Description |
| --- | --- |
| `area_type` | Type of property area, such as built-up or super built-up |
| `availability` | Availability status of the property |
| `location` | Property location in Bengaluru |
| `size` | BHK or bedroom count as listed in the dataset |
| `society` | Society or apartment project name |
| `total_sqft` | Total area in square feet |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |

The target column is:

| Target | Description |
| --- | --- |
| `price` | House price in lakhs |

## Machine Learning Pipeline

The project is divided into reusable components:

### 1. Data Ingestion

File: `src/Components/data_ingestion.py`

This component reads the dataset from:

```text
src/Notebooks/Data/House_Data.csv
```

It then creates:

```text
artifacts/data.csv
artifacts/train.csv
artifacts/test.csv
```

### 2. Data Transformation

File: `src/Components/data_transformation.py`

This component prepares the data for model training.

Numeric columns:

```text
total_sqft, bath, balcony
```

Categorical columns:

```text
area_type, availability, location, size, society
```

The preprocessing steps include:

- Median imputation for numeric columns
- Standard scaling for numeric columns
- Most-frequent imputation for categorical columns
- One-hot encoding for categorical columns
- Unknown category handling during prediction

The fitted preprocessor is saved as:

```text
artifacts/preprocessor.pkl
```

### 3. Model Training

File: `src/Components/model_trainer.py`

The training component compares multiple regression models:

- Linear Regression
- Lasso
- Ridge
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

The model with the best test score is saved as:

```text
artifacts/model.pkl
```

### 4. Prediction Pipeline

File: `src/Pipeline/predict_pipeline.py`

This file contains:

- `CustomData`: converts user input into a pandas DataFrame
- `PredictPipeline`: loads the model and preprocessor, transforms input, and returns the prediction

The model and preprocessor are cached in memory after the first load, so predictions after the first one are faster.

### 5. Streamlit Interface

File: `app.py`

The Streamlit app provides the user interface. It includes:

- Area type dropdown
- Availability dropdown
- Location search
- Size selector
- Square feet input
- Bathroom and balcony inputs
- Optional society search
- Prediction result panel

## Project Structure

```text
PRODIGY_ML_01/
│
├── app.py
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
│
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── logs/
│   └── log files generated while running components
│
└── src/
    ├── __init__.py
    ├── exception.py
    ├── logger.py
    ├── utils.py
    │
    ├── Components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    │
    ├── Pipeline/
    │   ├── __init__.py
    │   ├── train_pipeline.py
    │   └── predict_pipeline.py
    │
    └── Notebooks/
        ├── 1.EDA.ipynb
        ├── 2.Model_Training.ipynb
        └── Data/
            ├── House_Data.csv
            └── house_data_cleaned.csv
```

## How to Run the Project

Open a terminal in the project folder:

```powershell
cd C:\Users\Pruthviraj\PRODIGY_ML_01
```

### 1. Create and Activate a Virtual Environment

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train the Model

Run this only when you want to regenerate the model and preprocessing artifacts:

```powershell
python src\Pipeline\train_pipeline.py
```

This command creates or updates:

```text
artifacts/data.csv
artifacts/train.csv
artifacts/test.csv
artifacts/preprocessor.pkl
artifacts/model.pkl
```

### 4. Run the Streamlit App

```powershell
streamlit run app.py
```

If that command is not recognized, use:

```powershell
python -m streamlit run app.py
```

After running the command, open the local URL shown in the terminal. It is usually:

```text
http://localhost:8501/
```

## Typical Usage Flow

1. Start the Streamlit app.
2. Select the property area type.
3. Choose the availability status.
4. Search and select a location.
5. Select the size, such as `2 BHK` or `3 BHK`.
6. Enter total square feet.
7. Enter bathroom and balcony counts.
8. Optionally select a society.
9. Click `Predict Price`.
10. View the estimated price in lakhs.

## Important Notes

- The prediction value is an estimate, not a guaranteed market price.
- The model can handle unseen categorical values because `OneHotEncoder` uses `handle_unknown="ignore"`.
- The first prediction may take a little longer because the model and preprocessor are loaded from disk.
- Later predictions should be faster because the prediction pipeline caches loaded artifacts.
- If you retrain the model, the existing `model.pkl` and `preprocessor.pkl` files will be overwritten.

## Troubleshooting

### Streamlit command not found

Run Streamlit through Python:

```powershell
python -m streamlit run app.py
```

### Missing model or preprocessor file

Run the training pipeline:

```powershell
python src\Pipeline\train_pipeline.py
```

### Pickle or scikit-learn version warnings

If you see warnings while loading `model.pkl` or `preprocessor.pkl`, retrain the model in your current environment:

```powershell
python src\Pipeline\train_pipeline.py
```

This makes the saved artifacts match the installed scikit-learn version.

### App feels slow on first prediction

The first prediction loads the saved model and preprocessor. This is normal. Once loaded, the app reuses them for future predictions.

## Future Improvements

Some useful improvements that can be added later:

- Add model performance charts in the Streamlit app
- Store the best model name and score in a separate report file
- Add validation for unrealistic square feet or room counts
- Add better feature engineering for location and size
- Add deployment instructions for Streamlit Community Cloud
- Add automated tests for the training and prediction pipelines

## Author

Pruthviraj

## License

This project is licensed under the terms included in the `LICENSE` file.
