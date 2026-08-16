# 🚗 Car Price Prediction using Machine Learning

A complete, beginner-friendly Machine Learning project that predicts the selling price of used cars in India. Built for classroom training — every step is explained clearly.

---

## 📌 Project Overview

Given features like the car's brand, manufacturing year, fuel type, and kilometres driven, this project trains multiple ML regression models and deploys the best one as a **Streamlit web application**.

**This project teaches:**
- Exploratory Data Analysis (EDA)
- Feature Engineering and Preprocessing
- Training and evaluating 5 regression models
- Automatic model selection
- Deployment with Streamlit

---

## 📁 Folder Structure

```
Car_Price_Prediction/
│
├── artifacts/              ← Saved model and transformers
│   ├── model.pkl           ← Best trained ML model
│   ├── scaler.pkl          ← StandardScaler fitted on training data
│   ├── encoder.pkl         ← Company and fuel type category lists
│   └── columns.pkl         ← Exact feature column names and order
│
├── data/
│   └── cleaned_car.csv     ← Pre-cleaned dataset (725 rows × 6 columns)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb ← Preprocessing and encoding
│   └── 03_Model_Training.ipynb      ← Training, evaluation, model selection
│
├── src/
│   ├── __init__.py         ← Makes src/ a Python package
│   ├── utils.py            ← Shared helper functions (load pickle, format price)
│   ├── preprocessing.py    ← Transforms user input for the model
│   └── prediction.py       ← Loads model and returns prediction
│
├── app.py                  ← Streamlit web application
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 📊 Dataset Description

**Source:** Quikr used car listings (India)  
**File:** `data/cleaned_car.csv`  
**Rows:** 725 | **Columns:** 6

| Column | Type | Description |
|--------|------|-------------|
| `name` | text | Full car model name |
| `company` | text | Car manufacturer (25 unique brands) |
| `year` | int | Manufacturing year (1995–2019) |
| `Price` | int | Selling price in ₹ (Target variable) |
| `kms_driven` | int | Total kilometres driven |
| `fuel_type` | text | Petrol / Diesel / LPG |

---

## 🔍 Exploratory Data Analysis (Notebook 1)

Key findings from EDA:
- **No missing values** — dataset is pre-cleaned
- **No duplicate rows** — dataset is unique
- Price is **right-skewed** — most cars priced below ₹5 lakh; luxury outliers exist
- **Maruti** and **Hyundai** are the most common brands
- **Petrol** is the dominant fuel type
- Newer cars and lower-km cars consistently fetch higher prices

---

## 🔧 Feature Engineering (Notebook 2)

| Step | Action |
|------|--------|
| Feature Selection | Dropped `name` (200+ unique values, redundant with `company`) |
| Missing Values | None found — no imputation needed |
| Encoding: `fuel_type` | One-Hot Encoding (3 binary columns) |
| Encoding: `company` | One-Hot Encoding (25 binary columns) |
| Scaling | StandardScaler on `year` and `kms_driven` |
| Train-Test Split | 80% train (580 rows), 20% test (145 rows), random_state=42 |
| **Final features** | **28 columns** |

---

## 🤖 Models Used (Notebook 3)

Five regression models are trained and evaluated individually:

| Model | Key Characteristic |
|-------|-------------------|
| Linear Regression | Fast, interpretable baseline |
| Decision Tree Regressor | Non-linear, tree-based, can overfit |
| Random Forest Regressor | Ensemble of 100 trees, robust |
| KNN Regressor | Distance-based, sensitive to scale |
| Gradient Boosting Regressor | Sequential boosting, high accuracy |

---

## 📏 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | avg(\|actual - predicted\|) | Average error in ₹ |
| MSE | avg((actual - predicted)²) | Penalizes large errors |
| RMSE | √MSE | Error in ₹ (same unit as Price) |
| R² | 1 - SSres/SStot | % of price variance explained |

---

## 🏆 Model Selection

The best model is selected **automatically** using:
- **Highest R² score** (primary criterion)
- **Lowest RMSE** (tiebreaker)

Only the best model is saved to `artifacts/model.pkl`.

In practice, **Random Forest** or **Gradient Boosting** typically wins on this dataset because:
- Car prices have non-linear relationships with features
- Tree ensembles handle outliers and feature interactions well

---

## 🚀 Deployment

The Streamlit app (`app.py`) provides a clean UI where users enter:
- Car company / brand
- Manufacturing year
- Kilometres driven
- Fuel type

And receive an **estimated price in Indian Rupees**.

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone <repo-url>
cd Car_Price_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order

```bash
jupyter notebook
```

Open and run:
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Feature_Engineering.ipynb`
3. `notebooks/03_Model_Training.ipynb`

### 4. Launch the web app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📸 Screenshots

*(Add screenshots of the Streamlit app here after running it)*

---

## 🔮 Future Improvements

- Add more features: number of previous owners, transmission type, insurance status
- Apply log transformation to `Price` to reduce skew for linear models
- Hyperparameter tuning with cross-validation (GridSearchCV)
- Deploy to Streamlit Cloud or Hugging Face Spaces
- Add model explainability (SHAP values)
- Build an API using FastAPI for production-grade serving

---

## 🎓 About This Project

This project is designed as a teaching tool for first-year Machine Learning students.  
Every notebook includes:
- Clear explanations of what, why, and how
- Real-world analogies
- Common beginner mistakes
- Interview questions

Built with Python, Scikit-learn, and Streamlit.
