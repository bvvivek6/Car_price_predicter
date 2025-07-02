# Car Price Prediction Project Report

## 📋 Project Overview

This project implements a **Machine Learning-based Car Price Prediction System** that uses regression algorithms to predict the selling price of used cars based on various features. The system provides both a web interface and an API for making predictions using two different ML models.

## 🎯 Objective

The main objective is to develop an accurate car price prediction system that can help users estimate the selling price of used cars based on:

- Car specifications (Year, Present Price, Kilometers Driven)
- Categorical features (Fuel Type, Seller Type, Transmission)
- Ownership history

## 📊 Dataset Information

### Dataset Structure

- **Source**: `car data.csv`
- **Total Features**: 9 columns
- **Target Variable**: `Selling_Price` (in Lakhs)

### Features Description

| Feature         | Type        | Description                         | Example Values      |
| --------------- | ----------- | ----------------------------------- | ------------------- |
| `Car_Name`      | Categorical | Brand/Model name                    | ritz, sx4, ciaz     |
| `Year`          | Numerical   | Manufacturing year                  | 2011-2017           |
| `Selling_Price` | Numerical   | **Target** - Selling price in Lakhs | 2.85, 4.75, 7.25    |
| `Present_Price` | Numerical   | Current showroom price in Lakhs     | 4.15, 9.54, 9.85    |
| `Kms_Driven`    | Numerical   | Total kilometers driven             | 5200, 27000, 43000  |
| `Fuel_Type`     | Categorical | Type of fuel                        | Petrol, Diesel, CNG |
| `Seller_Type`   | Categorical | Type of seller                      | Dealer, Individual  |
| `Transmission`  | Categorical | Transmission type                   | Manual, Automatic   |
| `Owner`         | Numerical   | Number of previous owners           | 0, 1, 2, 3          |

## 🔄 Data Preprocessing

### 1. Categorical Encoding

The categorical variables were encoded using label encoding with predefined mappings:

```python
mappings = {
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Transmission': {'Manual': 0, 'Automatic': 1},
    'Seller_Type': {'Dealer': 0, 'Individual': 1}
}
```

### 2. Feature Selection

- **Dropped Features**: `Car_Name` (high cardinality, not directly predictive)
- **Target Variable**: `Selling_Price`
- **Input Features**: All remaining 7 features

### 3. Train-Test Split

- **Training Set**: 90% of the data
- **Test Set**: 10% of the data
- **Random State**: 2 (for reproducibility)

## 🤖 Machine Learning Models

### 1. Linear Regression

**Algorithm**: Ordinary Least Squares Linear Regression

**Characteristics**:

- Simple and interpretable model
- Assumes linear relationship between features and target
- Fast training and prediction
- Good baseline model

**Implementation**:

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
```

### 2. K-Nearest Neighbors (KNN) Regression

**Algorithm**: KNN Regression with k=5 neighbors

**Characteristics**:

- Non-parametric algorithm
- Makes predictions based on similarity to nearest neighbors
- Can capture non-linear relationships
- More flexible than linear regression

**Implementation**:

```python
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
```

**Hyperparameters**:

- `n_neighbors`: 5 (number of nearest neighbors to consider)

## 📈 Model Evaluation

### Evaluation Metric

**Mean Squared Error (MSE)** was used to evaluate model performance:

- Measures average squared difference between actual and predicted values
- Lower values indicate better model performance
- Suitable for regression problems

### Performance Comparison

The project includes visualization comparing training and test errors between both models using bar charts to help identify:

- Model accuracy
- Overfitting/underfitting patterns
- Best performing algorithm

## 🏗️ System Architecture

### 1. Backend (API Service)

**Framework**: FastAPI
**File**: `services/api.py`

**Features**:

- RESTful API with CORS support
- Model serialization using joblib
- Real-time predictions for both models
- Input validation using Pydantic

**API Endpoints**:

- `GET /` - Welcome message
- `POST /predict` - Car price prediction

**Input Schema**:

```json
{
  "Year": 2020,
  "Present_Price": 10.5,
  "Kms_Driven": 50000,
  "Fuel_Type": "Petrol",
  "Seller_Type": "Dealer",
  "Transmission": "Manual",
  "Owner": 1
}
```

**Output Schema**:

```json
{
  "knn_prediction": 8.45,
  "linear_regression_prediction": 8.32
}
```

### 2. Frontend (Web Interface)

**Technologies**: HTML5, CSS3, JavaScript (Vanilla)
**File**: `frontend/index.html`

**Features**:

- Modern, responsive web interface
- Interactive form with validation
- Model selection (Linear Regression/KNN)
- Real-time predictions with loading states
- Modal dialog for results display
- Error handling and user feedback

**UI Components**:

- Form inputs for all car features
- Dropdown selectors for categorical variables
- Prediction button with loading animation
- Results display with currency formatting
- Modal popup for prediction results

## 📁 Project Structure

```
ML_project/
├── car data.csv                    # Dataset
├── main.ipynb                      # ML pipeline & model training
├── requirements.txt                # Python dependencies
├── PROJECT_REPORT.md              # This documentation
├── frontend/
│   ├── index.html                 # Web interface
│   ├── style.css                  # Styling
│   └── favicon.png               # Website icon
└── services/
    ├── api.py                     # FastAPI backend
    └── models/
        ├── knn_regression_model.pkl      # Trained KNN model
        ├── linear_regression_model.pkl   # Trained Linear model
        └── mappings.pkl                  # Categorical mappings
```

## 🛠️ Technologies Used

### ML

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization

### Backend Development

- **FastAPI**: Modern web framework for APIs
- **uvicorn**: ASGI server for FastAPI
- **pydantic**: Data validation using Python type hints

### Frontend Development

- **HTML5**: Markup structure
- **CSS3**: Styling and responsive design
- **JavaScript**: Interactive functionality and API communication

## 🚀 Usage Instructions

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Start API Server

```bash
cd services
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open Web Interface

Open `frontend/index.html` in a web browser or serve it using a local server.

### 4. Make Predictions

1. Select ML model (Linear Regression or KNN)
2. Fill in car details in the form
3. Click "Predict Price" button
4. View results in the modal dialog

## 🔍 Key Features

### Machine Learning

- **Multi-model approach**: Compare Linear Regression vs KNN
- **Feature engineering**: Categorical encoding for ML compatibility
- **Model persistence**: Saved models for production use
- **Performance evaluation**: MSE-based model comparison

### Web Application

- **Real-time predictions**: Instant results via API calls
- **User-friendly interface**: Clean, intuitive design
- **Model selection**: Users can choose between algorithms
- **Error handling**: Graceful handling of API errors
- **Responsive design**: Works on desktop and mobile devices

### API Design

- **RESTful architecture**: Standard HTTP methods and status codes
- **CORS support**: Cross-origin requests enabled
- **Input validation**: Type checking and data validation
- **JSON communication**: Structured data exchange

## 📊 Model Performance Insights

The project implements comprehensive model evaluation including:

- **Training error analysis**: Understanding model fit on training data
- **Test error comparison**: Evaluating generalization capability
- **Visual comparisons**: Bar charts showing performance differences
- **Scatter plots**: Actual vs predicted price visualizations

## 🎯 Business Applications

This car price prediction system can be used for:

- **Used car dealerships**: Pricing inventory accurately
- **Individual sellers**: Setting competitive prices
- **Car buyers**: Evaluating fair market prices
- **Insurance companies**: Assessing vehicle values
- **Financial institutions**: Loan amount determination

## 📚 Learning Outcomes

This project demonstrates proficiency in:

- **Machine Learning**: Regression algorithms and model evaluation
- **Web Development**: Full-stack application development
- **API Design**: FASTAPI service architecture

---

_This report provides a comprehensive overview of the Car Price Prediction ML project, covering all technical aspects, implementation details, and system architecture._
