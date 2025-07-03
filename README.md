# 🚗 Car Price Predictor

A machine learning-powered web application that predicts used car prices using Linear Regression and K-Nearest Neighbors (KNN) algorithms.

![Project Banner](https://img.shields.io/badge/ML-Car%20Price%20Prediction-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)

## 🎯 Overview

This project implements a comprehensive car price prediction system that helps users estimate the selling price of used cars based on various features such as year, present price, kilometers driven, fuel type, seller type, transmission, and ownership history.

The system provides:

- **Dual ML Models**: Linear Regression and KNN Regression for comparison
- **Web Interface**: User-friendly frontend for easy interaction
- **REST API**: Backend service for programmatic access
- **Real-time Predictions**: Instant price estimates with loading animations

## ✨ Features

### 🤖 Machine Learning

- **Two ML Models**: Linear Regression and KNN Regression
- **Model Comparison**: Side-by-side performance evaluation
- **Feature Engineering**: Categorical encoding and preprocessing
- **Model Persistence**: Serialized models for production use

### 🌐 Web Application

- **Responsive Design**: Works on desktop and mobile devices
- **Interactive UI**: Modern, clean interface with form validation
- **Real-time Results**: Instant predictions with loading states
- **Error Handling**: Graceful error management and user feedback
- **Model Selection**: Choose between Linear Regression and KNN

### 🔧 API Service

- **FastAPI Backend**: High-performance async API
- **CORS Support**: Cross-origin resource sharing enabled
- **Input Validation**: Pydantic models for data validation
- **JSON Communication**: Structured request/response format

## 🎮 Demo

### Web Interface

1. Select your preferred ML model (Linear Regression or KNN)
2. Enter car details in the form
3. Click "Predict Price" to get instant results
4. View predictions in a modal dialog

### API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "Year": 2020,
  "Present_Price": 10.5,
  "Kms_Driven": 50000,
  "Fuel_Type": "Petrol",
  "Seller_Type": "Dealer",
  "Transmission": "Manual",
  "Owner": 1
}'
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter notebook** (Optional - for model training)

```bash
jupyter notebook main.ipynb
```

4. **Start the API server**

```bash
cd services
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

5. **Open the web interface**
   - Open `frontend/index.html` in your web browser
   - Or serve it using a local server:
   ```bash
   cd frontend
   python -m http.server 3000
   ```

## 💻 Usage

### Web Interface

1. Navigate to `frontend/index.html`
2. Fill in the car details form:
   - **Year**: Manufacturing year (e.g., 2020)
   - **Present Price**: Current showroom price in Lakhs
   - **Kms Driven**: Total kilometers driven
   - **Fuel Type**: Petrol, Diesel, or CNG
   - **Seller Type**: Dealer or Individual
   - **Transmission**: Manual or Automatic
   - **Owner**: Number of previous owners
3. Select ML model (Linear Regression or KNN)
4. Click "Predict Price" to get results

### API Endpoints

#### Health Check

```http
GET /
```

Returns a welcome message to verify API is running.

#### Predict Car Price

```http
POST /predict
```

**Request Body:**

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

**Response:**

```json
{
  "knn_prediction": 8.45,
  "linear_regression_prediction": 8.32
}
```

## 📁 Project Structure

```
car-price-predictor/
├── 📊 car data.csv
├── 📓 main.ipynb
├── 📋 requirements.txt
├── 📖 README.md
├── 📄 PROJECT_REPORT.md
├── 🌐 frontend/
│   ├── 🏠 index.html
│   ├── 🎨 style.css
│   └── 🖼️ favicon.png
└── 🔧 services/
    ├── 🚀 api.py
    └── 🤖 models/
        ├── 📦 knn_regression_model.pkl
        ├── 📦 linear_regression_model.pkl
        └── 📦 mappings.pkl
```

## 🧠 Models

### Linear Regression

- **Type**: Parametric regression algorithm
- **Use Case**: Baseline model for linear relationships
- **Advantages**: Simple, interpretable, fast training
- **Performance**: Good for linear patterns in data

### K-Nearest Neighbors (KNN)

- **Type**: Non-parametric regression algorithm
- **Parameters**: k=5 neighbors
- **Use Case**: Capturing non-linear relationships
- **Advantages**: Flexible, no assumptions about data distribution

### Model Evaluation

- **Metric**: Mean Squared Error (MSE)
- **Validation**: Train-test split (90%-10%)
- **Comparison**: Visual performance comparison charts

## 🛠️ Technologies Used

### Backend

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation using Python type hints
- **Joblib**: Model serialization and deserialization

### Machine Learning

- **scikit-learn**: ML algorithms and utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

### Frontend

- **HTML5**: Structure and markup
- **CSS3**: Styling and responsive design
- **JavaScript**: Interactive functionality and API calls

## 📊 Dataset

The dataset contains information about used cars with the following features:

| Feature       | Type    | Description                      |
| ------------- | ------- | -------------------------------- |
| Car_Name      | String  | Brand and model name             |
| Year          | Integer | Manufacturing year               |
| Selling_Price | Float   | Target variable (price in Lakhs) |
| Present_Price | Float   | Current showroom price           |
| Kms_Driven    | Integer | Total kilometers driven          |
| Fuel_Type     | String  | Petrol, Diesel, or CNG           |
| Seller_Type   | String  | Dealer or Individual             |
| Transmission  | String  | Manual or Automatic              |
| Owner         | Integer | Number of previous owners        |

**Dataset Size**: 301 records with 9 features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
