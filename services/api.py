from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import numpy as np

# Use relative paths for model files
mappings = joblib.load('./models/mappings.pkl')
knn_model = joblib.load('./models/knn_regression_model.pkl')
linear_model = joblib.load('./models/linear_regression_model.pkl')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {"message": "Welcome to the API"}

class CarFeatures(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int

@app.post('/predict')
def predict(features: CarFeatures):
    # Encode categorical features
    fuel = mappings['Fuel_Type'][features.Fuel_Type]
    seller = mappings['Seller_Type'][features.Seller_Type]
    trans = mappings['Transmission'][features.Transmission]
    # Prepare input for model
    input_data = np.array([[features.Year, features.Present_Price, features.Kms_Driven, fuel, seller, trans, features.Owner]])
    # Predict
    knn_pred = knn_model.predict(input_data)[0]
    linear_pred = linear_model.predict(input_data)[0]
    return {
        'knn_prediction': float(knn_pred),
        'linear_regression_prediction': float(linear_pred)
    }