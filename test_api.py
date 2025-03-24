# test_api.py - Script to test API endpoints

import requests

BASE_URL = "https://web-production-88853.up.railway.app"

# Test /predict endpoint
def test_predict():
    data = {"text1": "The cat sits on the mat.", "text2": "A feline is resting on the carpet."}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Predict API Response:", response.json())

# Test /test-dataset endpoint
def test_dataset(row_id=0):
    response = requests.get(f"{BASE_URL}/test-dataset/{row_id}")
    print("Test Dataset API Response:", response.json())

if __name__ == "__main__":
    test_predict()
    test_dataset(2)  
