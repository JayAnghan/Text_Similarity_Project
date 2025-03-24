# app.py - FastAPI script with CSV support

import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
DATA_PATH = "DataNeuron_Text_Similarity.csv"
df = pd.read_csv(DATA_PATH)

def compute_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings1, embeddings2).item()

# Initialize FastAPI app
app = FastAPI()

# Define request structure
class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/predict")
def predict_similarity(data: TextPair):
    similarity_score = compute_similarity(data.text1, data.text2)
    return {"similarity score": similarity_score}

@app.get("/test-dataset/{row_id}")
def test_dataset(row_id: int):
    if row_id >= len(df):
        return {"error": "Row ID out of range"}
    text1, text2 = df.iloc[row_id][0], df.iloc[row_id][1]
    similarity_score = compute_similarity(text1, text2)
    return {"text1": text1, "text2": text2, "similarity score": similarity_score}

# Run the API (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
