# app.py - FastAPI script for Semantic Similarity API

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FastAPI app
app = FastAPI()

# Define request structure
class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/predict")
def predict_similarity(data: TextPair):
    # Generate embeddings
    embeddings1 = model.encode(data.text1, convert_to_tensor=True)
    embeddings2 = model.encode(data.text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    
    return {"similarity score": similarity_score}

# Run the API (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
