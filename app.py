import os
import requests
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index & responses
FAISS_INDEX_PATH = "legal_faiss.index"
EMBEDDINGS_PATH = "instruction_embeddings.npy"
RESPONSES_CSV_PATH = "legal_responses.csv"

# Load the sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_PATH}")

index = faiss.read_index(FAISS_INDEX_PATH)

# Load responses CSV
if not os.path.exists(RESPONSES_CSV_PATH):
    raise FileNotFoundError(f"Responses CSV file not found: {RESPONSES_CSV_PATH}")

df = pd.read_csv(RESPONSES_CSV_PATH)

# Ensure column names are correct
df.rename(columns={"Instruction": "instruction", "Response": "response"}, inplace=True)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define API route
@app.post("/chat")
async def chat(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Convert user query to embedding
    query_embedding = model.encode(query).reshape(1, -1)

    # Perform FAISS search
    _, indices = index.search(query_embedding, k=1)  # Get the closest match

    # Retrieve response
    matched_index = indices[0][0]
    if matched_index == -1 or matched_index >= len(df):
        return {"response": "Sorry, I couldn't find relevant legal information."}

    matched_response = df.iloc[matched_index]["response"]

    return {"response": matched_response}

