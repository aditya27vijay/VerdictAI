import os
import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key for DeepInfra (LLM)
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# Load FAISS index and legal dataset
FAISS_INDEX_PATH = "legal_faiss.index"
LEGAL_RESPONSES_CSV = "legal_responses.csv"
EMBEDDINGS_PATH = "instruction_embeddings.npy"

# Load FAISS index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file '{FAISS_INDEX_PATH}' not found!")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Load legal responses
legal_data = pd.read_csv(LEGAL_RESPONSES_CSV)
if "instruction" not in legal_data.columns or "response" not in legal_data.columns:
    raise ValueError("CSV file must have 'instruction' and 'response' columns.")

legal_responses = legal_data["response"].tolist()

# Load precomputed instruction embeddings
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"Embeddings file '{EMBEDDINGS_PATH}' not found!")

instruction_embeddings = np.load(EMBEDDINGS_PATH)
embedding_dim = instruction_embeddings.shape[1]  # Embedding size

# Load sentence embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Check FAISS index dimensions
if faiss_index.d != embedding_dim:
    raise ValueError(f"FAISS index dimension ({faiss_index.d}) does not match embedding dimension ({embedding_dim})")

# Define request model
class QueryRequest(BaseModel):
    query: str

# Function to search FAISS
def search_faiss(query_embedding, threshold=0.7):  # Lowered threshold to 0.6
    """Searches FAISS for the most relevant legal response."""
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k=1)
    
    print(f"FAISS Search - Distance: {distances[0][0]}, Index: {indices[0][0]}")  # Debugging log

    if distances[0][0] > threshold:  # If distance is high, it's not a good match
        return None  
    
    return legal_responses[indices[0][0]]

# Function to get LLM response from DeepSeek-V3
def get_llm_response(query):
    """Fetches response from LLM if FAISS doesn't return relevant legal results."""
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for general topics."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 500
    }
    
    response = requests.post(DEEPINFRA_API_URL, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Sorry, I am unable to process your request right now."

# Function to classify if a query is legal
def is_legal_query(query):
    """Classifies if a query is legal by checking similarity with legal instructions."""
    query_embedding = embedding_model.encode([query])[0]
    similarity_scores = np.dot(instruction_embeddings, query_embedding) / (
        np.linalg.norm(instruction_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    return np.max(similarity_scores) > 0.6  # Adjusted threshold

# Chat endpoint
@app.post("/chat")
async def chat(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Convert query to FAISS-compatible embedding
    query_embedding = embedding_model.encode([query])[0].astype("float32").reshape(1, -1)

    # Check if the query is legal
    if is_legal_query(query):
        # Search FAISS for legal response
        legal_response = search_faiss(query_embedding)

        if legal_response:
            return {"response": legal_response}
        else:
            return {"response": "No relevant legal information found."}

    # If not a legal query, use LLM
    return {"response": get_llm_response(query)}
