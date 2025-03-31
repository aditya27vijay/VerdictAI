# 🏛️ VerdictAI - Legal Chatbot Backend  

## Overview  
VerdictAI is a legal chatbot designed to provide **accurate legal information and preliminary guidance** based on Indian legal codes, precedents, and regulations. This backend, built with **FastAPI**, integrates a **FAISS vector database** to retrieve legal responses and uses **DeepSeek-V3 via DeepInfra** for handling general queries.  

---

## ✨ Features  
✅ **Legal Query Handling** – Retrieves responses for legal questions using FAISS and a legal dataset.  
✅ **General Query Handling** – Uses DeepSeek-V3 (LLM) for non-legal queries.  
✅ **FastAPI Backend** – High-performance and scalable API.  
✅ **FAISS Vector Database** – Efficient similarity search for legal queries.  
✅ **CORS Enabled** – Allows seamless frontend integration.  

---

## 🛠 Tech Stack  
- **Backend:** FastAPI  
- **Database:** FAISS (for legal query retrieval)  
- **LLM API:** DeepSeek-V3 via DeepInfra  
- **Data Format:** CSV  

---

## 🚀 Getting Started  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/aditya27vijay/VerdictAI.git
cd VerdictAI/backend
```
## 2️⃣ Install Dependencies  
Ensure you have **Python 3.8+** installed. Then, run:  

```sh
pip install -r requirements.txt
```

## 3️⃣ Set Up Environment Variables 
 Create a .env file in the backend folder and add your API key: 
 ```sh
Copy Edit DEEPINFRA_API_KEY=your_api_key_here
```
## 4️⃣ Run the FastAPI Server 
```sh
 Copy Edit uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Your backend will be live at http://127.0.0.1:8000 🚀
