# AI-Powered Shopping Chatbot  

An intelligent shopping assistant chatbot that helps customers find products, answer queries, and provide store information using **LLM, RAG, and Knowledge Graph**.  

---

## Features  

- **Product Recommendation** – Retrieves product details using Neo4j Knowledge Graph.  
- **Smart Query Classification** – Uses BERT-based classifier for intent recognition.  
- **Generative AI Responses** – Powered by **Google Gemini** for natural conversations.  
- **Store Information & Terms Search** – Uses FAISS-based semantic search.  
- **Context-Aware Chat** – Maintains conversation history for a better experience.  

---

## Tech Stack  

- **Python** (Flask, FAISS, Torch, Transformers)  
- **Google Gemini API** (for AI-powered responses)  
- **Neo4j Graph Database** (for structured product queries)  
- **FAISS** (for fast semantic search)  
- **BERT** (for intent classification)  

---

## Installation  

### 1️⃣ Clone the repository  

```bash
git clone https://github.com/8qii/E-commerce-chatbot.git
cd E-commerce-chatbot

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Start the Flask server
python app.py
