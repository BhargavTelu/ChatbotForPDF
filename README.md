# 📄 ChatBot for Documents

An AI-powered chatbot that allows users to upload PDF documents and ask questions using natural language. The system uses Retrieval-Augmented Generation (RAG), vector embeddings, and OpenAI LLM to provide context-aware responses.

---

## 🚀 Features

- Upload PDF and ask questions
- Context-aware responses using RAG
- Conversational memory for follow-up queries
- Vector search using Pinecone
- Interactive UI built with Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenAI API
- LangChain
- Pinecone
- HuggingFace Transformers

---

## 📂 Project Structure
├── app.py # Main Streamlit application

├── Utils.py # Document processing and embeddings

├── .env # Environment variables

├── requirements.txt # Dependencies

└── README.md

## Install dependencies:
- pip install -r requirements.txt

## ▶️ Run the App
- streamlit run app.py

## How It Works
-Upload PDF document

-Text is split into chunks

-Embeddings are created and stored in Pinecone

-Relevant context is retrieved based on query

-OpenAI generates a response using the context

## Use Cases

- Document analysis

- Research assistance

- Knowledge base chatbot

- Business document querying
