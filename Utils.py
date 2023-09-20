import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv
load_dotenv(".env")

def docLoader(document):
    pdf_reader = PdfReader(document)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text  

def docsplitter(docs, chunk_size=1000,chunk_overlap=100 ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk = text_splitter.split_text(docs)  
    return chunk

def docembedding(chunk):
    embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    pinecone.init(api_key = os.getenv("api_key"), environment= os.getenv("environment") )
    index = Pinecone.from_texts(chunk, embedding_model, index_name=os.getenv("index_name"))
    return index

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def find_match(query, index):
    similar_docs = index.similarity_search(query, k=2)
    return similar_docs
