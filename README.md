# Chatbot
A customized chatbot for extracting and analyzing any text file

PDF Comparison and Analysis Tool

Overview

This project involves developing a system to analyze and compare multiple PDF documents, specifically identifying and highlighting their differences. The system utilizes Retrieval Augmented Generation (RAG) techniques to effectively retrieve, assess, and generate insights from the documents.

Project Components

Backend Framework

LlamaIndex or LangChain: Choose based on your comfort level and perception of which performs better for this use case.
LlamaIndex: A flexible framework for creating custom retrieval systems.
LangChain: A powerful toolkit for building LLM applications with a strong focus on retrieval-augmented generation.

Frontend Framework

Streamlit: An open-source app framework for Machine Learning and Data Science projects, allowing you to create interactive web applications easily.

Vector Store

Faiss: Used to manage and query the embeddings. Other options include ChromaDB, Pinecone, Milvus, Weaviate, etc.

Embedding Model

sentence-transformers/all-MiniLM-L6-v2: An embedding model for generating vectors from the PDF file content. Ensure the embedding model runs locally and is not exposed to any external services or APIs.

Local Language Model (LLM)

A local instance of a Large Language Model for processing and generating insights. Ensure the LLM runs locally and is not exposed to any external APIs.

Initialization

You are provided three PDF documents containing the Form 10-K filings of multinational companies. These documents will serve as the basis for your comparison analysis. The documents are as follows:

Alphabet Inc. Form 10-K
Tesla, Inc. Form 10-K
Uber Technologies, Inc. Form 10-K

Your task is to retrieve the content from these PDFs, compare them, and answer queries highlighting the information across all documents. Additionally, the end system should feature a chatbot interface where users can interact and obtain insights about information from the documents, compare numbers within these three documents, and more.

Sample Questions
What are the risk factors associated with Google and Tesla?
What is the total revenue for Google Search?
What are the differences in the business of Tesla and Uber?

Development Steps

1. Parse Documents
Extract text and structure from PDFs.

2. Generate Vectors
Use a local embedding model to create embeddings for document content.

3. Store in Vector Store
Utilize local persisting methods in the chosen vector store.

4. Configure Query Engine
Set up retrieval tasks based on document embeddings.

5. Integrate LLM
Run a local instance of a Large Language Model for contextual insights.

6. Develop Chatbot Interface
Use Streamlit to facilitate user interaction and display comparative insights.


Deploying on Streamlit Cloud
Create a GitHub Repository: Push your code to a GitHub repository.
Create requirements.txt: List all dependencies.
Deploy to Streamlit Cloud: Connect to your GitHub repository and deploy
