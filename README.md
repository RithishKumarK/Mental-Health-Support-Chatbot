# Mental-Health-Support-Chatbot
A fast RAG-based chatbot using Groq, LangChain, ChromaDB, and Gradio. Supports PDF document ingestion, vector search, and real-time LLM responses. Perfect for building knowledge-based AI assistants.
Mental Health Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, ChromaDB, Groq LLM, and Gradio. This chatbot can answer questions based on uploaded PDF documents or your custom data with context-aware responses.

Features

Upload PDF documents to ingest content
Store text chunks in ChromaDB vector database
Context-aware responses using Groq LLM
Easy web interface with Gradio
Fully deployable on Hugging Face Spaces

Repo Structure
mental_health_chatbot/
├── app.py                  # Gradio app
├── requirements.txt        # Python dependencies
├── README.md
├── config.py               # Configurations
├── .gitignore
├── chatbot/
│   ├── __init__.py
│   ├── ingestion.py        # PDF ingestion & text chunking
│   ├── retriever.py        # Vector store retrieval
│   ├── llm_response.py     # LLM response generation
│   └── utils.py
├── data/
│   └── docs/               # Store uploaded PDFs
└── vector_store/
    └── chroma_db/          # Vector DB storage

Installation (Local)
git clone <your-repo-url>
cd mental_health_chatbot
pip install -r requirements.txt
python app.py

Then open your browser at http://127.0.0.1:7860 to use the app.

Usage
Type your question in the input box.
Upload PDFs if you want the chatbot to reference them.
Click Submit to get context-aware answers.

This is the demo video link [Final Video][https://drive.google.com/file/d/1exNxFJX5nUVPfbENQl7RF9Aq5gulh1Bh/view?usp=sharing]
