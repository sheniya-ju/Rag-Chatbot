
 RAG Chatbot

This is a simple RAG (Retrieval-Augmented Generation) chatbot built as part of my GenAI learning task.

The chatbot takes information from PDF files, finds relevant content based on the user's question, and uses an LLM to generate an answer.

 Technologies Used

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- Streamlit
- PyPDF

 How it works

PDF → Text Chunks → Embeddings → ChromaDB → Similarity Search → LLM → Answer

The PDF is loaded and divided into smaller chunks. The chunks are converted into embeddings using `all-MiniLM-L6-v2` and stored in ChromaDB.

When a user asks a question, the system searches for relevant chunks from the documents and sends them as context to the Groq LLM. The LLM then generates an answer based on the retrieved context.

 Features

- Upload PDF files
- Upload multiple PDFs
- Ask questions about the documents
- Retrieve relevant chunks
- Display source chunks
- Chat history
- Streamlit interface
- Logging
- Compare answers with and without RAG

 Project Structure


llm-project/
│
├── app.py
├── chat.py
├── rag.py
├── requirements.txt
├── .gitignore
│
└── pdfs/
    ├── agentic.pdf
    ├── chatbots.pdf
    └── notes.pdf


 Files

 app.py

Streamlit application that provides the user interface for the RAG chatbot.

 rag.py

Main RAG implementation. It loads the PDFs, creates chunks and embeddings, stores them in ChromaDB, and retrieves relevant information for questions.

 chat.py

A basic Groq chatbot created while learning how to work with LLM APIs.

 requirements.txt

Contains the Python packages required to run the project.

 Setup

Clone the repository:


git clone https://github.com/sheniya-ju/Rag-Chatbot.git
cd Rag-Chatbot


Create a virtual environment:


python -m venv venv


Activate the virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

 Groq API Key

Set your Groq API key as an environment variable.

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Do not upload your API key to GitHub.

## Run the Project

To run the terminal version:

```bash
python rag.py
```

To run the Streamlit application:

```bash
streamlit run app.py
```

 Models Used

Embedding model:

`all-MiniLM-L6-v2`

LLM:

`openai/gpt-oss-20b`

The original task mentioned `llama3-8b-8192`, but that model was decommissioned on Groq, so `openai/gpt-oss-20b` was used instead.

 What I Learned

Through this project, I learned about:

* RAG
* Document loading
* Text chunking
* Embeddings
* Vector databases
* Similarity search
* LLM integration
* Chat history
* Multiple PDF handling
* Streamlit
* Logging

```

After pasting and saving it on GitHub, your repo will look much more complete. 👍
```
