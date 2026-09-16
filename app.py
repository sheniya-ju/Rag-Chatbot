import os
import streamlit as st

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 RAG AI Chatbot")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:

    documents = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        temp_path = f"temp_{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        loader = PyPDFLoader(temp_path)

        documents.extend(loader.load())

        os.remove(temp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    st.write("Pages loaded:", len(documents))
    st.write("Chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_documents"
    )

    st.session_state.vectorstore = vectorstore

    st.success(
        f"{len(uploaded_files)} PDF(s) loaded successfully."
    )

if st.session_state.vectorstore:

    question = st.text_input(
        "Ask a question about your PDF"
    )

    if question:

        retrieved_docs = (
            st.session_state.vectorstore
            .similarity_search(
                question,
                k=3
            )
        )

        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."
"""

        messages = [
            {
                "role": "system",
                "content": "Answer only using the provided context."
            }
        ]

        messages.extend(
            st.session_state.chat_history
        )

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for i, doc in enumerate(retrieved_docs):

            st.write(
                f"**Source {i + 1}**"
            )

            st.write(
                doc.page_content
            )

else:

    st.info(
        "Please upload at least one PDF to start."
    )