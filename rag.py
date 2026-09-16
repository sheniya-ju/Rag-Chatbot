import os
import glob
import logging

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

pdf_files = glob.glob("pdfs/*.pdf")

documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

print("Number of PDFs:", len(pdf_files))
print("Number of pages:", len(documents))

logging.info("Number of PDFs loaded: %s", len(pdf_files))
logging.info("Number of pages loaded: %s", len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

logging.info("Number of chunks created: %s", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded successfully.")
logging.info("Embedding model loaded successfully")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Documents stored in ChromaDB successfully.")
logging.info("ChromaDB loaded successfully")


def get_answer_without_rag(question):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content


chat_history = []

while True:
    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        print("Chat ended.")
        logging.info("Chat ended")
        break

    logging.info("User question: %s", question)

    search_query = question

    if chat_history:
        previous_question = chat_history[-2]["content"]
        search_query = previous_question + " " + question

    retrieved_docs = vectorstore.similarity_search(
        search_query,
        k=3
    )

    logging.info("Retrieved %s chunks", len(retrieved_docs))

    print("\n--- Retrieved Chunks ---")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i + 1}:")
        print(doc.page_content)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    answer_without_rag = get_answer_without_rag(question)

    prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I could not find the answer in the provided document."
"""

    messages = [
        {
            "role": "system",
            "content": "Answer the user's question using only the provided context."
        }
    ]

    messages.extend(chat_history)

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

    logging.info("Answer generated successfully")

    print("\n--- Answer Without RAG ---")
    print(answer_without_rag)

    print("\n--- Answer With RAG ---")
    print(answer)

    print("\n--- Sources ---")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nSource {i + 1}:")
        print(doc.page_content)

    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })