import streamlit as st
import os
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Streamlit Page Config
st.set_page_config(page_title="LexAI Legal Chatbot", page_icon="⚖️")
st.title("⚖️ LexAI Legal Chatbot")

# Load Embedding Model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF Files and Split into Chunks
def load_and_chunk_pdfs(pdf_folder):
    docs = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

# Load Chroma DB
@st.cache_resource
def load_chroma():
    return Chroma(
        persist_directory="legal_docs_chromadb",
        embedding_function=load_embeddings()
    )

# Load LLM (Flan-T5 for QA)
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.0
    )
    return HuggingFacePipeline(pipeline=pipe)

# Load Vector DB
db = load_chroma()

# Index PDFs if not already
if len(db.get()["documents"]) == 0:
    st.info("Indexing PDF documents...")
    chunks = load_and_chunk_pdfs("LexAi_pdfs")  # make sure this folder exists
    db.add_documents(chunks)
    st.success("Indexing complete!")
else:
    st.success("Documents already indexed.")

# Build QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    retriever=db.as_retriever(),
    return_source_documents=True
)

# Ask a Question
query = st.text_input("Ask a legal question:")
if query:
    result = qa_chain.invoke({"query": query})
    st.write("📄 **Answer:**", result["result"])

    # Show source documents
    st.markdown("---")
    st.markdown("📚 **Source Documents:**")
    for i, doc in enumerate(result["source_documents"]):
        source = doc.metadata.get("source", "Unknown Source")
        chunk = doc.metadata.get("chunk", i + 1)
        st.markdown(f"- `{source}` (Chunk {chunk})")
