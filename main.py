import os
import streamlit as st

import ssl
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_old_request = requests.Session.request

def new_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, *args, **kwargs)

requests.Session.request = new_request

ssl._create_default_https_context = ssl._create_unverified_context

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Question & Answer")


def extract_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def build_vectorstore(text, nvidia_api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=nvidia_api_key)
    return FAISS.from_texts(chunks, embeddings)


def build_chain(vectorstore, nvidia_api_key):
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=nvidia_api_key,
        temperature=0.3,
        top_p=1,
        max_tokens=4096,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, reformulate a standalone question. Return it as-is if no reformulation needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question using only the context below.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    def get_context(inputs):
        if inputs.get("chat_history"):
            standalone = contextualize_chain.invoke(inputs)
        else:
            standalone = inputs["input"]
        return format_docs(retriever.invoke(standalone))

    chain = (
        RunnablePassthrough.assign(context=get_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain


# Sidebar
with st.sidebar:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
    st.header("Upload PDF(s)")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files and nvidia_api_key:
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                text = extract_text(uploaded_files)
                vectorstore = build_vectorstore(text, nvidia_api_key)
                st.session_state.chain = build_chain(vectorstore, nvidia_api_key)
                st.session_state.chat_history = []
            st.success(f"✅ Processed {len(uploaded_files)} file(s)")
    elif uploaded_files and not nvidia_api_key:
        st.warning("Please enter your NVIDIA API key.")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.info("Upload and process a PDF to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about your PDF...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.chat_history[:-1]
                ]
                answer = st.session_state.chain.invoke({"input": question, "chat_history": history})
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
