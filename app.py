import streamlit as st 
import os
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


st.title("RAG Project")
api_key = st.secrets("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

prompt = ChatPromptTemplate.from_template("""
answer the question based on the context provided only
,please provide the most accurate response based on the context
                            
<context>
{context}
</context>

                            
Question : {input}
""")

def create_vectorDb():

    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectordb = Chroma.from_documents(st.session_state.final_docs, st.session_state.embeddings)

prompt1 = st.text_input("Ask anything related to the documents")



if st.button("Create Embeddings"):
    create_vectorDb()
    st.write("Vector db is ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectordb.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()

    response=retrieval_chain.invoke({'input':prompt1})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")



