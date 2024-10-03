# app.py

import streamlit as st
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import wikipediaapi

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API key in LangChain
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Streamlit app title
st.title("Wimbledon 2023 Q&A Chatbot")

# User input for Wikipedia article
url = st.text_input("Enter a Wikipedia article URL:", "https://en.wikipedia.org/wiki/2023_Wimbledon_Championships")

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

def load_data_from_wikipedia(url):
    # Specify a user agent string for the request (to comply with Wikipedia's guidelines)
    user_agent = 'WikiQnA/1.0 (https://github.com/bryankahlil/wikiqna; bryankahlil@gmail.com)'
    headers = {'User-Agent': user_agent}

    wiki_wiki = wikipediaapi.Wikipedia('en')
    
    # Extract the page title from the URL
    page_title = url.split("/")[-1]  # Get the last part of the URL, which is the title
    
    # Fetch the page content
    page = wiki_wiki.page(page_title)
    
    if not page.exists():
        st.error("The article could not be found.")
        return []
    
    # Now split the content
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Use a larger chunk size for better context in answers
        chunk_overlap=200,
    )

    # Page content as a document for splitting
    docs = [{"page_content": page.text, "metadata": {"source": url, "title": page_title}}]
    
    # Split the document into smaller chunks
    data = text_splitter.split_documents(docs)
    
    return data

# Store embeddings in ChromaDB
def create_vector_store(data):
    embeddings = OpenAIEmbeddings()
    store = Chroma.from_documents(
        data,
        embeddings,
        ids=[f"{item.metadata['source']}-{index}" for index, item in enumerate(data)],
        collection_name="Wimbledon-Embeddings",
        persist_directory='db',
    )
    store.persist()
    return store

# Set up the question-answering chain
def create_qa_chain(store):
    template = """You are a bot that answers questions about Wimbledon 2023, using only the context provided.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain

# Main interaction loop
if st.button("Load Article"):
    # Load and process data
    with st.spinner("Loading data..."):
        data = load_data_from_wikipedia(url)
        
        # Debugging step: Check the retrieved data before creating vector store
        st.write("### Article Content Preview (First 1000 characters):")
        st.write(data[0].page_content[:1000])  # Show a preview of the article content
        
        # Proceed to store embeddings and save vector store in session state
        st.session_state['vector_store'] = create_vector_store(data)
        st.success("Article loaded successfully!")

# Question input
question = st.text_input("Ask a question about the Wimbledon 2023:")

# Get and display the answer
if st.button("Get Answer"):
    if st.session_state['vector_store'] is None:
        st.warning("Please load the article first by clicking 'Load Article'.")
    elif question:
        with st.spinner("Getting answer..."):
            qa_chain = create_qa_chain(st.session_state['vector_store'])
            answer = qa_chain(question)
            st.write("### Answer:")
            st.write(answer['result'])
            st.write("### Source Document:")
            st.write(answer['source_documents'])
    else:
        st.warning("Please enter a question.")

# Footer for additional information
st.write("This app allows you to ask questions about the 2023 Wimbledon Championships using Wikipedia data.")
