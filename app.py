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
from datetime import datetime  # Importing datetime to get the current time

# Streamlit app title
st.set_page_config(page_title="WikiAsk", layout="wide")
st.title("🤖 WikiAsk")

# Access the API key from secrets.toml
openai_api_key = st.secrets["OPENAI_API_KEY"]


# Streamlit app title
st.set_page_config(page_title="WikiAsk", layout="wide")
st.title("🤖 WikiAsk")

# Initialize session state for vector store and chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for Wikipedia article
url = st.text_input("Enter a Wikipedia article URL:", 
                    placeholder="Type the Wikipedia URL here...", label_visibility="collapsed")

# Load data from Wikipedia
def load_data_from_wikipedia(url):
    # Extract the article title from the URL
    query = url.split("/")[-1]
    docs = WikipediaLoader(query=query, load_max_docs=1).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Use larger chunks if necessary
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

# Store embeddings in ChromaDB
def create_vector_store(data):
    embeddings = OpenAIEmbeddings()
    try:
        store = Chroma.from_documents(
            data,
            embeddings,
            ids=[f"{item.metadata['source']}-{index}" for index, item in enumerate(data)],
            collection_name="Wikipedia-Embeddings",
            persist_directory='db',
        )
        store.persist()  # Ensure the embeddings are persisted
        # st.write(f"Vector store created with {len(data)} documents.")  # Debugging line
        return store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Set up the question-answering chain
def create_qa_chain(store, article_title):
    if store is None:
        st.error("Vector store is not initialized. Please load the article first.")
        return None

    template = f"""You are a bot that answers questions based on the Wikipedia article about {article_title}. 
    Use only the context provided from the article. Make sure to read every article and use context clues to answer. 
    Please provide comprehensive answers using the context below, including full lists of names or items.

    {{context}}

    Question: {{question}}"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# Main interaction loop
if st.button("Load Article"):
    # Load and process data
    with st.spinner("Loading data..."):
        data = load_data_from_wikipedia(url)
        
        # # Debugging step: Check the retrieved data before creating vector store
        # st.write("### Article Content Preview (First 1000 characters):")
        # st.write(data[0].page_content[:1000])  # Show a preview of the article content
        
        # Proceed to store embeddings and save vector store in session state
        st.session_state['vector_store'] = create_vector_store(data)
        st.success("Article loaded successfully!")

# Question input
question = st.text_input("Ask a question about the article:", 
                         placeholder="Type your question here...")

# Get and display the answer
if st.button("Get Answer"):
    if st.session_state.vector_store is None:
        st.warning("Please load the article first by clicking 'Load Article'.")
    elif question:
        with st.spinner("Getting answer..."):
            # Extract article title from URL to display in the prompt
            article_title = url.split("/")[-1].replace("_", " ")
            
            # Check if qa_chain is already created in session state
            if 'qa_chain' not in st.session_state:
                st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store, article_title)
            
            # Now call the qa_chain with the question
            answer = st.session_state.qa_chain({"query": question})  # Pass the question as a dictionary with the correct key
            # st.write("### Retrieved Documents:")
            # for doc in answer['source_documents']:
            #     st.write(doc.page_content)
            # Store the chat history with a timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "user": question,
                "bot": answer['result']
            })

# Display chat messages in reverse order
if st.session_state.chat_history:
    st.write("### Chat History:")
    for chat in reversed(st.session_state.chat_history):
        # Display user messages in blue and bot messages in green
        st.markdown(f"<div style='background-color: #000000; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>"
                    f"<strong>[{chat['timestamp']}] User:</strong> {chat['user']}</div>", 
                    unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: #000000; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>"
                    f"<strong>[{chat['timestamp']}] Bot:</strong> {chat['bot']}</div>", 
                    unsafe_allow_html=True)

