import streamlit as st
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from google import genai
import os
# Set page config
st.set_page_config(
    page_title="Ready Tensor Publications Explorer",
    page_icon="📚",
    layout="wide"
)

# Initialize Gemini
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize sentence transformer model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_publication' not in st.session_state:
    st.session_state.selected_publication = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = None

# Load publications data
@st.cache_data
def load_publications():
    with open('project_1_publications.json', 'r') as f:
        return json.load(f)

# Load vector store for a specific publication
def load_vector_store(publication_id):
    persist_directory = Path(f"vector_stores/{publication_id}")
    
    # Load the FAISS index
    index = faiss.read_index(str(persist_directory / "index.faiss"))
    
    # Load the documents
    with open(persist_directory / "documents.pkl", "rb") as f:
        documents = pickle.load(f)
    
    return index, documents

def get_relevant_context(question, vector_store, k=3):
    index, documents = vector_store
    
    # Get embedding for the question
    model = get_embedding_model()
    question_embedding = model.encode(question)
    
    # Query the vector store
    distances, indices = index.search(question_embedding.reshape(1, -1).astype('float32'), k)
    
    # Get the relevant documents
    relevant_docs = [documents[i] for i in indices[0]]
    return "\n\n".join(relevant_docs)

def get_answer(question, context):
    # Create a prompt for Gemini
    prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    # Get response from Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

# Main app
st.title("📚 Ready Tensor Publications Explorer")

# Load publications
publications = load_publications()

# Create two columns for the layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select a Publication")
    # Create a selectbox with publication titles
    publication_titles = [pub['title'] for pub in publications]
    selected_title = st.selectbox(
        "Choose a publication to explore:",
        publication_titles,
        index=None,
        placeholder="Select a publication..."
    )
    
    if selected_title:
        # Find the selected publication
        selected_pub = next(pub for pub in publications if pub['title'] == selected_title)
        st.session_state.selected_publication = selected_pub
        
        # Display publication details
        st.markdown("### Publication Details")
        st.markdown(f"**Title:** {selected_pub['title']}")
        st.markdown(f"**Author:** {selected_pub['username']}")
        st.markdown(f"**License:** {selected_pub['license']}")
        
        # Load vector store
        try:
            st.session_state.vector_store = load_vector_store(selected_pub['id'])
            st.success("Publication loaded successfully! You can now ask questions about it.")
        except Exception as e:
            st.error(f"Error loading publication: {str(e)}")

with col2:
    st.subheader("Ask Questions")
    
    if st.session_state.vector_store is None:
        st.info("Please select a publication from the left to start asking questions.")
    else:
        # Chat interface
        user_question = st.text_input("Ask a question about the publication:")
        
        if user_question:
            with st.spinner("Thinking..."):
                # Get relevant context from vector store
                context = get_relevant_context(user_question, st.session_state.vector_store)
                
                # Get answer using Gemini
                answer = get_answer(user_question, context)
                
                # Display chat history
                for message in st.session_state.chat_history:
                    st.markdown(f"**You:** {message['question']}")
                    st.markdown(f"**Assistant:** {message['answer']}")
                    st.markdown("---")
                
                # Display current response
                st.markdown(f"**You:** {user_question}")
                st.markdown(f"**Assistant:** {answer}")
                
                # Update chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer
                }) 