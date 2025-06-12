import streamlit as st
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# Set page config
st.set_page_config(
    page_title="Ready Tensor Publications Explorer",
    page_icon="📚",
    layout="wide"
)

# Initialize Gemini
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

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

# Load publications data
@st.cache_data
def load_publications():
    with open('project_1_publications.json', 'r') as f:
        return json.load(f)

# Load vector store for a specific publication
def load_vector_store(publication_id):
    persist_directory = f"vector_stores/{publication_id}"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name="documents")
    return collection

def get_relevant_context(question, vector_store, k=3):
    # Get embedding for the question
    model = get_embedding_model()
    question_embedding = model.encode(question).tolist()
    
    # Query the vector store
    results = vector_store.query(
        query_embeddings=[question_embedding],
        n_results=k
    )
    
    # Extract and join the documents
    return "\n\n".join(results['documents'][0])

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
        st.markdown(f"**Authors:** {', '.join(selected_pub['authors'])}")
        st.markdown(f"**Year:** {selected_pub['year']}")
        
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