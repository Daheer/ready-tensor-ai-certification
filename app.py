import streamlit as st
import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai 

# --- Config ---
INDEX_FILE = "publications_faiss.index"
MAPPING_FILE = "publications_mapping.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# --- Gemini API Setup ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or st.secrets.get('GOOGLE_API_KEY', None)
if not GOOGLE_API_KEY:
    st.error("Please set your Gemini API key in the environment variable 'GOOGLE_API_KEY' or Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# --- Load FAISS index and mapping ---
def load_vector_store():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE)):
        return None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(MAPPING_FILE, "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, model

index, metadata, embed_model = load_vector_store()

# --- Streamlit UI ---
st.set_page_config(page_title="Ready Tensor Publications Assistant", page_icon="🤖")
st.title("Ready Tensor Publications Assistant 🤖 (RAG Mode)")
st.markdown("Ask questions about Ready Tensor publications! (e.g., What's this publication about? What models or tools were used? Any limitations or assumptions?)")

if index is None or metadata is None or embed_model is None:
    st.warning("Vector store not found. Please run build_vector_store.py first.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Helper: Retrieve relevant publications ---
def retrieve_relevant_pubs(query, top_k=TOP_K):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# --- Helper: Build Gemini context from retrieved pubs ---
def build_context_from_pubs(pubs, query):
    context = "You are a helpful assistant for Ready Tensor publications. Answer the user's question using only the information in the following publications.\n\n"
    for pub in pubs:
        context += f"Title: {pub.get('title', 'N/A')}\n"
        context += f"Author: {pub.get('username', 'N/A')}\n"
        context += f"License: {pub.get('license', 'N/A')}\n"
        context += "---\n"
    context += f"\nUser question: {query}\n"
    context += "If you don't know the answer from the sample, say you don't know."
    return context, pubs

# User input
if prompt := st.chat_input("Ask a question about the publications..."):
    # Clear previous messages
    st.session_state["messages"] = []
    
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG: Retrieve top-k relevant publications
    pubs = retrieve_relevant_pubs(prompt, TOP_K)
    if not pubs:
        answer = "Sorry, I couldn't find any relevant publications."
        references = []
    else:
        context, references = build_context_from_pubs(pubs, prompt)
        # Gemini API call
        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=context,
                    )
                    answer = response.text.strip()
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)
            
            # Display references
            if references:
                st.markdown("---")
                st.markdown("**References:**")
                for ref in references:
                    st.markdown(f"- **{ref.get('title', 'N/A')}** by {ref.get('username', 'N/A')}")
    
    # Add assistant message
    st.session_state["messages"].append({
        "role": "assistant", 
        "content": answer,
        "references": references
    }) 