import json
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def load_publications():
    with open('project_1_publications.json', 'r') as f:
        return json.load(f)

def create_vector_store(publication):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get the publication content
    content = publication['publication_description']
    
    # Split content into chunks (simple splitting by double newlines)
    chunks = [chunk.strip() for chunk in content.split('--DIVIDER--') if chunk.strip()]
    
    # Create embeddings for each chunk
    embeddings = model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    
    # Add vectors to the index
    index.add(embeddings.astype('float32'))
    
    # Create directory for this publication's vector store
    vector_store_dir = Path(f"vector_stores/{publication['id']}")
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the index
    faiss.write_index(index, str(vector_store_dir / "index.faiss"))
    
    # Save the documents
    with open(vector_store_dir / "documents.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Created vector store for publication: {publication['title']}")

def main():
    # Load publications
    publications = load_publications()
    
    # Create vector store for each publication
    for publication in publications:
        try:
            create_vector_store(publication)
        except Exception as e:
            print(f"Error creating vector store for {publication['title']}: {str(e)}")

if __name__ == "__main__":
    main() 