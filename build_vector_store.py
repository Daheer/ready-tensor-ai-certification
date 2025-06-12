import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Config
PUBLICATIONS_FILE = "project_1_publications.json"
INDEX_FILE = "publications_faiss.index"
MAPPING_FILE = "publications_mapping.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SAMPLE_SIZE = 50  # Use up to 50 publications for the vector store

# Load publications
with open(PUBLICATIONS_FILE, "r") as f:
    publications = json.load(f)

# Optionally sample for diversity
if len(publications) > SAMPLE_SIZE:
    import random
    publications = random.sample(publications, SAMPLE_SIZE)

# Prepare texts and metadata
texts = []
metadata = []
for pub in publications:
    # Use title + description for embedding
    text = f"Title: {pub.get('title', '')}\nDescription: {pub.get('publication_description', '')}"
    texts.append(text)
    metadata.append({
        "id": pub.get("id"),
        "title": pub.get("title"),
        "username": pub.get("username"),
        "license": pub.get("license"),
    })

# Compute embeddings
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and mapping
faiss.write_index(index, INDEX_FILE)
with open(MAPPING_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"FAISS index and mapping saved. {len(metadata)} publications indexed.") 