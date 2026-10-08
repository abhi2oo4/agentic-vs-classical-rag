import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("BAAI/bge-m3")

def build_index(corpus):
    # shape for the embedding matrix as they only accept numpy arrays
    array_embed=np.vstack(corpus["embeddings"]).astype("float32")

    # building the DB by adding to the index object which stores the data and metadata
    dim=array_embed.shape[1]
    index=faiss.IndexFlatIP(dim)
    index.add(array_embed)
    return index
def save_index(index,path) : 
    faiss.write_index(index,path)
    return
def load_index(path) : 
    index=faiss.read_index(path)
    return index
def search_index(index,query,corpus,k) : 
    encoded=model.encode([query],precision='float32')
    scores,indices=index.search(encoded,k)
    results=corpus.iloc[indices[0]].copy()
    results['score']=scores[0]
    return results


