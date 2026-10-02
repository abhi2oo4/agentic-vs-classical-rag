from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


model=SentenceTransformer('BAAI/bge-m3')
def get_embeddings(df):
    """
    Get embeddings for a list of texts using the BGE-M3 model.
    
    Args:
        texts (pandas.dataframe): A dataframe with a 'text' column containing the strings to be embedded.

    Returns:
        a pandas dataframe with the new column attached to it containing the embeddings for each text.
    """
    texts=list(df['text'])
    embeddings=model.encode(texts,batch_size=32,show_progress_bar=True)
    df["embeddings"]=list(embeddings)
    return df