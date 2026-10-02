from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import tiktoken

def chunk_text(text, chunk_size=300, chunk_overlap=20):
    """
    splitting the text into chunks of size 300 tokens with an overlap of 20
    
    """
    textsplitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        )

    chunks=textsplitter.split_text(text)
    return chunks


def chunk_corpus(corpus,threshold=300,overlap=20):
    """
    chunking the documents in the corpus if they are greater than threshold
    """

    rows=[]
    for title,text in corpus.items():
        enc=tiktoken.get_encoding("cl100k_base")
        t=enc.encode(text)
        if(len(t)>threshold):
            chunks=chunk_text(text=text,chunk_size=threshold,chunk_overlap=overlap)
            for i, chunk in enumerate(chunks):
                rows.append({"chunk_id":f"{title}_{i}","title":f"{title}","chunk_index" : i,"text" : f"{chunk}"})
        else : 
            rows.append({"chunk_id":f"{title}_0","title":f"{title}","chunk_index" : 0,"text" : f"{text}"})
    df=pd.DataFrame(rows)
    return df
            
    