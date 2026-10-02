import pandas as pd
from collections import defaultdict



def build_corpus(df):
    global_corpus=defaultdict(str)
    for _, row in df.iterrows() : 
        corpus=defaultdict(str)
        for title, content in zip(row['context']['title'], row['context']['sentences']):
            final_st=" ".join(content)
            corpus[title]=final_st
        global_corpus.update(corpus)
    return dict(global_corpus)

    