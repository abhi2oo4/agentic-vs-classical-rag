import pandas as pd

from chunking import chunk_corpus
from corpus import build_corpus
from embedding import get_embeddings
from vector_store import build_index, save_index

df = pd.read_parquet("../data/processed/df_val_enriched.parquet")

corpus = build_corpus(df)
print(f"corpus built: {len(corpus)} documents")

pilot = df.groupby(["type", "num_titles_in_question"]).apply(
    lambda g: g.sample(n=min(len(g), 150), random_state=4)
)
pilot = pilot.reset_index(drop=False)
pilot = pilot.drop(columns=["level_2"])
print(pilot.groupby(["type", "num_titles_in_question"]).size())
print(f"pilot built: {len(pilot)} rows")

final_corpus = chunk_corpus(corpus, threshold=300)
print(f"final_corpus built: {final_corpus.shape}")

pilot.to_parquet("../data/processed/pilot.parquet")
final_corpus.to_parquet("../data/processed/final_corpus.parquet")
print("saved pilot.parquet, final_corpus.parquet")

final_corpus_embed = get_embeddings(final_corpus)
final_corpus_embed.to_parquet("../data/processed/final_corpus_embed.parquet")
print("saved final_corpus_embed.parquet")

index = build_index(final_corpus_embed)
save_index(index, "../data/processed/VectorDB.index")
print("saved VectorDB.index")
