import asyncio

import pandas as pd

from classical_rag import classical_rag


pilot = pd.read_parquet("../data/processed/pilot.parquet")
corpus = pd.read_parquet("../data/processed/final_corpus_embed.parquet")


async def main():
    res = await classical_rag(pilot, corpus)
    assert len(res) == len(pilot), len(res)
    assert len({r["id"] for r in res}) == len(pilot)
    pd.DataFrame(res).to_parquet("../data/processed/classical_results.parquet")
    print(f"saved {len(res)} records")


asyncio.run(main())
