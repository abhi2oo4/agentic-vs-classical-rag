from agentic_rag import agentic_rag
import pandas as pd
import asyncio

pilot =pd.read_parquet("../data/processed/pilot.parquet")
async def main() : 
    res=await agentic_rag(pilot)
    assert len(res)==len(pilot),len(res)
    assert len({r["id"] for r in res})==len(pilot)
    pd.DataFrame(res).to_parquet("../data/processed/agentic_results.parquet")
    print(f"saved {len(res)} records")


asyncio.run(main())