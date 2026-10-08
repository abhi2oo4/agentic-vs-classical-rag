# agentic-vs-classical-rag

Compares **classical RAG** (single retrieval pass + one generation call) against
**agentic RAG** (a LangGraph-driven retrieve → reason → re-retrieve loop, up to
3 hops) on a stratified 798-question sample from HotpotQA, using a
self-hosted Qwen model served via vLLM. Evaluated on exact match, F1,
retrieval recall, latency, and token cost.

## Running it

Requires Docker with GPU support (NVIDIA Container Toolkit).

```bash
docker build -t agentic-vs-classical .

# 1. Generate the dataset (corpus, embeddings, FAISS index, pilot sample)
#    --no-deps: skip starting vLLM here, it isn't needed and would just
#    compete with BGE-M3 for GPU memory
docker compose run --no-deps app python prepare_data.py

# 2. Start the model server
docker compose up -d vllm

# 3. Run both pipelines
docker compose run app python run_classical.py
docker compose run app python run_agentic.py

# 4. Score and compare
docker compose run app python evaluate.py

docker compose down
```

Results (metrics CSVs + comparison graphs) are saved to `data/results/`.
