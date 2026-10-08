from typing import TypedDict
from vector_store import *
import pandas as pd
import operator
from typing import Annotated
from openai import AsyncOpenAI
import time
import asyncio
from langgraph.graph import StateGraph, END
import os



client=AsyncOpenAI(base_url=os.environ.get("VLLM_URL", "http://localhost:8000/v1"), api_key="not-needed")
index=load_index('../data/processed/VectorDB.index')
corpus=pd.read_parquet('../data/processed/final_corpus_embed.parquet')
semaphore=asyncio.Semaphore(5)

class RAGState(TypedDict):
    id : str
    question : str
    hop_counter : Annotated[int,operator.add]
    answer : str
    titles : Annotated[list[str],operator.add]
    context : Annotated[list[str],operator.add]
    latency : Annotated[float , operator.add]
    type : str
    num_titles_in_question : int
    prompt_tokens : Annotated[int,operator.add]
    completion_tokens : Annotated[int,operator.add]
    search : bool

graph=StateGraph(RAGState)

def retrieve_node(state : RAGState) :
    query=state['answer']
    context = search_index(index,query,corpus,5)
    already_seen=set(state['titles'])
    new_rows=context[~context['title'].isin(already_seen)]
    return {"context" : new_rows['text'].tolist(), "titles" : new_rows['title'].tolist()}

async def generate_response(state : RAGState) : 
    async with semaphore :
        context_text="\n".join(state["context"])
        time_start=time.perf_counter()
        hop=state['hop_counter']
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            messages=[
                {"role" : "system", 'content' : "Answer based on the given context. Reply with the shortest answer phrase possible, with no explanation, in the format 'ANSWER : ...'. Only do this if the context actually contains the answer. If the context does not contain the answer, instead reply in the format 'QUERY : ...' with a short, specific search phrase naming the exact entity or fact still needed — never leave it blank, vague, or just punctuation."},
                {"role" : "user", "content" : f"Context :\n{context_text}\n\nQuestion :\n{state['question']}"}
            ],
            max_tokens=32
        )
        time_end=time.perf_counter()
        latency=time_end-time_start
        prompt_tokens=response.usage.prompt_tokens
        completion_tokens=response.usage.completion_tokens
        answer=response.choices[0].message.content
        search=False
        if answer.startswith("QUERY :") :
            answer=answer.split("QUERY :")[1].strip()
            search=True
            if len(answer) < 4 :
                answer=state["question"]
        elif answer.startswith("ANSWER :") :
            answer=answer.split("ANSWER :")[1].strip()
            search=False
        return {"answer" : answer, "latency" : latency, "prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens, "hop_counter" : 1, "search" : search}

def should_continue(state : RAGState) : 
    if not state['search'] : 
        return "end"
    if state['hop_counter']>=3 : 
        return "force_answer"
    return "retrieve"

async def force_answer(state :RAGState) :
    context_text="\n".join(state["context"])
    async with semaphore :
        time_start=time.perf_counter()
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            messages=[
                {"role" : "system", 'content' : "Answer based on the given context. Reply with the shortest answer phrase possible, with no explanation in the format 'ANSWER : ...'. Always give your best short answer from the context, even if you are not certain. Never reply with 'I don't know' or 'unknown'."},
                {"role" : "user", "content" : f"Context :\n{context_text}\n\nQuestion :\n{state['question']}"}
            ],
            max_tokens=32
        )
        time_end=time.perf_counter()
    latency=time_end-time_start
    prompt_tokens=response.usage.prompt_tokens
    completion_tokens=response.usage.completion_tokens
    answer=response.choices[0].message.content
    if answer.startswith("ANSWER :") :
        answer=answer.split("ANSWER :")[1].strip()
    return {"answer" : answer, "latency" : latency, "prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens}


graph=StateGraph(RAGState)
graph.add_node("step1",retrieve_node)
graph.add_node("step2",generate_response)
graph.add_node("force_answer",force_answer)

graph.add_edge("step1","step2")
graph.add_edge("force_answer",END)
graph.set_entry_point("step1")
graph.add_conditional_edges("step2",
                            should_continue,
                            { "retrieve" : "step1",
                             "force_answer" : "force_answer",
                             "end" : END,
                             })

app=graph.compile()

def initial_state(row) :
    return{
        'id' : row['id'],
        'question' : row['question'],
        'hop_counter' : 0,              
        'answer' : row['question'],
        'titles' : [],
        'context' : [],     
        'latency' : 0.0,
        'type' : row['type'],
        'num_titles_in_question' : row['num_titles_in_question'],
        'prompt_tokens' : 0,
        'completion_tokens' : 0,
        'search' : True
    }

async def agentic_rag(df) : 
    tasks=[app.ainvoke(initial_state(row)) for _, row in df.iterrows()]
    res = await asyncio.gather(*tasks,return_exceptions=True)
    return res