from openai import AsyncOpenAI
import asyncio
from vector_store import *
import time
import os

client = AsyncOpenAI(base_url=os.environ.get("VLLM_URL", "http://localhost:8000/v1"), api_key="not-needed")


semaphore=asyncio.Semaphore(5)
index=load_index('../data/processed/VectorDB.index')

async def ask(question,context) :
    async with semaphore:
        time_start=time.perf_counter()
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "Answer the question using only the context. Reply with the shortest answer phrase possible, with no explanation. Always give your best short answer from the context, even if you are not certain."},
                {"role": "user", "content": f"Context :\n{context}\n\nQuestion :\n{question}"},
            ],
            max_tokens=32,
        )
    time_end=time.perf_counter()
    dic={}
    latency=time_end-time_start
    dic["latency"]=latency
    dic['question']=question
    dic['answer']=response.choices[0].message.content
    dic['prompt_tokens']=response.usage.prompt_tokens
    dic['completion_tokens']=response.usage.completion_tokens
    return dic

async def classical_rag(df,corpus):
    
    tasks=[]
    final_list=[]
    for _, row in df.iterrows() :
        final_dict={}
        final_dict['type']=row['type']
        final_dict['num_titles_in_question']=row['num_titles_in_question']
        question=row['question']
        context = search_index(index,question,corpus,5)
        context_text="\n".join(context['text'].tolist())
        retrieved_context=[title for title in context['title'].values]
        final_dict["titles"]=retrieved_context
        final_dict['id']=row['id']
        final_list.append(final_dict)
        tasks.append(ask(question,context_text))

    res=await asyncio.gather(*tasks,return_exceptions=True)

    for i, j in zip(final_list,res) : 
        i.update(j)

    return final_list