# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from openai import OpenAI
import main as rag

rag.client = OpenAI(base_url=rag.BASE_URL, api_key=rag.API_KEY)
_, chunks, index = rag.warmup()

for q in ["LangGraph 里短期记忆是怎么存的？", "怎么让查询参数变成必填？"]:
    hits = rag.retrieve(q, chunks, index)
    ans = rag.generate(q, hits)
    print("\n" + "=" * 60)
    print("问：" + q)
    print(ans.strip()[:700])
    print("-" * 60)
    for i, h in enumerate(hits, 1):
        print(f"[引用{i} {rag.cite(h['doc'])} 重排分 {h['score']:.3f}]")
