# -*- coding: utf-8 -*-
"""离线自检：不调 API，只验证「切分结果」和「评测集证据」是否对得上"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as rag


def norm(s):
    return re.sub(r"\s+", "", s)


pages = rag.load_document_pages("docs")
chunks = rag.chunk_pages(pages)

print(f"文档单元 {len(pages)} / 块 {len(chunks)}")
by_src = {}
for c in chunks:
    by_src.setdefault(c["source"], []).append(c)
for s, cs in by_src.items():
    print(f"  {s:44s} 块{len(cs):4d}  章节{len({c['locator'] for c in cs}):3d}")
print("\n章节示例：")
for c in chunks[:3]:
    print(f"  {rag.cite(c)}  ->  {c['text'][:50]!r}")

data = json.load(open("eval_set.json", encoding="utf-8"))
print(f"\n评测集 {len(data)} 条，开始核对证据 ……")
bad = 0
for item in data:
    ev = norm(item["evidence"])
    hits = [c for c in chunks if ev in norm(c["text"])]
    ok_src = [c for c in hits if c["source"] == item.get("source")]
    if not hits:
        bad += 1
        print(f"[{item['id']:02d}] 缺失 !! {item['source']}")
        # 给出该文档中与证据首词相近的行，便于修正
        kw = item["evidence"].split()[0]
        doc = [c for c in chunks if c["source"] == item.get("source")]
        cand = [ln for c in doc for ln in c["text"].splitlines() if kw in ln][:2]
        for ln in cand:
            print(f"     候选: {ln.strip()[:120]}")
    elif not ok_src:
        bad += 1
        print(f"[{item['id']:02d}] 来源不符 !! 命中在 {hits[0]['source']}")
    else:
        print(f"[{item['id']:02d}] OK  {rag.cite(ok_src[0])}")
print(f"\n结论：{len(data) - bad}/{len(data)} 条可用")
