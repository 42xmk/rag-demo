# -*- coding: utf-8 -*-
"""
诊断脚本：判断评测集的失败，到底是「切块把证据丢了」还是「召回到了但排不上」。

原理：
  recall@20 低 = 正确答案所在的块没进前 20 名。
  但这有两种完全不同的原因：
    A. 块里压根就没有这段文字  → 切块问题（答案被切断 / 跨块 / 被丢弃）
    B. 块里有这段文字，只是没排上来 → 检索问题（向量不接近 / 池子太小）

  本脚本只做「加载 + 切块」，不调 API、不花钱，几秒钟跑完。

用法：
  python diagnose.py
"""
import json
import re

import main as rag


def norm(s: str) -> str:
    """去掉所有空白再比对，和 eval.py 的 find_hits 保持一致"""
    return re.sub(r"\s+", "", s)


def main():
    print("正在加载文档并切块（不调 API）……\n")
    pages = rag.load_document_pages(rag.DOC_PATH)
    chunks = rag.chunk_pages(pages)

    texts = [(c.get("source", ""), c.get("locator", ""), norm(c["text"])) for c in chunks]
    print(f"当前切块设置：CHUNK_SIZE={rag.CHUNK_SIZE}, OVERLAP={rag.OVERLAP}, "
          f"MIN_CHUNK={rag.MIN_CHUNK}")
    print(f"共切成 {len(chunks)} 块\n")

    data = json.load(open("eval_set.json", encoding="utf-8"))

    exist, missing = [], []
    for item in data:
        ev = norm(item["evidence"])
        hit = [(s, loc) for s, loc, t in texts if ev in t]
        if hit:
            s, loc = hit[0]
            exist.append(item["id"])
            print(f"[{item['id']:02d}] ✅ 证据在语料中（{len(hit)} 块含它）  "
                  f"{s} | {loc}")
        else:
            missing.append(item)
            print(f"[{item['id']:02d}] ❌ 证据不在任何块里  {item['question'][:30]}")
            print(f"      证据片段：{item['evidence'][:60]}")

    n = len(data)
    print("\n" + "=" * 64)
    print(f"证据存在于语料块中：{len(exist)}/{n}")
    print(f"证据完全丢失（切块问题）：{len(missing)}/{n}")
    print("=" * 64)

    if missing:
        print("\n>>> 这 {} 条是「切块/语料」问题，再怎么调召回和重排都救不回来：".format(len(missing)))
        for it in missing:
            print(f"    id={it['id']}  问题：{it['question'][:40]}")
        print("\n>>> 对策：看这些证据原本在文档哪个位置，检查是不是被标题切分切开、"
              "或段落太长被硬切切断了。")
    else:
        print("\n>>> 所有证据都在块里，说明失败出在「检索」环节，"
              "下一步查：加大 RECALL_K、换 embedding、或加 BM25 关键词混合。")


if __name__ == "__main__":
    main()
