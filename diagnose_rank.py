# -*- coding: utf-8 -*-
"""
诊断脚本（第二层）：每条评测的「证据块」，在所有块里到底排第几名？

用途：
  diagnose.py 已经证明「证据都在块里」，也就是切块没问题。
  那接下来唯一的问题就是：那个块，向量检索把它排到了第几名？

    - 排名 1~5      → 召回和排序都没问题
    - 排名 6~20     → 召回到了，但没进前 5 → 排序问题
    - 排名 21~50    → RESCALL_K 调大就能救
    - 排名 50~200   → 池子不够大，或者向量匹配很差
    - 排名 >200/没找到 → 向量根本没匹配上 → 要换检索方式（BM25 / query 改写 / 换 embedding）

需要调用 API 做向量化（约 1 分钟），只跑一次。

用法：
  python diagnose_rank.py
"""
import json
import re
import sys

from openai import OpenAI

import main as rag


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main():
    if not rag.API_KEY or rag.API_KEY.startswith("在这里填"):
        print("请先配置 .env 中的 API_KEY")
        sys.exit(1)
    rag.client = OpenAI(base_url=rag.BASE_URL, api_key=rag.API_KEY)

    _, chunks, index = rag.warmup()
    texts = [norm(c["text"]) for c in chunks]
    data = json.load(open("eval_set.json", encoding="utf-8"))

    print(f"\n对 {len(data)} 个问题做全量排序，找出证据块的真实排名……\n")
    rows = []
    for item in data:
        qv = rag.embed_texts([item["question"]])[0]
        ranked = rag.search(index, qv, len(chunks))   # 全量排序

        ev = norm(item["evidence"])
        rank = None
        top5_score = ranked[0][1] if ranked else 0.0
        for pos, (cidx, score) in enumerate(ranked, start=1):
            if ev in texts[cidx]:
                rank = pos
                break

        rows.append((item["id"], rank, top5_score, item["question"][:24]))

        if rank is None:
            print(f"[{item['id']:02d}] ❌ 全量 {len(chunks)} 块里都找不到证据   {item['question'][:30]}")
        elif rank <= 5:
            print(f"[{item['id']:02d}] ✅ 排名 {rank:>4}  {item['question'][:30]}")
        elif rank <= 20:
            print(f"[{item['id']:02d}] ⚠️  排名 {rank:>4}（召回@20 里有，但没进前5）   {item['question'][:30]}")
        elif rank <= 50:
            print(f"[{item['id']:02d}] ⚠️  排名 {rank:>4}（调大 RECALL_K 能救）   {item['question'][:30]}")
        else:
            print(f"[{item['id']:02d}] ❌ 排名 {rank:>4}（向量基本没匹配上）   {item['question'][:30]}")

    found = [r for r in rows if r[1] is not None]
    in5 = sum(1 for r in found if r[1] <= 5)
    in20 = sum(1 for r in found if r[1] <= 20)
    in50 = sum(1 for r in found if r[1] <= 50)
    n = len(data)

    print("\n" + "=" * 68)
    print(f"证据块在全量排序中的位置分布（共 {n} 条）")
    print(f"  排进前 5 ：{in5:>2}/{n}")
    print(f"  排进前 20：{in20:>2}/{n}")
    print(f"  排进前 50：{in50:>2}/{n}")
    r20 = sorted([r[1] for r in found if r[1] > 20]) or [0]
    print(f"  排不进前20的：{len([r for r in found if r[1] > 20])} 条")
    print("=" * 68)

    if in50 > in20:
        print("\n>>> 说明「调大 RECALL_K 就能救」。下一步：把 main.py 第 47 行 RECALL_K 改成 50 再跑 eval.py。")
    elif len(found) < n:
        print("\n>>> 有证据在全量排序里都找不到，说明 embedding 对该问题几乎无效，"
              "要上 BM25 关键词混合检索。")
    else:
        print("\n>>> 证据都已经在前 20 里，问题在排序。下一步：加大 TOP_N，或换更强的 rerank 模型。")

    with open("rank_report.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"id": r[0], "rank": r[1], "question": r[3]} for r in rows],
            f, ensure_ascii=False, indent=2,
        )
    print("明细已写入 rank_report.json")


if __name__ == "__main__":
    main()
