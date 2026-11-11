# -*- coding: utf-8 -*-
"""
RAG 效果评测脚本（面向技术文档版）

指标：
  recall@5   仅向量召回 Top-5  vs  召回 Top-20 + 重排 Top-5（看重排收益）
  recall@20  仅向量召回 Top-20（判断瓶颈在「召回」还是「排序」）
  来源命中   命中块是否来自标注的源文档（防止"答对了但不是这篇文档"）
  重排耗时   单独计时，只计 rerank 这一步（ms）

评测集格式 eval_set.json：
  [{"id": 1, "question": "中文问题", "evidence": "答案在原文中的连续片段", "source": "源文件名"}]

运行：
  python eval.py
需要和 main.py 放同一目录，且 .env 已配好。
"""
import json
import re
import sys
import time

from openai import OpenAI

import main as rag


def norm_text(s):
    """比对前去掉所有空白：中文/英文换行、空格差异都不影响命中判定"""
    return re.sub(r"\s+", "", s)


def find_hits(hits, evidence, source=None):
    """返回命中的块（证据片段出现在块里；给了 source 还要求来自该文档）"""
    ev = norm_text(evidence)
    out = []
    for h in hits:
        d = h["doc"]
        if ev in norm_text(d["text"]):
            if source and d.get("source") != source:
                continue
            out.append(d)
    return out


def main():
    if not rag.API_KEY or rag.API_KEY.startswith("在这里填"):
        print("请先配置 .env 中的 API_KEY")
        sys.exit(1)
    rag.client = OpenAI(base_url=rag.BASE_URL, api_key=rag.API_KEY)

    data = json.load(open("eval_set.json", encoding="utf-8"))
    _, chunks, index = rag.warmup()

    base5 = rerank5 = recall20 = src5 = 0
    saved = hurt = 0
    lat = []
    rows = []

    print(f"\n开始评测，共 {len(data)} 条（重排后端：{rag.RERANK_BACKEND}）\n")
    for item in data:
        q = item["question"]

        # 1) 向量召回 Top-20（粗排）
        qv = rag.embed_texts([q])[0]
        cand = rag.search(index, qv, rag.RECALL_K)
        base20 = [{"doc": chunks[i], "score": s} for i, s in cand]

        # 2) 只对 Top-20 做 rerank，单独计时
        docs = [chunks[i] for i, _ in cand]
        t0 = time.perf_counter()
        rr = rag.rerank(q, docs, rag.TOP_N)
        lat.append((time.perf_counter() - t0) * 1000)

        h20 = bool(find_hits(base20, item["evidence"]))
        hb = bool(find_hits(base20[:5], item["evidence"]))
        hr = bool(find_hits(rr, item["evidence"]))
        # 命中块是否真的来自标注文档（更严格的溯源检查）
        hs = bool(item.get("source")) and bool(find_hits(rr, item["evidence"], item["source"]))

        recall20 += h20
        base5 += hb
        rerank5 += hr
        src5 += hs
        if hr and not hb:
            saved += 1
        if hb and not hr:
            hurt += 1

        got = find_hits(rr, item["evidence"])
        rows.append({
            "id": item["id"], "question": q, "source": item.get("source", ""),
            "hit_top5": hb, "hit_rerank5": hr, "hit_top20": h20, "hit_source": hs,
            "rerank_top1": (rag.cite(rr[0]["doc"]) if rr else ""),
            "hit_locator": (rag.cite(got[0]) if got else ""),
        })
        print(f"[{item['id']:02d}] {q[:20]:<20} 召回@5:{'Y' if hb else 'N'}  "
              f"重排@5:{'Y' if hr else 'N'} 召回@20:{'Y' if h20 else 'N'}  来源:{'Y' if hs else 'N'}")
        if hr and got:
            print(f"      命中位置 -> {rag.cite(got[0])}")

    n = len(data)
    print("\n" + "=" * 64)
    print(f"recall@5   仅向量召回：{base5}/{n} = {base5 / n:.0%}")
    print(f"recall@5   召回+重排：{rerank5}/{n} = {rerank5 / n:.0%}")
    print(f"recall@20  仅向量召回：{recall20}/{n} = {recall20 / n:.0%}   <-- 瓶颈在召回还是排序")
    print(f"来源正确率（命中块来自标注文档）：{src5}/{n} = {src5 / n:.0%}")
    print(f"重排：救回 {saved} 条，误伤 {hurt} 条")
    print(f"rerank 单次耗时（只计重排这一步）：平均 {sum(lat) / len(lat):.0f} ms"
          f"（min {min(lat):.0f} / max {max(lat):.0f}）")
    print("=" * 64)

    with open("eval_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "backend": rag.RERANK_BACKEND,
            "n": n,
            "chunks": len(chunks),
            "recall_at5_baseline": base5 / n,
            "recall_at5_rerank": rerank5 / n,
            "recall_at20_baseline": recall20 / n,
            "source_accuracy": src5 / n,
            "saved_by_rerank": saved,
            "hurt_by_rerank": hurt,
            "rerank_latency_ms_avg": sum(lat) / len(lat),
            "rerank_latency_ms_max": max(lat),
            "details": rows,
        }, f, ensure_ascii=False, indent=2)
    print("明细已写入 eval_result.json")


if __name__ == "__main__":
    main()
