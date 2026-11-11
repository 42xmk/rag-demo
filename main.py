# -*- coding: utf-8 -*-
"""
面向技术文档的 RAG 问答系统

相对基础版，这一版解决了三件事，也正好是简历里能讲的三点：
  1) 结构化切分：按 Markdown 标题层级切块，每个块带「章节路径」，不再是无脑 400 字符
  2) 两段式检索：FAISS 向量召回 Top-20 → bge-reranker-v2-m3 精排 Top-5
  3) 可溯源引用：回答标注 [资料N | 文档名｜章节]，而不是只在单篇 PDF 里标页码

完整链路：
  多格式加载(PDF/MD/MDX/TXT/DOCX) → 按标题切分(带章节) → bge-m3 向量化
  → FAISS 召回 Top-20 → bge-reranker 精排 Top-5 → 拼 Prompt(带章节) → DeepSeek 生成

运行：
  pip install -r requirements.txt
  python main.py          # 交互问答
  python eval.py          # 20 条评测集，recall@5（重排前后）+ recall@20
"""
import os
import re
import sys
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Windows 控制台默认 GBK，遇到编码不了的字符会直接崩，这里降级成 ?
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ---------- 配置（模型型号写死，方便写简历/面试） ----------
BASE_URL = os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-ai/DeepSeek-V3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "api")   # api / local / off
DOC_PATH = os.getenv("DOC_PATH", "")

CHUNK_SIZE = 400     # 每块目标字符数（实验对比 400 vs 200 后定回 400：效果同为 80%，块少一半、rerank 便宜一半）
OVERLAP = 60         # 超长段落硬切时的重叠字符数
MIN_CHUNK = 40       # 小于这个长度的块（纯标题行等）丢弃
RECALL_K = 20        # 向量召回条数（粗排）
TOP_N = 5            # 精排后进入 Prompt 的条数

client = None
_reranker = None


# ---------- 1. 文档加载（保留来源 / 类型） ----------
SUPPORTED_EXTS = (".pdf", ".txt", ".md", ".mdx", ".docx")


def load_pdf_pages(path: str) -> list:
    """逐页提取文本，返回 [{"page": 页码, "text": 文本}]，跳过空白页"""
    import fitz  # PyMuPDF，延迟导入：只跑 Markdown 时不必装
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def _read_docx(path: str) -> str:
    try:
        from docx import Document
    except Exception:
        return ""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _clean_markdown(text: str) -> str:
    """清理 markdown/mdx：去 frontmatter、import/export、JSX 标签、图片、链接语法"""
    lines, out, in_fm = text.splitlines(), [], False
    for i, ln in enumerate(lines):
        if i == 0 and ln.strip() == "---":
            in_fm = True
            continue
        if in_fm:
            if ln.strip() == "---":
                in_fm = False
            continue
        s = ln.strip()
        if s.startswith("import ") or s.startswith("export "):
            continue
        out.append(ln)
    text = "\n".join(out)
    text = re.sub(r"<[^>\n]{1,200}>", "", text)           # JSX / HTML 标签
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)       # 图片
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # 链接只留文字
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _doc_title(name: str) -> str:
    """文件名 -> 可读的文档名，引用里显示它比显示 fastapi_tutorial_xxx.md 好看"""
    stem = os.path.splitext(name)[0]
    if stem.startswith("fastapi_tutorial_"):
        return "fastapi/" + stem[len("fastapi_tutorial_"):]
    if stem.startswith("langchain_"):
        return "langchain/" + stem[len("langchain_"):]
    return stem


def load_document_pages(path: str) -> list:
    """path 可传单个文件或文件夹（递归读 pdf/txt/md/mdx/docx）

    返回 [{"source": 文件名, "title": 可读文档名, "kind": 类型, "page": 页码, "text": 文本}]
    """
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for n in sorted(names):
                if n.lower().endswith(SUPPORTED_EXTS):
                    files.append(os.path.join(root, n))
    else:
        files = [path]

    pages = []
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        name = os.path.basename(fp)
        title = _doc_title(name)
        kind = ext.lstrip(".")
        if ext == ".pdf":
            for p in load_pdf_pages(fp):
                pages.append({"source": name, "title": title, "kind": kind,
                              "page": p["page"], "text": p["text"]})
        elif ext == ".docx":
            text = _read_docx(fp)
            if text.strip():
                pages.append({"source": name, "title": title, "kind": kind,
                              "page": 1, "text": text})
        else:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = _clean_markdown(f.read())
            if text.strip():
                pages.append({"source": name, "title": title, "kind": kind,
                              "page": 1, "text": text})
    return pages


# ---------- 2. 结构化切分：按 Markdown 标题层级切块 ----------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _clean_heading(title: str) -> str:
    """去掉 MDX 的锚点 { #anchor }、粗体星号、反引号"""
    title = re.sub(r"\s*\{[^}]*\}\s*$", "", title)
    title = title.replace("**", "").replace("*", "").replace("`", "")
    return title.strip()


def split_markdown_sections(text: str) -> list:
    """按 # 层级切分，返回 [(章节路径, 正文)]

    章节路径形如 "Dependencies > First Steps > Create a dependency"，
    这样引用能精确到章节，而不是"第 1 页"。
    """
    stack = []      # [(level, title)] 当前标题栈
    sections = []   # [(path_list, lines)]
    cur_path, cur_lines = ["文档开头"], []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), _clean_heading(m.group(2))
            if not title:
                continue
            if "".join(cur_lines).strip():
                sections.append((list(cur_path), cur_lines))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path, cur_lines = [t for _, t in stack], []
        else:
            cur_lines.append(line)
    if "".join(cur_lines).strip():
        sections.append((list(cur_path), cur_lines))
    return [(" > ".join(p), "\n".join(ls).strip()) for p, ls in sections]


def _pack_paragraphs(paras: list, chunk_size: int, overlap: int) -> list:
    """把段落贪心装箱成 <= chunk_size 的块；块不跨段落，引用更准
    单段本身超长时，再按 chunk_size 硬切并保留 overlap 重叠
    """
    chunks, buf = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) > chunk_size:
            if buf.strip():
                chunks.append(buf.strip())
                buf = ""
            start = 0
            while start < len(p):
                chunks.append(p[start:start + chunk_size].strip())
                start += chunk_size - overlap
            continue
        if buf and len(buf) + len(p) + 2 > chunk_size:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_pages(pages: list, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """按章节切块；每块带 source / title / locator（章节路径或页码）
    块结构：{"id", "source", "title", "page", "locator", "text"}
    """
    chunks = []
    for p in pages:
        src, title, kind = p.get("source", ""), p.get("title", ""), p.get("kind", "")
        if kind in ("md", "mdx"):
            secs = split_markdown_sections(p["text"])
        elif kind == "txt":
            secs = [("全文", p["text"])]
        else:                     # pdf / docx 没有标题结构，退回按页
            secs = [(f"第{p.get('page', 1)}页", p["text"])]
        for locator, body in secs:
            paras = [x for x in re.split(r"\n{2,}", body) if x.strip()]
            for piece in _pack_paragraphs(paras, chunk_size, overlap):
                if len(piece) < MIN_CHUNK and len(paras) > 1:
                    continue
                chunks.append({
                    "id": len(chunks),
                    "source": src,
                    "title": title,
                    "page": p.get("page", 1),
                    "locator": locator,
                    "text": piece,
                })
    return chunks


# ---------- 3. 向量化 ----------
def embed_texts(texts: list, batch_size: int = 32) -> np.ndarray:
    """调用 Embedding 模型，把文字变成向量；分批防止一次请求过大"""
    vecs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vecs.extend(d.embedding for d in resp.data)
    return np.array(vecs, dtype="float32")


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 归一化：归一化后内积 == 余弦相似度"""
    norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norm


# ---------- 4. 向量索引（FAISS，装不上则退回 numpy） ----------
def build_index(vecs: np.ndarray):
    try:
        import faiss
        index = faiss.IndexFlatIP(vecs.shape[1])   # IP = Inner Product = 余弦（已归一化）
        index.add(vecs)
        return ("faiss", index)
    except Exception:
        return ("numpy", vecs)


def search(index, query_vec: np.ndarray, k: int) -> list:
    """返回 [(chunk下标, 相似度)]，按相似度降序"""
    kind, obj = index
    if kind == "faiss":
        D, I = obj.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]
    sims = obj @ query_vec
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


# ---------- 5. Rerank 重排 ----------
def rerank_api(query: str, docs: list, top_n: int = TOP_N) -> list:
    """调硅基流动 /rerank 接口（BAAI/bge-reranker-v2-m3），不用装 torch"""
    import requests
    resp = requests.post(
        BASE_URL.rstrip("/") + "/rerank",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": [d["text"] for d in docs],
            "top_n": top_n,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"doc": docs[item["index"]], "score": float(item.get("relevance_score", 0))}
        for item in data.get("results", [])
    ]


def rerank_local(query: str, docs: list, top_n: int = TOP_N) -> list:
    """本地 CrossEncoder（需 pip install sentence-transformers，会下载 torch + 模型）"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    pairs = [(query, d["text"]) for d in docs]
    scores = np.array(_reranker.predict(pairs))
    order = np.argsort(-scores)[:top_n]
    return [{"doc": docs[i], "score": float(scores[i])} for i in order]


def rerank(query: str, docs: list, top_n: int = TOP_N) -> list:
    if RERANK_BACKEND == "off" or not docs:
        return [{"doc": d, "score": 0.0} for d in docs[:top_n]]
    if RERANK_BACKEND == "local":
        return rerank_local(query, docs, top_n)
    return rerank_api(query, docs, top_n)


# ---------- 6. 检索：召回 + 重排 ----------
def retrieve(question: str, chunks: list, index, rerank_on: bool = True,
             recall_k: int = RECALL_K, top_n: int = TOP_N) -> list:
    qv = embed_texts([question])[0]
    cand = search(index, qv, recall_k)
    docs = [chunks[i] for i, _ in cand]
    if not rerank_on:
        return [{"doc": d, "score": s} for d, (i, s) in zip(docs[:top_n], cand[:top_n])]
    return rerank(question, docs, top_n)


# ---------- 7. Prompt 构造 ----------
def cite(doc: dict) -> str:
    """统一的引用标签：文档名｜章节"""
    return f"{doc.get('title') or doc.get('source', '')}｜{doc.get('locator') or ('第' + str(doc.get('page', 1)) + '页')}"


def build_prompt(question: str, hits: list) -> str:
    """把检索到的资料块（带章节）拼进 Prompt，要求只依据资料回答"""
    context = "\n\n".join(
        f"[资料{i + 1} | {cite(h['doc'])}]\n{h['doc']['text']}"
        for i, h in enumerate(hits)
    )
    return (
        "你是技术文档助手，只能根据下面给出的资料回答问题。\n"
        "规则：\n"
        "1. 只依据资料回答，不要引入资料以外的知识；\n"
        "2. 资料不足以回答时，直接回答“资料中没有相关内容”；\n"
        "3. 回答末尾用 [资料N | 文档名｜章节] 的形式标注依据，可以标多条。\n\n"
        f"{context}\n\n"
        f"问题：{question}\n回答："
    )


# ---------- 8. 生成 ----------
def generate(question: str, hits: list) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def warmup(doc_path: str = None):
    """加载文档 → 按章节切块 → 向量化 → 建索引，返回 (pages, chunks, index)"""
    doc_path = doc_path or DOC_PATH
    print(f"正在加载文档：{doc_path}")
    pages = load_document_pages(doc_path)
    chunks = chunk_pages(pages)
    sources = sorted({c["source"] for c in chunks})
    print(f"共 {len(sources)} 个文件 / {len(pages)} 个文档单元 / 切成 {len(chunks)} 个块，开始向量化……")
    for s in sources:
        sub = [c for c in chunks if c["source"] == s]
        print(f"    - {s}: {len(sub)} 块 / {len({c['locator'] for c in sub})} 个章节")
    vecs = normalize(embed_texts([c["text"] for c in chunks]))
    index = build_index(vecs)
    kind = "FAISS" if index[0] == "faiss" else "numpy（未装 faiss）"
    print(f"向量化完成，索引：{kind}，可以开始提问（输入 exit 退出）。\n")
    return pages, chunks, index


def main():
    global client
    if not API_KEY or API_KEY.startswith("在这里填"):
        print("请先复制 .env.example 为 .env，并填入你的 API_KEY，再运行本脚本。")
        return
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    _, chunks, index = warmup()

    while True:
        q = input("你的问题：").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        hits = retrieve(q, chunks, index, rerank_on=(RERANK_BACKEND != "off"))
        answer = generate(q, hits)
        print("\n" + "=" * 50)
        print(answer)
        print("-" * 50)
        for i, h in enumerate(hits, 1):
            snippet = h["doc"]["text"][:60].replace("\n", " ")
            print(f"[引用{i} {cite(h['doc'])} 重排分 {h['score']:.3f}] {snippet}…")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
