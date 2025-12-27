import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize

# =======================
# INIT
# =======================
load_dotenv()
app = Flask(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =======================
# UTILS
# =======================
def semantic_similarity(a, b):
    va = embedder.encode([a])
    vb = embedder.encode([b])
    return cosine_similarity(va, vb)[0][0] * 100

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def lcs_percentage(a, b):
    A, B = tokenize(a), tokenize(b)
    m, n = len(A), len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if A[i] == B[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return (dp[m][n] / max(len(A), 1)) * 100

def classify(sem, lex, sem_t, lex_t):
    if sem >= sem_t and lex <= lex_t:
        return "Valid"
    if sem >= sem_t and lex > lex_t:
        return "Plagiarized"
    if sem < sem_t and lex <= lex_t:
        return "Copied"
    return "Invalid"

# =======================
# PROMPT BUILDER
# =======================
def build_prompt(passage, readability, sem_t, lex_t):
    return f"""
Rewrite the following passage.

Rules:
- Output ONLY the rewritten passage.
- Do NOT include explanations, notes, headings, or comments.
- Do NOT mention similarity scores or evaluation.
- Preserve the original meaning.

Guidance:
- Aim for semantic similarity around {sem_t}%.
- Keep lexical reuse below approximately {lex_t}%.
- Freely rephrase wording and sentence structure.
- Match readability level: {readability}.

Text:
{passage}
""".strip()

# =======================
# LLM CALLS
# =======================
def chatgpt_call(prompt):
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return r.choices[0].message.content.strip()
    except:
        return ""

def openrouter_call(model, prompt):
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return ""

# =======================
# ROUTE
# =======================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    best_index = None

    passage = request.form.get("passage", "")
    readability = request.form.get("readability", "Undergraduate")
    semantic_t = float(request.form.get("semantic", 70))
    lexical_t = float(request.form.get("lexical", 30))

    if request.method == "POST" and passage.strip():
        prompt = build_prompt(passage, readability, semantic_t, lexical_t)

        outputs = [
            ("ChatGPT", chatgpt_call(prompt)),
            ("DeepSeek", openrouter_call("deepseek/deepseek-chat", prompt)),
            ("Gemini", openrouter_call("google/gemini-flash-1.5", prompt)),
            ("GPT-OSS", openrouter_call("openai/gpt-oss-20b", prompt))
        ]

        # Human reference
        results.append({
            "model": "Human",
            "text": passage,
            "semantic": 100.0,
            "lexical": 100.0,
            "decision": "Reference",
            "words": len(word_tokenize(passage)),
            "sentences": len(sent_tokenize(passage)),
            "score": 999
        })

        for model, text in outputs:
            if not text:
                continue

            sem = semantic_similarity(passage, text)
            lex = lcs_percentage(passage, text)
            decision = classify(sem, lex, semantic_t, lexical_t)

            results.append({
                "model": model,
                "text": text,
                "semantic": round(sem, 2),
                "lexical": round(lex, 2),
                "decision": decision,
                "words": len(word_tokenize(text)),
                "sentences": len(sent_tokenize(text)),
                "score": sem - lex
            })

        results[1:] = sorted(results[1:], key=lambda x: x["score"], reverse=True)
        best_index = 1 if len(results) > 1 else None

    return render_template(
        "index.html",
        results=results,
        best_index=best_index,
        semantic=semantic_t,
        lexical=lexical_t,
        readability=readability,
        passage=passage
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
