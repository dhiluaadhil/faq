"""
app.py — Gradio UI for the FAQ Semantic Search Engine
Deployed on HuggingFace Spaces.

Search logic is identical to the notebook — no changes to the model or data pipeline.
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
import gradio as gr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ── Suppress noisy logs ──────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = "wiki_qa"
CACHE_DIR    = Path("./cache")
EMB_FILE     = CACHE_DIR / "embeddings.npy"
META_FILE    = CACHE_DIR / "metadata.json"

# ── Boot: load model, dataset, embeddings ────────────────────
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

if EMB_FILE.exists() and META_FILE.exists():
    print("Cache found — loading from disk...")
    embeddings = np.load(EMB_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        faqs = json.load(f)
    print(f"Loaded {len(faqs)} FAQs from cache.")
else:
    print("Downloading wiki_qa dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    seen, faqs = set(), []
    for row in ds:
        q, a = row["question"].strip(), row["answer"].strip()
        if not q or not a:
            continue
        if q.lower() not in seen:
            seen.add(q.lower())
            faqs.append({"question": q, "answer": a})

    print(f"Encoding {len(faqs)} FAQs...")
    CACHE_DIR.mkdir(exist_ok=True)
    questions  = [f["question"] for f in faqs]
    embeddings = model.encode(questions, batch_size=64,
                              show_progress_bar=True, normalize_embeddings=True)
    np.save(EMB_FILE, embeddings)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)
    print("Embeddings cached.")

print("Search engine ready!")


# ── Core search ──────────────────────────────────────────────
def search(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        return []
    q_vec  = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q_vec
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [
        {
            "rank":     int(r),
            "score":    round(float(scores[i]), 4),
            "question": faqs[i]["question"],
            "answer":   faqs[i]["answer"],
        }
        for r, i in enumerate(top_idx, 1)
    ]


# ── Result renderer ──────────────────────────────────────────
def render_html(results: list[dict], query: str) -> str:
    if not results:
        return "<p style='color:#64748b;font-family:Inter,sans-serif;padding:16px'>No results found for this query.</p>"

    cards = ""
    for r in results:
        score = r["score"]
        score_pct = int(score * 100)
        bar_w = score_pct
        color = (
            "#4ade80" if score > 0.70
            else "#facc15" if score > 0.50
            else "#f87171"
        )
        answer = r["answer"][:350] + ("..." if len(r["answer"]) > 350 else "")
        delay  = (r["rank"] - 1) * 80
        cards += f"""
        <div class="faq-card" style="animation-delay:{delay}ms">
          <div class="card-header">
            <span class="rank-badge">#{r['rank']}</span>
            <div class="score-wrap">
              <div class="score-bar-bg">
                <div class="score-bar" style="--w:{bar_w}%; background:{color};"></div>
              </div>
              <span class="score-val" style="color:{color};">{score_pct}%</span>
            </div>
          </div>
          <p class="card-question">{r['question']}</p>
          <p class="card-answer">{answer}</p>
        </div>"""

    return f"""
    <div class="results-wrap">
      <p class="results-meta">Showing top {len(results)} matches for <em>"{query}"</em></p>
      {cards}
    </div>"""



# ── Gradio handler ───────────────────────────────────────────
def handle_search(query: str, top_k: int) -> str:
    if not query.strip():
        return "<p style='color:#888;font-family:sans-serif;padding:12px'>Type a question above and click Search.</p>"
    results = search(query, top_k=int(top_k))
    return render_html(results, query)


# ── UI ───────────────────────────────────────────────────────
EXAMPLES = [
    ["What is the capital of France?", 5],
    ["Who invented the telephone?", 5],
    ["How does photosynthesis work?", 5],
    ["What causes earthquakes?", 5],
    ["When did World War 2 end?", 5],
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Keyframes ────────────────────────────────────────── */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes titleGlow {
  0%, 100% { text-shadow: 0 0 0px rgba(148,163,184,0); }
  50%       { text-shadow: 0 0 18px rgba(148,163,184,0.18); }
}
@keyframes cardIn {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes barExpand {
  from { width: 0%; }
  to   { width: var(--w); }
}
@keyframes btnPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.35); }
  50%       { box-shadow: 0 0 0 6px rgba(99,102,241,0); }
}
@keyframes inputFocus {
  from { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
  to   { box-shadow: 0 0 0 3px rgba(99,102,241,0.3); }
}

/* ── Base ─────────────────────────────────────────────── */
body, .gradio-container {
  background: #0a0a0f !important;
  font-family: 'Inter', sans-serif !important;
}
footer { display: none !important; }

/* ── Header ───────────────────────────────────────────── */
#site-title {
  text-align: center;
  font-family: 'Inter', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: #f1f5f9;
  margin: 32px 0 6px 0;
  animation: fadeSlideUp 0.7s cubic-bezier(0.16,1,0.3,1) both,
             titleGlow 4s ease-in-out 0.7s infinite;
}
#site-divider {
  width: 48px;
  height: 3px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  border-radius: 2px;
  margin: 0 auto 28px auto;
  animation: fadeSlideUp 0.7s 0.15s cubic-bezier(0.16,1,0.3,1) both;
}

/* ── Search button ────────────────────────────────────── */
.gr-button-primary, button[variant='primary'] {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: 0.01em;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
  animation: btnPulse 2.4s ease-in-out 1s infinite;
}
.gr-button-primary:hover, button[variant='primary']:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
}
.gr-button-primary:active, button[variant='primary']:active {
  transform: translateY(0px) !important;
}

/* ── Search area width ───────────────────────────────── */
#search-row {
  max-width: 740px !important;
  margin: 0 auto !important;
}
#search-btn {
  max-width: 740px !important;
  margin: 0 auto !important;
  display: block;
}

.results-wrap {
  padding: 4px;
}
.results-meta {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: #64748b;
  margin: 0 0 14px 2px;
  letter-spacing: 0.02em;
  animation: fadeSlideUp 0.4s ease both;
}
.faq-card {
  background: #13131a;
  border: 1px solid #1e1e2e;
  border-radius: 14px;
  padding: 18px 22px;
  margin-bottom: 10px;
  font-family: 'Inter', sans-serif;
  opacity: 0;
  animation: cardIn 0.45s cubic-bezier(0.16,1,0.3,1) forwards;
  transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.faq-card:hover {
  transform: translateY(-3px);
  border-color: #6366f1;
  box-shadow: 0 8px 28px rgba(99,102,241,0.12);
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.rank-badge {
  font-size: 11px;
  font-weight: 600;
  color: #475569;
  letter-spacing: 0.04em;
}
.score-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
.score-bar-bg {
  width: 72px;
  height: 4px;
  background: #1e1e2e;
  border-radius: 4px;
  overflow: hidden;
}
.score-bar {
  height: 100%;
  border-radius: 4px;
  width: 0%;
  animation: barExpand 0.7s cubic-bezier(0.16,1,0.3,1) 0.3s forwards;
}
.score-val {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.card-question {
  margin: 0 0 8px 0;
  font-size: 15px;
  font-weight: 600;
  color: #e2e8f0;
  line-height: 1.45;
}
.card-answer {
  margin: 0;
  font-size: 13.5px;
  color: #64748b;
  line-height: 1.65;
}
"""


with gr.Blocks(
    css=CSS,
    title="FAQ Semantic Search",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as demo:

    gr.HTML("<h1 id='site-title'>General Knowledge FAQ</h1>")
    gr.HTML("<div id='site-divider'></div>")


    with gr.Row(elem_id="search-row"):
        with gr.Column(scale=5):
            query_box = gr.Textbox(
                placeholder="Ask anything... e.g. 'Who wrote Hamlet?'",
                label="Your Question",
                lines=1,
                autofocus=True,
            )
        with gr.Column(scale=1, min_width=160):
            top_k_slider = gr.Number(
                value=5,
                minimum=1,
                maximum=20,
                step=1,
                label="Enter the number of relevant questions needed",
                precision=0,
            )

    search_btn = gr.Button("Search", variant="primary", elem_id="search-btn")

    output = gr.HTML(
        value="<p style='color:#585b70;font-family:sans-serif;padding:12px'>"
              "Enter a question above to get started.</p>"
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_box, top_k_slider],
        outputs=output,
        fn=handle_search,
        cache_examples=False,
        label="Try these examples",
    )

    # Triggers
    search_btn.click(fn=handle_search, inputs=[query_box, top_k_slider], outputs=output)
    query_box.submit(fn=handle_search, inputs=[query_box, top_k_slider], outputs=output)


if __name__ == "__main__":
    demo.launch()
