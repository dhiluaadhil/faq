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
        return "<p style='color:#888;font-family:sans-serif'>No results found.</p>"

    cards = ""
    for r in results:
        score = r["score"]
        bar_w = int(score * 100)
        color = (
            "#a6e3a1" if score > 0.70
            else "#f9e2af" if score > 0.50
            else "#f38ba8"
        )
        answer = r["answer"][:350] + ("..." if len(r["answer"]) > 350 else "")
        cards += f"""
        <div style="
            background:#1e1e2e; border:1px solid #313244; border-radius:12px;
            padding:16px 20px; margin-bottom:12px; font-family:sans-serif;
        ">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:12px; color:#6c7086; font-weight:600;">#{r['rank']}</span>
            <div style="display:flex; align-items:center; gap:8px;">
              <div style="width:80px; height:6px; background:#313244; border-radius:4px; overflow:hidden;">
                <div style="width:{bar_w}%; height:100%; background:{color}; border-radius:4px;"></div>
              </div>
              <span style="font-size:13px; font-weight:700; color:{color};">{score:.3f}</span>
            </div>
          </div>
          <p style="margin:0 0 8px 0; font-size:15px; font-weight:600; color:#cdd6f4;">
            {r['question']}
          </p>
          <p style="margin:0; font-size:14px; color:#a6adc8; line-height:1.55;">
            {answer}
          </p>
        </div>"""

    return f"""
    <div style="background:#11111b; padding:4px; border-radius:14px;">
      <p style="font-family:sans-serif; color:#cba6f7; font-size:13px; margin:0 0 12px 8px;">
        Top {len(results)} results for <em>"{query}"</em>
      </p>
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
body, .gradio-container { background: #11111b !important; }
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #6c7086; margin-bottom: 20px; font-size: 14px; }
.gr-button-primary { background: linear-gradient(135deg, #cba6f7, #89b4fa) !important;
                     border: none !important; color: #11111b !important; font-weight: 700 !important; }
footer { display: none !important; }
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

    gr.HTML("<h1 id='title'>🔍 FAQ Semantic Search Engine</h1>")
    gr.HTML("<p id='subtitle'>Powered by <code>all-MiniLM-L6-v2</code> · wiki_qa dataset · cosine similarity</p>")

    with gr.Row():
        with gr.Column(scale=5):
            query_box = gr.Textbox(
                placeholder="Ask anything... e.g. 'Who wrote Hamlet?'",
                label="Your Question",
                lines=1,
                autofocus=True,
            )
        with gr.Column(scale=1, min_width=120):
            top_k_slider = gr.Slider(
                minimum=1, maximum=15, value=5, step=1, label="Top-K"
            )

    search_btn = gr.Button("Search", variant="primary")

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
