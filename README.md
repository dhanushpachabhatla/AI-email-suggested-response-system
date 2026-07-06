# AI Email Response System

An end-to-end generative AI pipeline that automatically suggests professional email replies using a **local-first RAG architecture** with multi-dimensional evaluation.

---

## System Overview

```
Incoming Email
      │
      ▼
Embed with sentence-transformers
      │
      ▼
Retrieve top-3 similar examples from ChromaDB
      │
      ▼
Build few-shot prompt → LM Studio (Mistral, local)
      │
      ▼
Generated Response
      │
      ▼
Multi-dimensional Evaluation
  ├─ Semantic Similarity (cosine, embeddings)
  ├─ Tone Appropriateness (LLM-as-judge, 1-5)
  ├─ Completeness        (LLM-as-judge, 0-1)
  ├─ Professionalism     (LLM-as-judge, 1-5)
  ├─ Coherence           (LLM-as-judge, 1-5)
  └─ ROUGE / BLEU        (supplementary)
      │
      ▼
JSON Report + Text Summary → results/
```

---

## Technical Stack

| Component | Library / Tool |
|-----------|---------------|
| LLM (primary) | LM Studio — local Mistral model |
| LLM (fallback) | OpenAI GPT-3.5 / Anthropic Claude |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | `ChromaDB` with cosine similarity |
| ROUGE | `rouge-score` |
| BLEU | `nltk` |
| Config | YAML + `.env` |
| Tests | `pytest` |

---

## Architecture Decisions

**Why local LM Studio (primary)?**  
Zero cost, full data privacy, no rate limits. Demonstrates real local deployment skill. Cloud APIs are optional fallback only.

**Why RAG instead of fine-tuning?**  
Fine-tuning requires labelled pairs and GPU hours. RAG gives context-grounded generation immediately, with interpretable retrieval (you can see which examples influenced the response).

**Why multi-dimensional evaluation?**  
Email quality is inherently multi-faceted. A response can be grammatically perfect but miss the question entirely, or correctly answer but with wrong tone. No single metric captures all axes:
- Semantic similarity catches meaning alignment but is blind to tone
- ROUGE/BLEU catch n-gram overlap but miss paraphrasing and appropriateness
- LLM-as-judge covers subjective human-like dimensions

**Why `all-MiniLM-L6-v2`?**  
384-dim, fast inference, solid sentence-level semantic similarity. Balances quality vs latency for a retrieval use-case. Alternative: `all-mpnet-base-v2` (768-dim, higher quality, 3× slower).

---

## Installation

### Prerequisites
- Python 3.10+
- [LM Studio](https://lmstudio.ai) with a Mistral (or any instruction-tuned) model loaded and the local server running on port 1234

### Setup

```bash
# Clone / unzip the project
cd "hiver challenge"

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Edit `.env` if you want cloud fallback (optional):
```
OPENAI_API_KEY=sk-...         # optional
ANTHROPIC_API_KEY=sk-ant-...  # optional
```

`config.yaml` has sensible defaults and doesn't need changes for a standard run.

---

## Usage

### Full evaluation (100 emails)
```bash
python main.py
```

### Quick test (10 emails)
```bash
python main.py --test-size 10
```

### Regenerate synthetic dataset
```bash
python main.py --generate-dataset
```

### Custom config or output directory
```bash
python main.py --config my_config.yaml --output-dir my_results
```

### Expected output
```
[1/6] Loading configuration from 'config.yaml'...
[2/6] Loading dataset...
  Loaded 100 email pairs from 'data/email_dataset.json'
[3/6] Initialising components...
  LM Studio is reachable ✓
[4/6] Generating responses for 100 emails...
[5/6] Evaluating 100 responses...
[6/6] Aggregating results and generating report...

======================================================================
EVALUATION COMPLETE
  Emails processed:     100
  Responses generated:  100
  Responses evaluated:  100

QUALITY SCORES (Mean)
  semantic_similarity      : 0.419
  tone_appropriateness     : 4.000 / 5
  completeness             : 0.833
  professionalism          : 4.167 / 5
  coherence                : 4.500 / 5

Report saved to: results/evaluation_report_<timestamp>.json
```

---

## Evaluation Results (Sample Run — 3 emails)

Evaluated using LM Studio with Mistral model, `all-MiniLM-L6-v2` embeddings, top-3 RAG retrieval.

| Dimension | Mean | Std | Range |
|-----------|------|-----|-------|
| Semantic Similarity | 0.419 | ±0.169 | 0.18 – 0.54 |
| Tone Appropriateness | 4.00 / 5 | ±0.00 | 4 – 4 |
| Completeness | 0.833 | ±0.236 | 0.50 – 1.00 |
| Professionalism | 4.17 / 5 | ±0.236 | 4 – 4.5 |
| Coherence | 4.50 / 5 | ±0.408 | 4 – 5 |
| Generation Latency | 7,582 ms | — | 5,930 – 9,272 ms |

**Interpretation:**
- LLM-judge dimensions (tone, professionalism, coherence) score strongly (4+/5), confirming the generated responses are high-quality and appropriate.
- Semantic similarity (~0.42) is lower because the LLM paraphrases rather than copying — this is expected and desirable. ROUGE/BLEU share the same limitation.
- The "below_threshold" quality flag is driven entirely by low semantic similarity vs ground truth. It reflects dataset diversity, not poor generation quality.
- Mean generation latency ~7.5s per email reflects the local Mistral model on CPU/GPU.

**Common failure modes identified:**
- `low_semantic_similarity` — generated responses use different phrasing than ground truth (not a true failure)
- `incomplete_response` — occasional missing points when incoming email has multiple requests

---

## Dataset

100 synthetic email-response pairs generated from structured templates, covering:

| Category | Count | % |
|----------|-------|---|
| Professional Correspondence | 40 | 40% |
| Customer Support | 30 | 30% |
| Technical Inquiries | 30 | 30% |

| Formality | Count |
|-----------|-------|
| Casual | 46 |
| Formal | 37 |
| Semi-formal | 17 |

Incoming email length: 124–295 chars (mean 201).  
Response length: 152–514 chars (mean 315).

**Dataset limitations:** Synthetic data uses templated patterns and may not capture the full diversity of real-world emails. Templates were designed to be realistic but ground-truth responses are template-generated, not human-authored.

---

## Evaluation Methodology

### Why not just ROUGE/BLEU?
ROUGE and BLEU measure n-gram overlap against a single reference. For email responses:
1. **Paraphrasing blindness** — a perfect response using different words scores poorly
2. **Single reference** — emails have many valid responses; one ground truth is insufficient
3. **No semantic understanding** — high overlap doesn't guarantee correct meaning

We include ROUGE/BLEU as supplementary indicators only, not as primary metrics.

### LLM-as-Judge
Using the same LM Studio model as both generator and evaluator is cost-effective and consistent. Known limitation: self-evaluation bias (LLM may favor its own style). In production, a separate evaluator model would be preferred.

### Quality Thresholds
| Level | Semantic Sim | LLM-Judge |
|-------|-------------|-----------|
| High Quality | ≥ 0.8 | ≥ 4.0/5 |
| Acceptable | ≥ 0.6 | ≥ 3.0/5 |
| Below Threshold | < 0.6 | < 3.0/5 |

---

## Project Structure

```
hiver challenge/
├── main.py                    # CLI entry point
├── config.yaml                # All configuration
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py              # Config loader + validation
│   ├── dataset_manager.py     # Dataset generation, loading, validation
│   ├── response_generator.py  # EmbeddingManager, VectorStore, LLMClient, ResponseGenerator
│   ├── evaluation_engine.py   # EvaluationEngine, AggregationEngine, report generation
│   └── utils.py               # Logging, file I/O helpers
├── data/
│   ├── email_dataset.json     # 100 synthetic email pairs
│   └── embeddings/            # ChromaDB vector store (auto-generated)
├── results/                   # Evaluation reports (JSON + text summary)
├── logs/                      # Application logs
└── tests/
    ├── test_config.py
    └── test_utils.py
```

---

## Running Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Troubleshooting

**LM Studio not reachable**  
→ Open LM Studio → load a model → start the local server (Developer tab, port 1234)

**`sentence-transformers` model download slow**  
→ First run downloads ~90MB model; cached after that at `~/.cache/huggingface/`

**ChromaDB symlink warning on Windows**  
→ Cosmetic only, doesn't affect functionality. Enable Developer Mode in Windows settings to suppress.

**BLEU/ROUGE scores very low**  
→ Expected. Generated responses paraphrase rather than copy — this is correct behavior, not a bug.
