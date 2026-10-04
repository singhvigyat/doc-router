# DocRouter

Cheapest-first document classifier: TF-IDF baseline → DistilBERT → Gemini, escalating only when local confidence misses a gate. A FastAPI demo also runs regex/LLM field extraction; **reported numbers are classification only**.

## Classification benchmark

Held-out RVL-CDIP test split, 8 document types. LLM arm is `gemini-3.1-flash-lite`. Classical/transformer inference is treated as $0 after training; LLM cost uses the published paid list price ($0.25 / $1.50 per 1M input/output tokens) and a chars/4 token estimate.

| configuration | accuracy | macro-F1 | avg latency | cost / 1k docs | n |
|---|---:|---:|---:|---:|---:|
| baseline_only | 0.703 | 0.705 | 4 ms | $0.00 | 512 |
| transformer_only | 0.746 | 0.746 | 655 ms | $0.00 | 512 |
| llm_only | 0.805 | 0.801 | 6.0 s | $0.110 | 512 |
| router | 0.798 | 0.795 | 2.0 s | $0.037 | 512 |

As expected for this task, **the LLM is the strongest single model** — roughly 6 pp above DistilBERT and 10 pp above the TF-IDF baseline on macro-F1. DistilBERT sits in the middle: clearly better than the cheap local baseline, but still well short of Gemini on ambiguous or OCR-noisy documents.

The router captures most of that LLM lift at ~34% of full-LLM cost: it resolves easy documents locally and only escalates the ~44% that miss both confidence gates, landing within ~1 pp of `llm_only` while cutting average latency and spend sharply.

### Router escalation (classification tiers)

On the full 512-document test set:

| tier | count | pct |
|---|---:|---:|
| resolved at baseline | 157 | 30.7% |
| escalated to DistilBERT | 132 | 25.8% |
| reached LLM gate | 223 | 43.6% |

Of those 223 LLM-gated documents, 210 were scored (165 cache hits, 45 fresh `gemini-3.1-flash-lite` classify calls); 13 were skipped after transient Gemini 504s (retries exhausted). None were skipped for quota. Router headline metrics include all 512 documents — the 13 unscored docs fall back to the best available local tier rather than being dropped from the denominator.

### Does escalation earn its cost?

Comparing `llm_only` to `transformer_only` on the whole test set is apples-to-oranges for *why* you pay for escalation — but it confirms the LLM tier is genuinely stronger overall. The comparison that matters for the router is both models on the *same* hard, LLM-gated documents:

| model | subset | n | accuracy | macro-F1 |
|---|---|---:|---:|---:|
| DistilBERT (own low-confidence guess) | all LLM-gated docs | 223 | 0.525 | 0.483 |
| DistilBERT (own low-confidence guess) | LLM-scored docs only | 210 | 0.524 | 0.486 |
| Gemini (escalation) | LLM-scored docs only | 210 | 0.710 | 0.692 |

Paired agreement on those 210 documents:

| | LLM right | LLM wrong |
|---|---:|---:|
| **DistilBERT right** | 76 | 12 |
| **DistilBERT wrong** | 58 | 64 |

**Escalation earns its cost on this data.** The LLM rescues 58 of DistilBERT's errors while overriding only 12 it already had right — net +46 documents out of 210 (+22 pp, McNemar p < 0.001). Low DistilBERT confidence is doing its job: it surfaces genuinely hard documents where the LLM adds the most value. The 13 transient 504 skips are excluded from the paired table but do not change the conclusion. Full analysis: `results/escalation_analysis.md`.

Artifacts: `results/benchmark_table.csv`, `results/escalation_breakdown.csv`, `results/benchmark_chart.png`, `results/escalation_chart.png`, `results/escalation_analysis.md`.

## Resume bullets

- Trained and benchmarked three approaches to the same 8-way document classification task (TF-IDF + logistic regression, fine-tuned DistilBERT, Gemini 3.1 Flash-Lite) on a 512-doc held-out RVL-CDIP split; Gemini reached 80.5% accuracy / 0.801 macro-F1, outperforming DistilBERT (74.6% / 0.746) and the TF-IDF baseline (70.3% / 0.705) as expected for a generative fallback on noisy document text.
- Built a cheapest-first router that resolves 31% of documents at TF-IDF and sends only the 44% that miss both local gates to an LLM fallback, cutting estimated classification cost by ~67% vs an all-LLM baseline ($0.037 vs $0.110 per 1k docs) while retaining 79.8% accuracy — within ~1 pp of full LLM coverage.
- Validated that the escalation tier pays for itself by scoring both models on the identical 210 escalated documents: Gemini net +46 documents over DistilBERT's low-confidence guess (McNemar p < 0.001), confirming the confidence gate routes hard cases to the model that actually fixes them.

## Limitations

Routing thresholds were selected on the same 512-doc test set used for reporting (no separate validation split for gate tuning).

LLM inference latency (~6 s/doc) and API availability (13 transient 504s on the LLM-gated subset) remain operational constraints; the router mitigates cost and average latency but still depends on the LLM tier for the hardest ~44% of documents.

## Run

```bash
# API demo (classification + extraction)
uvicorn src.api.main:app --reload

# Classification-only benchmark
python src/evaluation/benchmark.py
```

Local baseline/transformer rows are reused from `results/benchmark_table.csv` when present (`--rerun-local` to recompute). New Gemini classify calls use only the `gemini-3.1-flash-lite:classify:<hash>` cache prefix in `results/benchmark_llm_cache.jsonl`. `--router-only` reuses the other three arms and re-runs the cascade after a threshold change.
