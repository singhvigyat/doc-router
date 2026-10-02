# Does escalating to the LLM earn its cost?

Read-only analysis of the tuned router run (`THRESHOLD_1=0.60`, `THRESHOLD_2=0.75`). No new inference and no new API calls: DistilBERT's side comes from `results/local_predictions.jsonl`, the LLM's side from the `router` rows of `results/benchmark_predictions.jsonl` (165 cache hits, 45 fresh calls). Regenerate with `python scripts/escalation_subset_analysis.py`.

## Why the benchmark table can't answer this

`llm_only` (0.672) and `transformer_only` (0.746) are measured on different populations: DistilBERT on all 512 docs including the easy ones it is confident about, Gemini on a random 64-doc subsample that is also a mix of easy and hard. Neither describes the docs the router actually escalates. The honest comparison is both models on the same gated subset: the 223 docs where the baseline cleared neither gate and DistilBERT's own confidence fell below 0.75.

## Same-population accuracy on the LLM-gated subset

| model | subset | n | accuracy | macro-F1 |
|---|---|---:|---:|---:|
| DistilBERT (own low-confidence guess) | all LLM-gated docs | 223 | 0.525 | 0.483 |
| DistilBERT (own low-confidence guess) | LLM-scored docs only | 210 | 0.524 | 0.486 |
| Gemini (escalation) | LLM-scored docs only | 210 | 0.533 | 0.490 |

The last two rows are the paired comparison: same 210 documents, same gold labels. The LLM is right on 112 of them, DistilBERT on 110 — a difference of 2 documents, +1.0 pp.

The 13 remaining gated docs are excluded, not scored as wrong: their Gemini calls returned 504 DEADLINE_EXCEEDED with retries exhausted, so no LLM prediction exists (they are absent from the classify cache). Counting them either way would bias the comparison. For reference DistilBERT got 7 of those 13 right.

## Agreement breakdown on the paired subset

All 210 docs where both models produced a prediction:

| | LLM right | LLM wrong | total |
|---|---:|---:|---:|
| **DistilBERT right** | 76 | 34 | 110 |
| **DistilBERT wrong** | 36 | 64 | 100 |
| **total** | 112 | 98 | 210 |

The two off-diagonal cells are what matters. The LLM rescues 36 docs DistilBERT got wrong, but breaks 34 docs DistilBERT already had right — a net gain of 2 documents out of 210. Exact McNemar on the 70 discordant pairs gives p = 0.90; the 95% CI on the paired accuracy difference is [-6.9, +8.8] pp, comfortably straddling zero. Escalation is not adding accuracy here, it is reshuffling errors: 64 docs (30%) are missed by both models, so the gate is mostly selecting documents that are genuinely hard for either model rather than ones the LLM can fix.

### Where the swaps happen, by true class

| true label | n | LLM rescues | LLM breaks | net |
|---|---:|---:|---:|---:|
| budget | 31 | 2 | 10 | -8 |
| email | 10 | 0 | 1 | -1 |
| form | 40 | 7 | 9 | -2 |
| invoice | 45 | 9 | 4 | +5 |
| letter | 23 | 5 | 3 | +2 |
| memo | 33 | 5 | 3 | +2 |
| resume | 4 | 1 | 1 | +0 |
| scientific report | 24 | 7 | 3 | +4 |

The net figure is close to zero because the per-class effects cancel: the LLM is stronger on prose-like classes and weaker where DistilBERT has learned this corpus's layout cues.

## Cost of those corrections

The 210 escalations cost an estimated $0.0184 ($0.087 per 1k escalated docs) and bought 2 net-correct documents, i.e. roughly $0.009 per additional correct classification. Escalation also dominates the router's latency profile (2.0 s average vs 655 ms for DistilBERT alone).

Replacing the LLM tier with DistilBERT's own guess and leaving everything else identical scores 379/499 = 0.760, against the router's actual 381/499 = 0.764 — the same +0.4 pp, 2 documents.

## Conclusion

**Escalation is not clearly earning its cost on this data.** On the 210 gated documents both models were scored on, Gemini reached 0.533 accuracy against DistilBERT's own 0.524 on the exact same documents — 2 additional correct documents out of 210 (+1.0 pp, McNemar p = 0.90), which is indistinguishable from noise at this sample size. The 2x2 breakdown shows why: the LLM fixes 36 of DistilBERT's errors while introducing 34 new ones, and 64 of 210 documents defeat both models. The useful reading is that low DistilBERT confidence is doing a good job of identifying genuinely hard documents and a poor job of identifying documents an LLM can rescue — those are different things, and the current gate only measures the first. The right follow-up is a different escalation criterion (for example gating on the classes where the LLM actually wins, or on disagreement between the baseline and DistilBERT) rather than a lower confidence threshold, which would only send more of the same both-wrong documents to a paid model. The overall `llm_only` figure (0.672) should not be used to argue either side of this: it is measured on a different population and understates the LLM on hard documents just as it overstates it on easy ones.
