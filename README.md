# Scaling Bias or Biasing Scale? Measuring Social Stereotype Bias Across Ten Small Language Models

Systematic evaluation of social stereotype bias across **ten open-source Small
Language Models (SLMs)** spanning **135M → 9B parameters** and five model
families (SmolLM2, Qwen2.5, TinyLlama, Gemma2, Phi), measured on three
established benchmarks — **BBQ**, **CrowS-Pairs**, and **StereoSet** — under
identical conditions.

## Key Findings

- **Parameter count alone does not predict fairness.** The log-linear fit of
  the Mean Bias Index (MBI) against scale is weak and non-significant
  (*r* = 0.49, *p* = 0.154).
- **Bias scaling is benchmark-contingent.** BBQ shows a significant *negative*
  trend (*r* = −0.69, *p* = 0.028) — larger models are less biased — while
  StereoSet shows a significant *positive* trend (*r* = 0.79, *p* = 0.007).
  The two benchmarks are negatively correlated with each other.
- **Training data composition matters more than scale.** Phi-2 (synthetic data)
  has the lowest BBQ bias of all ten models; TinyLlama (over-trained, 3T tokens)
  has the highest CrowS-Pairs stereotype preference.

## Repository Structure

```
bias-scale-slm/
├── notebooks/                  Self-contained Colab notebooks (one per model family)
│   ├── bias_scale_smollm2.ipynb          SmolLM2-135M, SmolLM2-1.7B
│   ├── bias_scale_qwen_0.5-1.5-3B.ipynb  Qwen2.5-0.5B/1.5B/3B
│   ├── bias_scale_qwen_7B.ipynb          Qwen2.5-7B   (needs L4/A100)
│   ├── bias_scale_tinyllama.ipynb        TinyLlama-1.1B
│   ├── bias_scale_gemma2_2B.ipynb        Gemma2-2B
│   ├── bias_scale_gemma2_9B.ipynb        Gemma2-9B    (needs L4/A100)
│   ├── bias_scale_phi2.ipynb             Phi-2
│   └── combined_analysis.ipynb           Merges all results → figures + stats
├── results/                    Per-item predictions (JSONL) + per-model summaries (CSV)
│   ├── <family>/*.jsonl                  Raw per-item scores
│   ├── <family>/summary_*.csv            Aggregated per-model metrics
│   └── combined_summary.csv              All ten models, one row each
├── figures/                    Publication figures (PNG)
└── paper/                      IEEE conference manuscript (final PDF)
```

## Reproducing the Results

### 1. Scoring (per model)
Each notebook in `notebooks/` is self-contained and runs on Google Colab.
They pin `transformers==4.49.0` and `datasets==3.6.0`, write per-item
predictions to JSONL, and resume automatically on reconnect by skipping
cached items. The two largest models (Qwen2.5-7B, Gemma2-9B) require an
L4 or A100 GPU; the rest run on a free-tier T4.

### 2. Combined Analysis
`notebooks/combined_analysis.ipynb` reads every JSONL in `results/`,
recomputes all metrics uniformly, and regenerates the figures and the
statistical tables. Run it from the repository root.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/combined_analysis.ipynb
```

## Methodology Notes

- **Likelihood-based scoring** throughout (no generation), matching the
  EleutherAI `lm-evaluation-harness` paradigm for base models.
- **bfloat16** precision; 4-bit quantisation is deliberately avoided as it
  perturbs logits and breaks cross-model comparability.
- **Gemma2** models use `attn_implementation="eager"` — required for correct
  logit soft-capping (SDPA/FlashAttention corrupt the log-probabilities).
- **Mean Bias Index (MBI)** = mean of three normalised, magnitude-based bias
  indices (one per benchmark). The StereoSet term uses |SS−50|/50 rather than
  ICAT, which would conflate fluency with bias and penalise smaller models.

## Benchmarks

| Benchmark   | Items  | Source |
|-------------|--------|--------|
| BBQ         | 11,000 (1,000 × 11 categories, balanced) | `heegyu/bbq` |
| CrowS-Pairs | 1,508 (full)  | authors' canonical CSV |
| StereoSet   | ~2,123 (intrasentence validation) | `McGill-NLP/stereoset` |

## Citation

If you use this code or data, please cite the accompanying paper (see
`paper/`). BibTeX will be added upon publication.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
