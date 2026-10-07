# OKF Retrieval Ablation Study

## 1. Objective

This ablation study evaluates the contribution of the OKF-based knowledge expansion and structural scoring mechanisms to retrieval performance.

The study addresses two questions:

1. Does OKF-based knowledge expansion improve retrieval performance compared with conventional semantic retrieval?
2. Does OKF structural scoring provide an additional benefit after OKF-based knowledge expansion is applied?

The study separates the two mechanisms so that their individual and combined contributions can be observed.

---

## 2. Experimental Configurations

Three retrieval configurations were evaluated using the same evaluation benchmark.

| Configuration | Semantic Retrieval | OKF Knowledge Expansion | OKF Structural Scoring |
|---|---|---|---|
| Baseline RAG | Yes | No | No |
| OKF Ablation | Yes | Yes | No |
| Full OKF-aware | Yes | Yes | Yes |

### Baseline RAG

The baseline uses conventional semantic retrieval without OKF-based concept expansion or structural scoring.

### OKF Ablation

The ablation retains OKF-based knowledge expansion but disables the OKF structural scoring component.

This isolates the contribution of the OKF-expanded candidate set.

### Full OKF-aware Retrieval

The full OKF-aware configuration uses both:

- OKF-based knowledge expansion
- OKF structural scoring for reranking

---

## 3. Controlled Variables

The following were kept consistent across the three configurations:

- Evaluation questions
- Knowledge corpus
- Required evidence definitions
- Embedding model
- Retrieval methodology
- Evaluation metrics
- Top-K evaluation
- Execution environment

The evaluation benchmark contains 30 questions covering multiple retrieval categories.

The categories include:

- Explainability
- Factual
- Multi-entity
- Multi-hop
- Relationship reasoning

---

## 4. Evaluation Metrics

The following metrics were used.

### Hit@1

Measures whether the required evidence is retrieved at rank 1.

### Hit@3

Measures whether the required evidence is retrieved within the Top-3 results.

### Evidence Recall@3

Measures the proportion of required evidence items retrieved within the Top-3 results.

Evidence Recall@3 is particularly relevant for questions requiring multiple pieces of evidence because Hit@3 alone does not capture the completeness of the retrieved evidence set.

---

## 5. Hypotheses

### H1 — Knowledge Expansion

OKF-based knowledge expansion is expected to improve retrieval evidence coverage by using structured relationships to identify additional relevant knowledge beyond the documents surfaced by conventional semantic retrieval.

### H2 — Structural Scoring

Adding OKF structural scoring to the expanded candidate set is expected to provide additional retrieval improvement beyond knowledge expansion alone.

---

## 6. Results

### 6.1 Overall Retrieval Results

| Metric | Baseline RAG | OKF Ablation | Full OKF-aware |
|---|---:|---:|---:|
| Hit@1 | 0.833 | 0.833 | 0.933 |
| Hit@3 | 0.367 | 0.667 | 0.733 |
| Evidence Recall@3 | 0.660 | 0.854 | 0.882 |

### 6.2 Incremental Effect of OKF Knowledge Expansion

Comparing the baseline with the OKF ablation:

| Metric | Baseline → OKF Ablation |
|---|---:|
| Hit@1 | +0.000 |
| Hit@3 | +0.300 |
| Evidence Recall@3 | +0.194 |

The OKF expansion therefore produced a substantial improvement in Top-3 retrieval and evidence coverage even without structural scoring.

### 6.3 Incremental Effect of Structural Scoring

Comparing the OKF ablation with the full OKF-aware configuration:

| Metric | OKF Ablation → Full OKF |
|---|---:|
| Hit@1 | +0.100 |
| Hit@3 | +0.066 |
| Evidence Recall@3 | +0.028 |

The results indicate an additional improvement when structural scoring is applied to the OKF-expanded candidate set.

---

## 7. Category-Level Analysis

The full OKF-aware configuration produced the following category-level results.

### Category Impact

| Category | Questions | Improved | No Change | Regressed |
|---|---:|---:|---:|---:|
| Explainability | 6 | 5 | 1 | 0 |
| Factual | 6 | 0 | 6 | 0 |
| Multi-entity | 1 | 1 | 0 | 0 |
| Multi-hop | 8 | 6 | 2 | 0 |
| Relationship reasoning | 9 | 6 | 3 | 0 |

### Category Hit@3

| Category | Baseline Hit@3 | Full OKF Hit@3 | Δ Hit@3 |
|---|---:|---:|---:|
| Explainability | 0.167 | 1.000 | +0.833 |
| Factual | 1.000 | 1.000 | 0.000 |
| Multi-entity | 0.000 | 0.000 | 0.000 |
| Multi-hop | 0.125 | 0.375 | +0.250 |
| Relationship reasoning | 0.333 | 0.778 | +0.444 |

### Category Evidence Recall@3

| Category | Baseline ER@3 | Full OKF ER@3 | Δ ER@3 |
|---|---:|---:|---:|
| Explainability | 0.694 | 1.000 | +0.306 |
| Factual | 1.000 | 1.000 | 0.000 |
| Multi-entity | 0.250 | 0.500 | +0.250 |
| Multi-hop | 0.548 | 0.744 | +0.196 |
| Relationship reasoning | 0.556 | 0.889 | +0.333 |

---

## 8. Observations

The results show that OKF-based retrieval provides the largest improvement for categories involving structured relationships and multiple pieces of evidence.

In particular:

- Explainability shows a substantial improvement in Hit@3.
- Relationship reasoning shows a substantial improvement in both Hit@3 and Evidence Recall@3.
- Multi-hop retrieval also shows measurable improvement.
- Factual questions show no measurable improvement because the baseline already achieves complete retrieval performance for this category.
- No regression was observed in the evaluated benchmark.

The ablation results indicate that a significant portion of the overall improvement comes from OKF-based knowledge expansion. Structural scoring provides an additional, smaller improvement after the candidate set has been expanded.

---

## 9. Overall Impact

For the full OKF-aware configuration, the evaluation produced:

- Improved: 18 questions
- No change: 12 questions
- Regressed: 0 questions
- Total: 30 questions

The results therefore indicate a positive retrieval impact for the evaluated benchmark, with no observed regression.

The ablation further indicates that the improvement can be separated into two contributions:

1. **OKF knowledge expansion** provides the major improvement in Top-3 retrieval and evidence coverage.
2. **OKF structural scoring** provides an additional improvement beyond knowledge expansion.

---

## 10. Interpretation

The experiment provides evidence that a lightweight OKF-based knowledge representation layer can influence retrieval by introducing structured knowledge relationships into the retrieval process.

The comparison between the baseline and OKF ablation demonstrates the contribution of knowledge expansion independently of structural scoring.

The comparison between the OKF ablation and the full OKF-aware configuration demonstrates that structural scoring can provide an additional retrieval benefit after knowledge expansion.

The results also suggest that the benefit of structured knowledge is more pronounced for questions requiring relationships, multi-hop reasoning, explainability, or multiple pieces of evidence than for straightforward factual questions.

---

## 11. Limitations

The evaluation is based on a 30-question synthetic telecom/network knowledge benchmark.

Therefore, the results should not be interpreted as evidence of general superiority across domains or larger datasets.

The current experiment evaluates retrieval performance and does not yet evaluate:

- Runtime overhead
- Memory overhead
- Storage overhead
- Knowledge authoring effort
- Knowledge maintenance effort
- Scalability to larger knowledge bases

These aspects will be evaluated separately as part of the cost-benefit study.

---

## 12. Experiment Status

The retrieval implementation and evaluation configuration used for this ablation study are considered **frozen**.

Future experiments should not modify this implementation when comparing against these recorded results.

The next evaluation phase will investigate the cost-benefit relationship between the observed retrieval improvement and the additional computational, storage, and engineering overhead introduced by OKF.

