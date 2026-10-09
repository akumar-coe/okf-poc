===================================================================================================================
Repeatability: Full OKF-Aware Experiment
===================================================================================================================
Configuration: src/run_okf_retrieval.py
Repetitions: 3
Environment: Linux VM; local ONNX embedding model; CPU execution.
Metric                      Run 1   Run 2   Run 3   Mean
Elapsed time (s)            5.69    5.58    5.59    5.62
Peak resident memory (KB)   958032  991720  958136  969296

All three runs exited successfully (exit status 0).

Observation: Elapsed time was consistent across the three runs. Peak memory showed greater variation.

Scope: These are end-to-end process measurements, including initialisation and both baseline and OKF-aware retrieval. 
They must not be interpreted as the incremental resource cost of OKF.

Status: Initial repeatability measurements recorded; comparative resource measurements remain in progress.

===================================================================================================================
Repeatability: OKF Ablation Experiment
===================================================================================================================
Configuration: src/run_okf_ablation.py
Mode: ABLATION_MODE = True
Repetitions: 3
Environment: Linux VM; local ONNX embedding model; CPU execution.
Metric                      Run 1   Run 2   Run 3   Mean
Elapsed time (s)            5.69    5.56    5.59    5.61
Peak resident memory (KB)   955064  957032  956832  956309

All three runs exited successfully (exit status 0).

Observation: Runtime and peak memory were consistent across the three repetitions.

Scope: These are end-to-end process measurements, including initialisation, corpus embedding and indexing, and the complete 
ablation experiment. They do not isolate the incremental resource cost of OKF expansion or structural scoring.

Comparison with full OKF-aware retrieval: The ablation and full OKF-aware configurations showed similar end-to-end runtime
and memory usage in these runs. This is an observation from the current VM and workload, not evidence that structural scoring
has zero overhead.

Status: Ablation repeatability measurements recorded.


===================================================================================================================
Retrieval Quality Comparison
===================================================================================================================
Objective - This experiment compares conventional RAG, OKF-based candidate expansion without structural scoring, and
full OKF-aware retrieval. 
The objective is to assess retrieval benefits alongside the measured resource requirements of each configuration.

Retrieval Results - All three configurations were evaluated using the same 30-question synthetic telecom benchmark.
Metric            Baseline RAG           OKF Ablation              Full OKF
Hit@1             0.833                  0.833                     0.933
Hit@3             0.367                  0.667                     0.733
Evidence Recall@3 0.660                  0.854                     0.882

============
Observations
============
Candidate expansion: Compared with baseline RAG, the OKF ablation increased Hit@3 from 0.367 to 0.667 and 
Evidence Recall@3 from 0.660 to 0.854, while Hit@1 remained unchanged.

Structural scoring: Compared with the ablation, full OKF increased Hit@1 from 0.833 to 0.933, Hit@3 from
0.667 to 0.733, and Evidence Recall@3 from 0.854 to 0.882.

Resource requirements: The measured mean runtime was comparable across all configurations. 
Full OKF had a higher mean peak resident memory than both baseline RAG and the ablation configuration.
These results suggest that candidate expansion accounts for most of the observed improvement in top-three
retrieval and evidence recall, while structural scoring provides an additional improvement in this benchmark.
The results are preliminary and specific to the evaluated synthetic telecom dataset. 
They do not establish general superiority over conventional RAG.

==============================================================================================================================
Comparative Resource Analysis
==============================================================================================================================
Current measurements
Configuration                    Mean elapsed time (s)   Mean peak resident memory (KB)
Full OKF-aware                   5.62                    969296
OKF ablation                     5.61                    956309
Difference (full minus ablation) +0.01                   +12987

The full OKF-aware configuration had a mean elapsed time approximately 0.01 seconds higher than the ablation configuration. Its mean peak resident memory was approximately 12.5 MB higher, using decimal MB.
These differences are observations from the current VM and workload. They should not be interpreted as statistically significant differences or as isolated measurements of the structural-scoring overhead.

Experimental comparability
The configurations use the same local ONNX embedding model, CPU execution environment and evaluation benchmark. However, the baseline retrieval script retrieves the Top-3 directly, whereas the OKF-aware pipeline uses a larger initial semantic candidate pool (BASELINE_K = 10) before selecting the final Top-3.
Consequently, comparisons involving baseline RAG and the OKF configurations reflect differences in the overall retrieval pipelines, including candidate-pool size. They do not isolate the effect of OKF structure alone.

The three configurations use the same 30-question synthetic telecom benchmark and the same local ONNX embedding model. However, their retrieval pipelines differ: baseline RAG retrieves the top results directly, while the OKF configurations use candidate expansion and, for full OKF, structural scoring.
Consequently, the observed differences reflect the combined effects of these pipeline changes rather than the isolated effect of knowledge representation alone. The ablation comparison helps distinguish the contribution of candidate expansion from the additional contribution of structural scoring.
Resource measurements are end-to-end process measurements from the current Linux VM. Three repetitions were recorded for each configuration. These measurements provide an initial comparison under the current environment and workload; they are not a statistical demonstration of equivalence or proof that any component introduces no overhead.


Limitations
Measurements include model initialization, corpus embedding, indexing and retrieval.
Only three repetitions have been recorded for each configuration.
Peak resident memory is a process-level measurement, not the incremental memory attributable to OKF.
The current measurements do not establish the statistical significance of the observed differences.
Baseline resource measurements must be verified before completing the three-way resource comparison.

Status
Repeatability measurements are recorded for full OKF-aware retrieval and OKF ablation. Comparative resource analysis remains preliminary pending verification of baseline measurements and interpretation of the candidate-pool difference.
