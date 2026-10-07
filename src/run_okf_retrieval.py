import json
import re
import faiss
import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer
from collections import defaultdict
from evaluation import evaluate, mean_metric

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = "all-MiniLM-L6-v2"
MODEL_PATH = f"{MODEL_DIR}/onnx/model_quantized.onnx"
CORPUS_PATH = "research/dataset/baseline_chunks.json"
QUESTIONS_PATH = "research/questions/evaluation_questions.yaml"
INVENTORY_PATH = "reports/knowledge_inventory.json"
BASELINE_K = 10
FINAL_K = 3
# Number of OKF concepts selected from the query.
OKF_SEED_CONCEPTS = 2
# We will use the OKF signal only for reranking.
SEMANTIC_WEIGHT = 0.75
OKF_WEIGHT = 0.25


def interpret_no_change(result, question):
   """
   Explain why OKF produced no measurable change.
   """
   base_hit3 = bool(result["baseline_hit3"])
   okf_hit3 = bool(result["okf_hit3"])
   base_recall = float(result["baseline_recall"])
   okf_recall = float(result["okf_recall"])
   base_sources = set(result.get("baseline_sources", []))
   okf_sources = set(result.get("okf_sources", []))
   required = {
       item.replace("\\", "/")
       for item in question.get("required_evidence", [])
   }
   base_required = base_sources.intersection(required)
   okf_required = okf_sources.intersection(required)
   added_sources = okf_sources - base_sources
   # ---------------------------------------------------------
   # Case 1: Baseline already had complete evidence
   # ---------------------------------------------------------
   if base_hit3 and okf_hit3 and base_recall == 1.0:
       return (
           "Baseline already retrieved the complete required "
           "evidence set in Top-3, leaving no measurable "
           "retrieval gap for OKF to improve."
       )
   # ---------------------------------------------------------
   # Case 2: Same evidence and same metrics
   # ---------------------------------------------------------
   if (
       base_sources == okf_sources
       and base_hit3 == okf_hit3
       and base_recall == okf_recall
   ):
       return (
           "OKF did not change the retrieved Top-3 evidence or "
           "the evaluation outcome."
       )
   # ---------------------------------------------------------
   # Case 3: OKF changed sources but not required evidence
   # ---------------------------------------------------------
   if added_sources and okf_required == base_required:
       return (
           "OKF changed the retrieved sources but did not add "
           "any additional required evidence, so the evaluation "
           "outcome remained unchanged."
       )
   # ---------------------------------------------------------
   # Case 4: Metrics unchanged despite source movement
   # ---------------------------------------------------------
   if base_hit3 == okf_hit3 and base_recall == okf_recall:
       return (
           "OKF altered the retrieval composition without changing "
           "the required evidence coverage or Top-3 evaluation outcome."
       )
   # ---------------------------------------------------------
   # Fallback
   # ---------------------------------------------------------
   return (
       "OKF did not produce a measurable change in the evaluated "
       "retrieval outcome."
   )

def normalize_source(source):
   """Normalize source paths for comparison."""
   return source.replace("\\", "/")

def get_top3_sources(results):
   """Return unique source paths from Top-3 results."""
   return [
       normalize_source(item["source"])
       for item in results[:FINAL_K]
   ]

def explain_okf_change(question, baseline_top3, okf_top3):
   """
   Explain what changed between baseline and OKF-aware Top-3.
   """
   baseline_sources = get_top3_sources(baseline_top3)
   okf_sources = get_top3_sources(okf_top3)
   baseline_set = set(baseline_sources)
   okf_set = set(okf_sources)
   added = okf_set - baseline_set
   removed = baseline_set - okf_set
   required = {
       normalize_source(item)
       for item in question.get("required_evidence", [])
   }
   added_required = added.intersection(required)
   added_non_required = added - required
   if added_required:
       interpretation = (
           "OKF surfaced additional required evidence into the Top-3."
       )
   elif added:
       interpretation = (
           "OKF changed the Top-3 composition by surfacing additional "
           "evidence, but the newly surfaced source was not itself a "
           "required evidence source."
       )
   elif removed:
       interpretation = (
           "OKF changed the Top-3 composition by replacing one or more "
           "baseline sources."
       )
   else:
       interpretation = (
           "OKF changed the evidence coverage without changing the "
           "Top-3 source composition."
       )
   return {
       "added": sorted(added),
       "removed": sorted(removed),
       "added_required": sorted(added_required),
       "added_non_required": sorted(added_non_required),
       "interpretation": interpretation,
   }


# ============================================================
# Local ONNX embedding pipeline
# ============================================================
def mean_pool(token_embeddings, attention_mask):
   mask = attention_mask[..., None].astype(np.float32)
   summed = np.sum(
       token_embeddings * mask,
       axis=1
   )
   counts = np.clip(
       mask.sum(axis=1),
       a_min=1e-9,
       a_max=None
   )
   return summed / counts

def embed(texts, tokenizer, session):
   inputs = tokenizer(
       texts,
       return_tensors="np",
       padding=True,
       truncation=True,
       max_length=256,
   )
   outputs = session.run(
       None,
       {
           "input_ids":
               inputs["input_ids"].astype(np.int64),
           "attention_mask":
               inputs["attention_mask"].astype(np.int64),
           "token_type_ids":
               inputs["token_type_ids"].astype(np.int64),
       },
   )
   embeddings = mean_pool(
       outputs[0],
       inputs["attention_mask"],
   )
   # L2 normalization.
   # With normalized vectors, inner product == cosine similarity.
   embeddings = embeddings / np.linalg.norm(
       embeddings,
       axis=1,
       keepdims=True,
   )
   return embeddings.astype(np.float32)

# ============================================================
# Text normalization
# ============================================================
def normalize(text):
   return re.sub(
       r"[^a-z0-9]+",
       " ",
       text.lower()
   ).strip()

# ============================================================
# Load corpus
# ============================================================
with open(CORPUS_PATH, encoding="utf-8") as f:
   corpus = json.load(f)
print("Loaded corpus:", len(corpus), "chunks")

# ============================================================
# Load questions
# ============================================================
with open(QUESTIONS_PATH, encoding="utf-8") as f:
   benchmark = yaml.safe_load(f)
questions = benchmark["questions"]
print("Loaded questions:", len(questions))

# ============================================================
# Load OKF inventory
# ============================================================
with open(INVENTORY_PATH, encoding="utf-8") as f:
   inventory = json.load(f)
concepts = inventory["concepts"]
relationships = inventory["relationships"]
print("OKF concepts:", len(concepts))
print("OKF relationships:", len(relationships))

# ============================================================
# Build concept lookup
# ============================================================
concept_by_id = {
   concept["id"]: concept
   for concept in concepts
}

# ============================================================
# Build undirected relationship graph
# ============================================================
#
# For this PoC we treat relationships as navigational links.
# This allows one-hop expansion in either direction.
#
# IMPORTANT:
# We are not using required_evidence here.
# ============================================================
graph = {}
for relationship in relationships:
   source = relationship["source"]
   target = relationship["target"]
   graph.setdefault(source, set()).add(target)
   graph.setdefault(target, set()).add(source)

# ============================================================
# Map corpus source -> OKF concept
# ============================================================
def source_to_concept(source):
   source = source.replace("\\", "/")
   if source in concept_by_id:
       return source
   return None

# ============================================================
# Create searchable OKF concept descriptions
# ============================================================
def concept_text(concept):
   title = concept.get("title", "")
   description = concept.get("description", "")
   concept_type = concept.get("type", "")
   tags = concept.get("tags", [])
   domain = concept.get("domain", "")
   if isinstance(tags, list):
       tags_text = " ".join(
           str(tag)
           for tag in tags
       )
   else:
       tags_text = str(tags)
   return (
       f"{title}. "
       f"{description}. "
       f"{concept_type}. "
       f"{tags_text}. "
       f"{domain}"
   )

concept_texts = [
   concept_text(concept)
   for concept in concepts
]

# ============================================================
# Load LOCAL tokenizer
# ============================================================
print("Loading local tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
   MODEL_DIR,
   local_files_only=True,
)

# ============================================================
# Load LOCAL ONNX model
# ============================================================
print("Loading local ONNX model...")
session = ort.InferenceSession(
   MODEL_PATH,
   providers=["CPUExecutionProvider"],
)
print("Local MiniLM ONNX model loaded")

# ============================================================
# Embed corpus
# ============================================================
corpus_texts = [
   chunk["text"]
   for chunk in corpus
]
corpus_embeddings = embed(
   corpus_texts,
   tokenizer,
   session,
)
print(
   "Corpus embeddings:",
   corpus_embeddings.shape
)

# ============================================================
# FAISS index
# ============================================================
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(
   corpus_embeddings
)
print(
   "FAISS index created"
)
print(
   "Vectors indexed:",
   index.ntotal
)

# ============================================================
# Embed OKF concepts
# ============================================================
concept_embeddings = embed(
   concept_texts,
   tokenizer,
   session,
)
print(
   "OKF concept embeddings:",
   concept_embeddings.shape
)

# ============================================================
# Semantic baseline retrieval
# ============================================================
def semantic_search(
   query,
   k
):
   query_embedding = embed(
       [query],
       tokenizer,
       session,
   )[0]
   scores, indices = index.search(
       query_embedding.reshape(1, -1),
       k,
   )
   results = []
   for score, idx in zip(
       scores[0],
       indices[0],
   ):
       if idx < 0:
           continue
       results.append({
           "index": int(idx),
           "source": corpus[idx]["source"],
           "chunk_id": corpus[idx]["chunk_id"],
           "semantic_score": float(score),
           "text": corpus[idx]["text"],
       })
   return results, query_embedding

# ============================================================
# OKF concept identification
# ============================================================
def identify_seed_concepts(
   query_embedding
):
   scores = np.dot(
       concept_embeddings,
       query_embedding,
   )
   ranking = np.argsort(
       -scores
   )
   seeds = []
   for idx in ranking[
       :OKF_SEED_CONCEPTS
   ]:
       concept = concepts[idx]
       seeds.append({
           "id": concept["id"],
           "title": concept["title"],
           "score": float(scores[idx]),
       })
   return seeds

# ============================================================
# One-hop OKF expansion
# ============================================================
def expand_okf(
   seeds
):
   expanded = set()
   for seed in seeds:
       seed_id = seed["id"]
       expanded.add(
           seed_id
       )
       expanded.update(
           graph.get(
               seed_id,
               set()
           )
       )
   return expanded

# ============================================================
# Calculate OKF structural score
# ============================================================
def okf_score(
   source,
   seeds,
   expanded
):
   concept_id = source_to_concept(
       source
   )
   if concept_id is None:
       return 0.0
   seed_ids = {
       seed["id"]
       for seed in seeds
   }
   # Directly identified concept.
   if concept_id in seed_ids:
       return 1.0
   # One-hop related concept.
   if concept_id in expanded:
       return 0.5
   return 0.0

# ============================================================
# Run experiment
# ============================================================
print()
print("=" * 70)
print("OKF-AWARE RETRIEVAL EXPERIMENT")
print("=" * 70)
baseline_metrics = []
okf_metrics = []
question_results = []

for question in questions:
   qid = question["id"]
   query = question["question"]
   print()
   print("-" * 70)
   print(qid)
   print("Question:", query)
   print("-" * 70)
   # --------------------------------------------------------
   # Baseline semantic retrieval
   # --------------------------------------------------------
   baseline_results, query_embedding = semantic_search(
       query,
       BASELINE_K,
   )
   baseline_top3 = baseline_results[
       :FINAL_K
   ]
   baseline_eval = evaluate(
       baseline_top3,
       question,
   )
   # --------------------------------------------------------
   # Identify OKF concepts
   # --------------------------------------------------------
   seeds = identify_seed_concepts(
       query_embedding
   )
   expanded = expand_okf(
       seeds
   )
   # --------------------------------------------------------
   # Candidate sources
   # --------------------------------------------------------
   candidate_sources = set(
       expanded
   )
   candidate_indices = []
   for idx, chunk in enumerate(
       corpus
   ):
       source = chunk["source"]
       concept_id = source_to_concept(
           source
       )
       if concept_id in candidate_sources:
           candidate_indices.append(
               idx
           )
   
   assert len(baseline_results) == BASELINE_K
   print(
           f"OKF reranking candidate pool: "
           f"{len(baseline_results)} semantic_results"
        )
   # --------------------------------------------------------
   # OKF-aware reranking
   #
   # IMPORTANT:
   # We start from the SAME semantic
   # candidate pool as the baseline.
   #
   # OKF only changes ranking.
   # --------------------------------------------------------
   okf_results = []
   for result in baseline_results:
       structural = okf_score(
           result["source"],
           seeds,
           expanded,
       )
       combined = (
           SEMANTIC_WEIGHT
           * result["semantic_score"]
           +
           OKF_WEIGHT
           * structural
       )
       print(
             f"DEBUG source={result['source']} "
             f"concept={source_to_concept(result['source'])} "
             f"semantic={result['semantic_score']:.4f} "
             f"okf={structural:.2f} "
             f"combined={combined:.4f}"
            )
       enriched = dict(result)
       enriched["okf_score"] = structural
       enriched["combined_score"] = combined
       okf_results.append(
           enriched
       )
   
   assert len(okf_results) == len(baseline_results)
   okf_results.sort(
       key=lambda item: item["combined_score"],
       reverse=True,
   )
   okf_top3 = []
   seen_concepts = set()
   for item in okf_results:
       concept_id = source_to_concept(item["source"])
       if concept_id in seen_concepts:
           continue
       okf_top3.append(item)
       seen_concepts.add(concept_id)
       if len(okf_top3) == FINAL_K:
           break
   #okf_results.sort(
   #    key=lambda item:
   #        item["combined_score"],
   #    reverse=True,
   #)
   #okf_top3 = okf_results[
   #    :FINAL_K
   #]
   okf_eval = evaluate(
       okf_top3,
       question,
   )
   baseline_metrics.append(
       baseline_eval
   )
   okf_metrics.append(
       okf_eval
   )
   question_results.append({
       "id": qid,
       "category": question.get("category", "unknown"),
       "baseline_hit1": baseline_eval["hit1"],
       "okf_hit1": okf_eval["hit1"],
       "baseline_hit3": baseline_eval["hit3"],
       "okf_hit3": okf_eval["hit3"],
       "baseline_recall": baseline_eval["evidence_recall"],
       "okf_recall": okf_eval["evidence_recall"],
       # Keep Top-3 sources for interpretation
       "baseline_sources": [
           item["source"].replace("\\", "/")
           for item in baseline_top3
       ],
       "okf_sources": [
           item["source"].replace("\\", "/")
           for item in okf_top3
       ],
       # Required evidence for this question
       "required_evidence": [
           item.replace("\\", "/")
           for item in question.get("required_evidence", [])
       ],
   })
   question_results[-1]["no_change_interpretation"] = (
       interpret_no_change(
           question_results[-1],
           question
       )
   ) 
   # ------------------------------------------------------------
   # Diagnostic: compare baseline and OKF-aware Top-3
   # ------------------------------------------------------------
   if (
      baseline_eval["hit3"] != okf_eval["hit3"]
      or baseline_eval["evidence_recall"] != okf_eval["evidence_recall"]
   ):
      print()
      print("OKF IMPACT:")
      print(
         f"  Hit@3: "
         f"{baseline_eval['hit3']:.3f} -> "
         f"{okf_eval['hit3']:.3f}"
      )
      print(
         f"  Evidence Recall@3: "
         f"{baseline_eval['evidence_recall']:.3f} -> "
         f"{okf_eval['evidence_recall']:.3f}"
      )
      print()
      print("  Baseline Top-3:")
      for i, item in enumerate(baseline_top3, 1):
          print(
             f"    #{i} "
             f"{item['source']} "
             f"score={item.get('score', item.get('semantic_score', 0.0)):.4f}"
          )
      print()
      print("  OKF-aware Top-3:")
      for i, item in enumerate(okf_top3, 1):
         print(
            f"    #{i} "
            f"{item['source']} "
            f"combined={item['combined_score']:.4f}"
         )
   
   # --------------------------------------------------------
   # What changed in the Top-3?
   # --------------------------------------------------------
   baseline_sources = [
       item["source"]
       for item in baseline_top3
   ]
   okf_sources = [
       item["source"]
       for item in okf_top3
   ]
   added = [
       source
       for source in okf_sources
       if source not in baseline_sources
   ]
   removed = [
       source
       for source in baseline_sources
       if source not in okf_sources
   ]
   print()
   print("  TOP-3 SOURCE CHANGES:")
   if added:
       print("    Added by OKF:")
       for source in added:
           print(f"      + {source}")
   else:
       print("    Added by OKF: None")
   if removed:
       print("    Removed by OKF:")
       for source in removed:
           print(f"      - {source}")
   else:
       print("    Removed by OKF: None")

   # --------------------------------------------------------
   # Output
   # --------------------------------------------------------
   print()
   print("OKF seeds:")
   for seed in seeds:
       print(
           f"  {seed['title']} "
           f"(score={seed['score']:.4f})"
       )
   print()
   print(
       "OKF expanded concepts:",
       len(expanded)
   )
   print(
       "OKF graph coverage:",
       len(candidate_indices),
       "/",
       len(corpus)
   )
   if len(corpus) > 0:
       reduction = (
           1
           -
           len(candidate_indices)
           / len(corpus)
       )
       print(
           f"Semantic reranking pool: "
           f"{reduction * 100:.1f}%"
       )
   # --------------------------------------------------------
   # Baseline results
   # --------------------------------------------------------
   print()
   print("BASELINE TOP-3:")
   for rank, result in enumerate(
       baseline_top3,
       start=1,
   ):
       print(
           f"#{rank} "
           f"score={result['semantic_score']:.4f} "
           f"source={result['source']} "
           f"chunk={result['chunk_id']}"
       )
   print(
       "Baseline Hit@1:",
       baseline_eval["hit1"]
   )
   print(
       "Baseline Hit@3:",
       baseline_eval["hit3"]
   )
   print(
       "Baseline Evidence Recall@3:",
       f"{baseline_eval['evidence_recall']:.3f}"
   )
   # --------------------------------------------------------
   # OKF results
   # --------------------------------------------------------
   print()
   print("OKF-AWARE TOP-3:")
   for rank, result in enumerate(
       okf_top3,
       start=1,
   ):
       print(
           f"#{rank} "
           f"combined={result['combined_score']:.4f} "
           f"semantic={result['semantic_score']:.4f} "
           f"okf={result['okf_score']:.2f} "
           f"source={result['source']} "
           f"chunk={result['chunk_id']}"
       )
   print(
       "OKF Hit@1:",
       okf_eval["hit1"]
   )
   print(
       "OKF Hit@3:",
       okf_eval["hit3"]
   )
   print(
       "OKF Evidence Recall@3:",
       f"{okf_eval['evidence_recall']:.3f}"
   )

# ============================================================
# Summary
# ============================================================
baseline_hit1 = mean_metric(
   baseline_metrics,
   "hit1"
)
baseline_hit3 = mean_metric(
   baseline_metrics,
   "hit3"
)
baseline_recall = mean_metric(
   baseline_metrics,
   "evidence_recall"
)
okf_hit1 = mean_metric(
   okf_metrics,
   "hit1"
)
okf_hit3 = mean_metric(
   okf_metrics,
   "hit3"
)
okf_recall = mean_metric(
   okf_metrics,
   "evidence_recall"
)

print()
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(
   f"Questions: {len(questions)}"
)
print()
print(
   "                    BASELINE       OKF-AWARE"
)
print("-" * 55)
print(
   f"Hit@1               "
   f"{baseline_hit1:.3f}          "
   f"{okf_hit1:.3f}"
)
print(
   f"Hit@3               "
   f"{baseline_hit3:.3f}          "
   f"{okf_hit3:.3f}"
)
print(
   f"Evidence Recall@3   "
   f"{baseline_recall:.3f}          "
   f"{okf_recall:.3f}"
)


# ------------------------------------------------------------
# Per-question results
# ------------------------------------------------------------
# ------------------------------------------------------------
# Per-question results + OKF impact
# ------------------------------------------------------------
print()
print("PER-QUESTION RESULTS")
print("=" * 100)
print(
   f"{'Question':<10}"
   f"{'Base H@1':>10}"
   f"{'OKF H@1':>10}"
   f"{'Base H@3':>10}"
   f"{'OKF H@3':>10}"
   f"{'Base R@3':>10}"
   f"{'OKF R@3':>10}"
   f"{'Δ H@3':>10}"
   f"{'Δ R@3':>10}"
   f"{'Impact':>15}"
)
print("-" * 100)
improved = 0
unchanged = 0
regressed = 0
for question, b, o in zip(questions, baseline_metrics, okf_metrics):
   base_h1 = b["hit1"]
   okf_h1 = o["hit1"]
   base_h3 = b["hit3"]
   okf_h3 = o["hit3"]
   base_r3 = b["evidence_recall"]
   okf_r3 = o["evidence_recall"]
   delta_base_r3 = o["hit3"] - b["hit3"]
   delta_okf_r3 = o["evidence_recall"] - b["evidence_recall"]
   # Overall impact is based on Hit@3 and Evidence Recall@3
   if okf_h3 > base_h3 or okf_r3 > base_r3:
       impact = "IMPROVED"
       improved += 1
   elif okf_h3 < base_h3 or okf_r3 < base_r3:
       impact = "REGRESSED"
       regressed += 1
   else:
       impact = "NO CHANGE"
       unchanged += 1
   print(
       f"{question['id']:<10}"
       f"{base_h1:>10.3f}"
       f"{okf_h1:>10.3f}"
       f"{base_h3:>10.3f}"
       f"{okf_h3:>10.3f}"
       f"{base_r3:>10.3f}"
       f"{okf_r3:>10.3f}"
       f"{delta_base_r3:>10.3f}"
       f"{delta_okf_r3:>10.3f}"
       f"{impact:>15}"
   )
print("-" * 100)
print()
print("OKF IMPACT SUMMARY")
print("-" * 40)
print(f"Improved   : {improved}")
print(f"No change  : {unchanged}")
print(f"Regressed  : {regressed}")
print(f"Total      : {len(baseline_metrics)}")

# ------------------------------------------------------------
# Changed results
# ------------------------------------------------------------
print()
print("=" * 70)
print("CHANGED RESULTS")
print("=" * 70)
improved = 0
no_change = 0
regressed = 0
for result in question_results:
   h3_delta = float(result["okf_hit3"]) - float(result["baseline_hit3"])
   recall_delta = (
       float(result["okf_recall"])
       - float(result["baseline_recall"])
   )
   # Only show questions where something changed
   if h3_delta == 0 and recall_delta == 0:
       no_change += 1
       continue
   baseline_sources = set(result["baseline_sources"])
   okf_sources = set(result["okf_sources"])
   required_sources = set(result["required_evidence"])
   added = okf_sources - baseline_sources
   removed = baseline_sources - okf_sources
   added_required = added.intersection(required_sources)
   print()
   print(result["id"])
   # Determine overall impact
   if h3_delta > 0 or recall_delta > 0:
       impact = "IMPROVED"
       improved += 1
   elif h3_delta < 0 or recall_delta < 0:
       impact = "REGRESSED"
       regressed += 1
   else:
       impact = "NO CHANGE"
   # Interpretation
   if h3_delta > 0:
       interpretation = (
           "OKF brought the required evidence into the Top-3, "
           "improving retrieval completeness."
       )
   elif recall_delta > 0 and added_required:
       interpretation = (
           "OKF surfaced additional required evidence in the Top-3, "
           "increasing evidence coverage."
       )
   elif recall_delta > 0:
       interpretation = (
           "OKF increased evidence coverage by surfacing additional "
           "relevant evidence in the Top-3."
       )
   elif h3_delta < 0 or recall_delta < 0:
       interpretation = (
           "OKF reduced retrieval performance for this question."
       )
   else:
       interpretation = (
           "OKF changed the ranking but did not change retrieval metrics."
       )
   print(f"  Impact: {impact}")
   print(f"  Interpretation: {interpretation}")
   if added:
       print("  Sources added by OKF:")
       for source in sorted(added):
           if source in required_sources:
               print(f"    + {source} [REQUIRED]")
           else:
               print(f"    + {source}")
   if removed:
       print("  Sources removed by OKF:")
       for source in sorted(removed):
           print(f"    - {source}")
print()
print("=" * 70)
print("OKF IMPACT SUMMARY")
print("=" * 70)
print(f"Improved   : {improved}")
print(f"No change  : {no_change}")
print(f"Regressed  : {regressed}")
print(f"Total      : {len(question_results)}")

print()
print("=" * 80)
print("OKF NO-CHANGE ANALYSIS")
print("=" * 80)
for result in question_results:
   base_hit3 = bool(result["baseline_hit3"])
   okf_hit3 = bool(result["okf_hit3"])
   base_recall = float(result["baseline_recall"])
   okf_recall = float(result["okf_recall"])
   if (
       base_hit3 == okf_hit3
       and base_recall == okf_recall
   ):
       print()
       print(result["id"])
       print(
           f"  Interpretation: "
           f"{result['no_change_interpretation']}"
       )

# ============================================================
# Category-level analysis
# ============================================================
category_results = defaultdict(list)
for result in question_results:
   category = result.get("category", "unknown")
   category_results[category].append(result)
print()
print("=" * 80)
print("CATEGORY IMPACT SUMMARY")
print("=" * 80)
print(
   f"{'Category':<25}"
   f"{'Questions':>10}"
   f"{'Improved':>10}"
   f"{'No Change':>12}"
   f"{'Regressed':>12}"
)
for category in sorted(category_results):
   results = category_results[category]
   improved = sum(
       1 for r in results
       if (
           r["okf_hit3"] > r["baseline_hit3"]
           or r["okf_recall"] > r["baseline_recall"]
       )
   )
   regressed = sum(
       1 for r in results
       if (
           r["okf_hit3"] < r["baseline_hit3"]
           or r["okf_recall"] < r["baseline_recall"]
       )
   )
   no_change = len(results) - improved - regressed
   print(
       f"{category:<25}"
       f"{len(results):>10}"
       f"{improved:>10}"
       f"{no_change:>12}"
       f"{regressed:>12}"
   )
print()
print("=" * 80)
print("CATEGORY METRICS")
print("=" * 80)
print(
   f"{'Category':<25}"
   f"{'Base H@3':>10}"
   f"{'OKF H@3':>10}"
   f"{'Δ H@3':>10}"
   f"{'Base ER@3':>12}"
   f"{'OKF ER@3':>12}"
   f"{'Δ ER@3':>12}"
)
for category in sorted(category_results):
   results = category_results[category]
   base_h3 = sum(r["baseline_hit3"] for r in results) / len(results)
   okf_h3 = sum(r["okf_hit3"] for r in results) / len(results)
   base_recall = sum(r["baseline_recall"] for r in results) / len(results)
   okf_recall = sum(r["okf_recall"] for r in results) / len(results)
   print(
       f"{category:<25}"
       f"{base_h3:>10.3f}"
       f"{okf_h3:>10.3f}"
       f"{okf_h3 - base_h3:>10.3f}"
       f"{base_recall:>12.3f}"
       f"{okf_recall:>12.3f}"
       f"{okf_recall - base_recall:>12.3f}"
   )

print()
print("=" * 70)
print("END")
print("=" * 70)
