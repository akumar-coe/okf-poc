import json
import re
import faiss
import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer
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
# Evaluation
# ============================================================
"""
def evaluate(
   results,
   question
):
   required = {
       item.replace("\\", "/")
       for item in question.get(
           "required_evidence",
           []
       )
   }
   retrieved = [
       result["source"].replace("\\", "/")
       for result in results
   ]
   if not required:
       return {
           "hit1": False,
           "hit3": False,
           "evidence_recall": 0.0,
       }
   retrieved_set = set(
       retrieved
   )
   # Hit@1 = at least one required
   # evidence source at rank 1.
   hit1 = (
       retrieved[0]
       in required
   )
   # Hit@3 = ALL required evidence
   # appears in top 3.
   hit3 = required.issubset(
       set(retrieved[:FINAL_K])
   )
   # Evidence recall tells us how much
   # of the required evidence was retrieved.
   evidence_recall = (
       len(
           required.intersection(
               set(retrieved[:FINAL_K])
           )
       )
       / len(required)
   )
   return {
       "hit1": hit1,
       "hit3": hit3,
       "evidence_recall": evidence_recall,
   }
"""
# ============================================================
# Run experiment
# ============================================================
print()
print("=" * 70)
print("OKF-AWARE RETRIEVAL EXPERIMENT")
print("=" * 70)
baseline_metrics = []
okf_metrics = []

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
       enriched = dict(result)
       enriched["okf_score"] = structural
       enriched["combined_score"] = combined
       okf_results.append(
           enriched
       )
   
   assert len(okf_results) == len(baseline_results)

   okf_results.sort(
       key=lambda item:
           item["combined_score"],
       reverse=True,
   )
   okf_top3 = okf_results[
       :FINAL_K
   ]
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

"""
# ============================================================
# Summary
# ============================================================
def mean_metric(
   metrics,
   field
):
   values = []
   for metric in metrics:
       value = metric[field]
       if isinstance(
           value,
           bool
       ):
           value = (
               1.0
               if value
               else 0.0
           )
       values.append(
           float(value)
       )
   return (
       sum(values)
       /
       len(values)
   )
"""
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
print()
print("=" * 70)
print("END")
print("=" * 70)
