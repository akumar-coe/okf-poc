import json
import numpy as np
import onnxruntime as ort
import faiss
from transformers import AutoTokenizer

MODEL_DIR = "all-MiniLM-L6-v2"
CORPUS_PATH = "research/dataset/baseline_chunks.json"
QUESTIONS_PATH = "research/questions/evaluation_questions.yaml"

def mean_pool(token_embeddings, attention_mask):
   mask = attention_mask[..., None].astype(np.float32)
   summed = np.sum(token_embeddings * mask, axis=1)
   counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
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
           "input_ids": inputs["input_ids"].astype(np.int64),
           "attention_mask": inputs["attention_mask"].astype(np.int64),
           "token_type_ids": inputs["token_type_ids"].astype(np.int64),
       },
   )
   embeddings = mean_pool(outputs[0], inputs["attention_mask"])
   # Normalize so inner product == cosine similarity
   embeddings = embeddings / np.linalg.norm(
       embeddings, axis=1, keepdims=True
   )
   return embeddings.astype(np.float32)

# --------------------------------------------------
# Load corpus
# --------------------------------------------------
with open(CORPUS_PATH) as f:
   corpus = json.load(f)
texts = [item["text"] for item in corpus]
print("Loaded corpus:", len(corpus), "chunks")

# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------
# We deliberately use PyYAML only for the benchmark file.
import yaml
with open(QUESTIONS_PATH) as f:
   benchmark = yaml.safe_load(f)
questions = benchmark["questions"]
print("Loaded questions:", len(questions))

# --------------------------------------------------
# Load tokenizer + ONNX model
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
   MODEL_DIR,
   local_files_only=True
)
session = ort.InferenceSession(
   MODEL_DIR + "/onnx/model_quantized.onnx",
   providers=["CPUExecutionProvider"],
)
print("MiniLM model loaded")

# --------------------------------------------------
# Embed corpus
# --------------------------------------------------
corpus_embeddings = embed(
   texts,
   tokenizer,
   session
)
print("Corpus embeddings:", corpus_embeddings.shape)

# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(corpus_embeddings)
print("FAISS index created")
print("Vectors indexed:", index.ntotal)

# --------------------------------------------------
# Evaluate retrieval
# --------------------------------------------------
hit_at_1 = 0
hit_at_3 = 0
mrr_total = 0.0
print("\n==============================")
print("BASELINE RETRIEVAL RESULTS")
print("==============================")
for q in questions:
   query = q["question"]
   required = set(q.get("required_evidence", []))
   query_embedding = embed(
       [query],
       tokenizer,
       session
   )
   scores, indices = index.search(
       query_embedding,
       3
   )
   retrieved = [
       corpus[i]["source"]
       for i in indices[0]
   ]
   # ------------------------------------------------
   # Hit@1
   # ------------------------------------------------
   hit1 = retrieved[0] in required
   if hit1:
       hit_at_1 += 1
   # ------------------------------------------------
   # Hit@3
   # ------------------------------------------------
   retrieved_required = set(retrieved) & required
   hit3 = required.issubset(set(retrieved))
   if hit3:
       hit_at_3 += 1
   # ------------------------------------------------
   # MRR
   # For multi-evidence questions we use the first
   # required evidence encountered.
   # ------------------------------------------------
   reciprocal_rank = 0.0
   for rank, source in enumerate(retrieved, start=1):
       if source in required:
           reciprocal_rank = 1.0 / rank
           break
   mrr_total += reciprocal_rank
   # ------------------------------------------------
   # Print result
   # ------------------------------------------------
   print("\n", q["id"])
   print("Question:", query)
   print("Required:", list(required))
   for rank, (idx, score) in enumerate(
       zip(indices[0], scores[0]),
       start=1
   ):
       print(
           f"  #{rank} "
           f"score={score:.4f} "
           f"source={corpus[idx]['source']} "
           f"chunk={corpus[idx]['chunk_id']}"
       )
   print("Hit@1:", hit1)
   print("Hit@3:", hit3)

# --------------------------------------------------
# Summary
# --------------------------------------------------
n = len(questions)
print("\n==============================")
print("SUMMARY")
print("==============================")
print(f"Questions : {n}")
print(f"Hit@1     : {hit_at_1 / n:.3f}")
print(f"Hit@3     : {hit_at_3 / n:.3f}")
print(f"MRR       : {mrr_total / n:.3f}")
