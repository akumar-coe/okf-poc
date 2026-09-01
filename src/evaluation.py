def evaluate(results, question, k=3):
   """
   Common evaluation metrics for all retrieval experiments.
   required_evidence is used ONLY for evaluation,
   never for retrieval.
   """
   required = {
       item.replace("\\", "/")
       for item in question.get("required_evidence", [])
   }
   retrieved = [
       result["source"].replace("\\", "/")
       for result in results[:k]
   ]
   if not required:
       return {
           "hit1": False,
           "hit3": False,
           "evidence_recall": 0.0,
           "mrr": 0.0,
       }
   retrieved_top_k = set(retrieved)
   # At least one required evidence source at rank 1.
   hit1 = retrieved[0] in required
   # ALL required evidence sources must occur in Top-K.
   hit_k = required.issubset(retrieved_top_k)
   # Fraction of required evidence sources retrieved.
   evidence_recall = (
       len(required.intersection(retrieved_top_k))
       / len(required)
   )
   # Reciprocal rank of the first required evidence source.
   mrr = 0.0
   for rank, source in enumerate(retrieved, start=1):
       if source in required:
           mrr = 1.0 / rank
           break
   return {
       "hit1": hit1,
       "hit3": hit_k,
       "evidence_recall": evidence_recall,
       "mrr": mrr,
   }

def mean_metric(metrics, field):
   """
   Calculate the mean of a metric across questions.
   """
   values = []
   for metric in metrics:
       value = metric[field]
       if isinstance(value, bool):
           value = 1.0 if value else 0.0
       values.append(float(value))
   if not values:
       return 0.0
   return sum(values) / len(values)
