from pathlib import Path
import json
import re

DOC_DIR = Path("rag_baseline/documents")
OUTPUT = Path("research/dataset/baseline_chunks.json")
MAX_WORDS = 80
OVERLAP_WORDS = 15

def normalize_text(text):
   return re.sub(r"\s+", " ", text).strip()

def words(text):
   return text.split()

def chunk_paragraph(paragraph):
   paragraph_words = words(paragraph)
   if len(paragraph_words) <= MAX_WORDS:
       return [paragraph]
   chunks = []
   start = 0
   while start < len(paragraph_words):
       end = min(start + MAX_WORDS, len(paragraph_words))
       chunk = " ".join(paragraph_words[start:end])
       chunks.append(chunk)
       if end == len(paragraph_words):
           break
       start = end - OVERLAP_WORDS
   return chunks

def create_chunks():
   chunks = []
   chunk_number = 1
   for path in sorted(DOC_DIR.rglob("*.md")):
       if path.name == "index.md":
           continue
       text = path.read_text(encoding="utf-8")
       # Remove Markdown heading lines but retain the actual content.
       text = re.sub(r"(?m)^\s*#+\s+.*$", "", text)
       # Split remaining content into paragraphs.
       paragraphs = re.split(r"\n\s*\n", text)
       for paragraph in paragraphs:
           paragraph = normalize_text(paragraph)
           if not paragraph:
               continue
           for chunk_text in chunk_paragraph(paragraph):
               chunks.append({
                   "chunk_id": f"C0-{chunk_number:03d}",
                   "source": str(path.relative_to(DOC_DIR)),
                   "source_path": str(path),
                   "text": chunk_text,
                   "word_count": len(words(chunk_text))
               })
               chunk_number += 1
   return chunks

def main():
   chunks = create_chunks()
   OUTPUT.parent.mkdir(parents=True, exist_ok=True)
   OUTPUT.write_text(
       json.dumps(chunks, indent=2),
       encoding="utf-8"
   )
   print("Baseline Chunking")
   print("=================")
   print(f"Documents : {len([p for p in DOC_DIR.rglob('*.md') if p.name != 'index.md'])}")
   print(f"Chunks    : {len(chunks)}")
   print(f"Output    : {OUTPUT}")
   print("\nChunk inventory:")
   for chunk in chunks:
       print(
           f"{chunk['chunk_id']} | "
           f"{chunk['source']} | "
           f"{chunk['word_count']} words"
       )

if __name__ == "__main__":
   main()
