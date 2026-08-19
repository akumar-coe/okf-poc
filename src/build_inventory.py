from pathlib import Path
from datetime import date, datetime
import json
import re
import yaml

BUNDLE = Path("knowledge/telecom").resolve()
OUTPUT = Path("reports/knowledge_inventory.json")

def parse_frontmatter(path):
   text = path.read_text(encoding="utf-8")
   if not text.startswith("---"):
       raise ValueError(f"{path}: missing YAML frontmatter")
   match = re.match(
       r"^---\s*\n(.*?)\n---\s*\n",
       text,
       re.DOTALL,
   )
   if not match:
       raise ValueError(f"{path}: invalid YAML frontmatter")
   metadata = yaml.safe_load(match.group(1))
   if not isinstance(metadata, dict):
       raise ValueError(f"{path}: frontmatter must be a YAML mapping")
   body = text[match.end():]
   return metadata, body

def find_links(body):
   return re.findall(r"\[[^\]]+\]\(([^)]+)\)", body)

def json_safe(value):
   if isinstance(value, (datetime, date)):
       return value.isoformat()
   if isinstance(value, dict):
       return {
           key: json_safe(item)
           for key, item in value.items()
       }
   if isinstance(value, list):
       return [json_safe(item) for item in value]
   return value

def build_inventory():
   concepts = []
   relationships = []
   for path in sorted(BUNDLE.rglob("*.md")):
       if path.name == "index.md":
           continue
       metadata, body = parse_frontmatter(path)
       concept_id = str(path.relative_to(BUNDLE))
       concepts.append({
           "id": concept_id,
           "title": metadata.get("title", path.stem),
           "type": metadata.get("type"),
           "description": metadata.get("description"),
           "tags": metadata.get("tags", []),
           "domain": metadata.get("domain"),
           "lifecycle": metadata.get("lifecycle"),
           "status": metadata.get("status"),
           "generated": metadata.get("generated"),
           "verified": metadata.get("verified"),
           "stale_after": metadata.get("stale_after"),
           "sources": metadata.get("sources", []),
       })
       for link in find_links(body):
           if link.startswith(("http://", "https://")):
               continue
           target = (path.parent / link).resolve()
           if target.exists() and target.suffix == ".md":
               target_id = str(target.relative_to(BUNDLE))
               relationships.append({
                   "source": concept_id,
                   "target": target_id,
               })
   inventory = {
       "bundle": "telecom",
       "concept_count": len(concepts),
       "relationship_count": len(relationships),
       "concepts": concepts,
       "relationships": relationships,
   }
   inventory = json_safe(inventory)
   OUTPUT.parent.mkdir(parents=True, exist_ok=True)
   OUTPUT.write_text(
       json.dumps(inventory, indent=2),
       encoding="utf-8",
   )
   print(f"Inventory written to: {OUTPUT}")
   print(f"Concepts: {len(concepts)}")
   print(f"Relationships: {len(relationships)}")

if __name__ == "__main__":
   build_inventory()
