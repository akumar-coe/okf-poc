from pathlib import Path
import re
import sys
import yaml

BUNDLE = Path("knowledge/telecom")

def parse_frontmatter(path):
   text = path.read_text(encoding="utf-8")
   if not text.startswith("---"):
       raise ValueError("Missing YAML frontmatter")
   match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
   if not match:
       raise ValueError("Invalid YAML frontmatter")
   metadata = yaml.safe_load(match.group(1))
   if not isinstance(metadata, dict):
       raise ValueError("Frontmatter must be a YAML mapping")
   return metadata, text[match.end():]

def find_links(body):
   return re.findall(r"\[[^\]]+\]\(([^)]+)\)", body)

def validate_bundle():
   concepts = []
   concept_metadata = []
   errors = []
   links = []
   for path in BUNDLE.rglob("*.md"):
       # index.md is an index, not a concept
       if path.name == "index.md":
           continue
       try:
           metadata, body = parse_frontmatter(path)
       except Exception as exc:
           errors.append(f"{path}: {exc}")
           continue
       if "type" not in metadata:
           errors.append(f"{path}: missing required 'type'")
       concepts.append(path)
       concept_metadata.append({
           "path": path,
           "title": metadata.get("title", path.stem),
           "type": metadata["type"],
           "tags": metadata.get("tags", []),
           "generated": metadata.get("generated"),
           "verified": metadata.get("verified"),
           "status": metadata.get("status", "stable"),
           "stale_after": metadata.get("stale_after"),
       })
       for link in find_links(body):
           # Ignore external URLs
           if link.startswith(("http://", "https://")):
               continue
           target = (path.parent / link).resolve()
           if not target.exists():
               errors.append(
                   f"{path}: broken link -> {link}"
               )
           else:
               links.append((path, target))
   print("OKF Bundle Validation")
   print("=====================")
   print(f"Concepts:      {len(concepts)}")
   print(f"Relationships: {len(links)}")
   print(f"Errors:        {len(errors)}")
   if errors:
       print("\nValidation errors:")
       for error in errors:
           print(f"  - {error}")
       return 1
   print("\nConcept Inventory:")
   print("------------------")
   for concept in concept_metadata:
       verified = concept["verified"]
       if not verified:
           trust = "unverified"
       elif isinstance(verified, dict):
           actor = verified.get("by", "")
           trust = (
               "human-reviewed"
               if actor.startswith("human:")
               else "machine-confirmed"
           )
       else:
           actor = verified[-1].get("by", "")
           trust = (
               "human-reviewed"
               if actor.startswith("human:")
               else "machine-confirmed"
           )
       print(
           f"  {concept['title']:<30} "
           f"{concept['type']:<20} "
           f"{trust:<18} "
           f"status={concept['status']:<10} "
           f"stale_after={concept['stale_after']}"
       )
   print("\nRelationships:")
   for source, target in links:
       print(f"  {source} -> {target}")
   print("\nStatus: PASS")
   return 0

if __name__ == "__main__":
   sys.exit(validate_bundle())
