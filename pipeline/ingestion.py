import json
import gzip
from pathlib import Path
from typing import Dict, Iterator, Any

def iter_reviews(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def count_reviews(path: str | Path) -> int:
    return sum(1 for _ in iter_reviews(path))
