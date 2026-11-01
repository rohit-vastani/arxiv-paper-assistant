import json
from pathlib import Path
from paper_agent.config import ResourcePath


PATHS = ResourcePath()
CATALOG = Path(PATHS.MAIN_DIR) / "catalog.json"

def _load() -> dict:
    return json.loads(CATALOG.read_text()) if CATALOG.exists() else {}

def _save(catalog: dict) -> None:
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2))
    tmp.replace(CATALOG)