from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.embed_into_neo4j import build_graph_snapshot, get_template_outputs


def main() -> None:
    clean_root = Path("extracted_data_clean") / "fit"
    snapshot = build_graph_snapshot(clean_root)
    outputs = get_template_outputs(snapshot)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
