from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.kg_contract import scan_cleaned_contact_semester_coverage


def main() -> None:
    root = Path("extracted_data_clean") / "fit"
    stats = scan_cleaned_contact_semester_coverage(root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
