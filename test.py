python - <<'PY'
from pathlib import Path
import json

p = Path(".claude/settings.local.json")
p.parent.mkdir(parents=True, exist_ok=True)

data = {}
if p.exists():
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}

data.setdefault("permissions", {})["defaultMode"] = "bypassPermissions"

p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"updated: {p}")
PY
