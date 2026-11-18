"""Run a shell command and record its wall-clock time in results/timings.json:  python scripts/timed.py <stage> <cmd...>"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

stage, cmd = sys.argv[1], " ".join(sys.argv[2:])
t = time.time()
rc = subprocess.call(cmd, shell=True)
p = Path("results/timings.json")
d = json.loads(p.read_text()) if p.exists() else {}
d[stage] = {"wall_seconds": round(time.time() - t, 1), "cmd": cmd, "loadavg_after": [round(x, 1) for x in os.getloadavg()]}
p.write_text(json.dumps(d, indent=1))
sys.exit(rc)
