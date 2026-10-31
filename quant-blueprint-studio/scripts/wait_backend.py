"""Wait for the engine, failing if the port belongs to another process."""
import sys
import time
import urllib.request
import json

pid = int(sys.argv[1])
for _ in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=1) as response:
            if json.load(response)['pid'] != pid:
                raise RuntimeError('Port 8000 belongs to another process.')
        break
    except (OSError, ValueError):
        time.sleep(0.5)
else:
    raise SystemExit('Backend did not start within 60 seconds. Check the console.')
