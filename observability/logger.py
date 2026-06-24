import json
from sys import exception
import time
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR/"pipeline.json1"

#unique session id for each program run
SESSION_ID = str(uuid.uuid4())[:8]

def _write_log(event : dict) -> None:
    event["session_id"] = SESSION_ID
    event["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE,"a") as f:
        f.write(json.dumps(event)+'\n')

@contextmanager
def log_stage(stage : str, **context):
    start = time.perf_counter()
    log_data = {"stage" : stage, **context}

    try:
        yield log_data
        log_data["status"] = "success"
    except Exception as e:
        log_data["status"] = "error"
        log_data["error"] = str(e)
        raise
    finally:
        log_data["latency_ms"] = round((time.perf_counter() - start)*1000,2)
        _write_log(log_data)