"""Persistent, nonblocking retraining jobs for the HTTP API."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'training_job.json'
LOCK = threading.Lock()


def save_state(state):
    temporary = STATE.with_name(f'.training-job-{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
    os.replace(temporary,STATE)


def get_status():
    if not STATE.exists():
        return {'status':'idle'}
    state = json.loads(STATE.read_text(encoding='utf-8'))
    stale = False
    if state['status']=='running' and state.get('pid'):
        stale = not process_exists(state['pid'])
    elif state['status']=='queued':
        stale = (datetime.now(timezone.utc)-datetime.fromisoformat(state['started_at'])).total_seconds()>60
    if stale:
        latest=json.loads(STATE.read_text(encoding='utf-8'))
        if latest.get('job_id')==state.get('job_id') and latest['status'] in ('queued','running'):
            latest.update(status='failed',error='The training worker stopped before recording a result.',
                          message='Training was interrupted. You can start a new job.')
            save_state(latest)
        return latest
    return state


def process_exists(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            return ctypes.get_last_error()==5
        code=wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def start_job():
    with LOCK:
        previous = get_status()
        if previous['status'] in ('queued','running'):
            return previous
        if not (ROOT/'data/training_50000/metadata.json').is_file():
            raise ValueError('The 50,000-clip corpus is missing. Generate data/training_50000 before retraining.')
        metadata=json.loads((ROOT/'data/training_50000/metadata.json').read_text())
        if metadata.get('target_count')!=50000:
            raise ValueError('Expected exactly 50,000 training clips; regenerate the training corpus.')
        job_id = uuid.uuid4().hex
        state = {'job_id':job_id,'status':'queued','started_at':datetime.now(timezone.utc).isoformat(),
                 'samples':50000,'message':'Training queued on the 50,000-clip corpus.'}
        save_state(state)
        try:
            log_dir=ROOT/'.logs'
            log_dir.mkdir(exist_ok=True)
            with (log_dir/'training-job.log').open('w',encoding='utf-8') as log:
                subprocess.Popen([sys.executable,str(ROOT/'training_worker.py'),job_id],cwd=ROOT,
                                 stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except Exception as error:
            state.update(status='failed',error=str(error),message='Could not start training.')
            save_state(state)
            raise
        return state
