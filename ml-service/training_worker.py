"""Train outside the web server and persist the outcome for status polling."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from app.ml.training_jobs import ROOT, get_status, save_state


def main():
    state=get_status()
    if state.get('job_id')!=sys.argv[1]:
        raise RuntimeError('Training job identifier changed.')
    state.update(status='running',pid=os.getpid(),message='Training on 50,000 audio clips. This can take several minutes.')
    save_state(state)
    try:
        subprocess.run([sys.executable,'train_augmented.py','--corpus-dir','data/training_50000',
                        '--models-dir','saved_models_50000','--epochs','40','--batch-size','128'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'promote_model.py','--source','saved_models_50000',
                        '--report-file','training_50000_evaluation.json'],cwd=ROOT,check=True)
        report=json.loads((ROOT/'training_50000_evaluation.json').read_text())
        state.update(status='completed',promoted=report['promoted'],
                     message='Training complete. New model deployed.' if report['promoted'] else
                             'Training complete. The existing model performed better on validation and was retained.')
    except Exception as error:
        log=ROOT/'.logs/training-job.log'
        tail='\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-12:]) if log.exists() else ''
        state.update(status='failed',error=f'{error}\n{tail}',message='Training failed. See the details below.')
    state['finished_at']=datetime.now(timezone.utc).isoformat()
    save_state(state)


if __name__=='__main__':
    main()
