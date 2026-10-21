import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from app.ml import training_jobs as jobs
import training_worker


class TrainingJobTests(unittest.TestCase):
    def test_http_start_returns_202_and_status_without_waiting_for_training(self):
        from fastapi.testclient import TestClient
        from app.main import app
        job={'job_id':'test-job','status':'queued','samples':50000}
        client=TestClient(app)
        with patch('app.api.routes.start_job',return_value=job) as start,patch('app.api.routes.get_status',return_value=job):
            response=client.post('/api/train')
            self.assertEqual(response.status_code,202)
            self.assertEqual(response.json()['job_id'],'test-job')
            self.assertEqual(client.get('/api/train/status').json()['status'],'queued')
            start.assert_called_once()

    def test_http_start_preserves_actionable_error(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with patch('app.api.routes.start_job',side_effect=ValueError('50,000-clip corpus is missing')):
            response=TestClient(app).post('/api/train')
        self.assertEqual(response.status_code,400)
        self.assertIn('corpus is missing',response.json()['detail'])

    def test_start_returns_queued_and_duplicate_does_not_launch_another_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            data=root/'data/training_50000'
            data.mkdir(parents=True)
            (data/'metadata.json').write_text(json.dumps({'target_count':50000}))
            with patch.object(jobs,'ROOT',root),patch.object(jobs,'STATE',root/'job.json'),patch.object(jobs.subprocess,'Popen') as launch:
                first=jobs.start_job()
                second=jobs.start_job()
                self.assertEqual(first['status'],'queued')
                self.assertEqual(first['job_id'],second['job_id'])
                launch.assert_called_once()

    def test_missing_training_data_returns_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(jobs,'ROOT',root),patch.object(jobs,'STATE',root/'job.json'):
                with self.assertRaisesRegex(ValueError,'corpus is missing'):
                    jobs.start_job()

    def test_dead_worker_can_be_recovered_after_web_service_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            state=Path(temporary)/'job.json'
            state.write_text(json.dumps({'job_id':'old','status':'running','pid':999999}))
            with patch.object(jobs,'STATE',state),patch.object(jobs,'process_exists',return_value=False):
                self.assertEqual(jobs.get_status()['status'],'failed')
                self.assertIn('interrupted',jobs.get_status()['message'])

    def test_worker_reports_real_training_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'.logs').mkdir()
            (root/'.logs/training-job.log').write_text('Training failed: disk full')
            snapshots=[]
            with patch.object(training_worker,'ROOT',root),patch.object(training_worker.sys,'argv',['worker','job']),patch.object(training_worker,'get_status',return_value={'job_id':'job','status':'queued'}),patch.object(training_worker,'save_state',side_effect=lambda state:snapshots.append(dict(state))),patch.object(training_worker.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'train')):
                training_worker.main()
            self.assertEqual(snapshots[-1]['status'],'failed')
            self.assertIn('disk full',snapshots[-1]['error'])

    def test_success_distinguishes_retained_model_from_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'training_50000_evaluation.json').write_text(json.dumps({'promoted':False}))
            snapshots=[]
            with patch.object(training_worker,'ROOT',root),patch.object(training_worker.sys,'argv',['worker','job']),patch.object(training_worker,'get_status',return_value={'job_id':'job','status':'queued'}),patch.object(training_worker,'save_state',side_effect=lambda state:snapshots.append(dict(state))),patch.object(training_worker.subprocess,'run'):
                training_worker.main()
            self.assertEqual(snapshots[-1]['status'],'completed')
            self.assertFalse(snapshots[-1]['promoted'])
            self.assertIn('retained',snapshots[-1]['message'])
