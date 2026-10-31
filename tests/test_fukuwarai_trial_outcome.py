import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_trial_outcome import countable_model_stop

class TrialOutcome(unittest.TestCase):
    def test_tracking_timeout_is_failure_not_infrastructure(self):
        self.assertTrue(countable_model_stop({'reason':'Endpoint tracking timeout','recording_audit':{'quality_pass':True}}))
    def test_recording_and_infrastructure_faults_remain_blocking(self):
        for r in [{'reason':'Endpoint tracking timeout'}, {'reason':'Fresh camera frame unavailable','recording_audit':{'quality_pass':True}}, {'reason':'Endpoint tracking timeout','recording_audit':{'quality_pass':False}}]:
            self.assertFalse(countable_model_stop(r))
