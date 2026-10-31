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

    def test_short_model_stop_counts_as_failure_without_relaxing_quality(self):
        from fukuwarai_trial_outcome import short_model_failure
        r={'reason':'Endpoint tracking timeout','success':False,'recording_audit':{'quality_pass':False,'frames_written':54,'frames_received':54,'frames_rejected':0,'fps_requested':10,'maximum_frame_interval_sim_s':.1,'maximum_nearest_state_offset_ms':0.}}
        self.assertTrue(short_model_failure(r));self.assertTrue(countable_model_stop(r))
        self.assertFalse(r['recording_audit']['quality_pass'])
        r['recording_audit']['frames_rejected']=1
        self.assertFalse(short_model_failure(r))
