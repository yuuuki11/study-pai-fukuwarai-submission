"""Compact only newly authorized cohort artifacts after verified video encoding."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def keep_video(placement_success, strict_success, failed_video_count, failure_limit=3):
    return bool(placement_success or strict_success or failed_video_count < failure_limit)


def compact_episode(episode, cohort, video_required, fps=10):
    episode, cohort = Path(episode).resolve(), Path(cohort).resolve()
    registration = json.loads((cohort/'pre_registration.json').read_text())
    if registration.get('retention',{}).get('delete_new_trial_images_and_rosbag_after_verification') is not True:
        raise ValueError('Missing cohort-specific retention authorization')
    if not episode.is_relative_to(cohort) or episode == cohort:
        raise ValueError('Refusing to compact outside the new cohort')
    marker = episode/'retention.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('complete'):
            return saved
    # Validate every destructive target before encoding or deleting anything.
    targets=[episode/'images',episode/'rosbag']
    for target in targets:
        if target.is_symlink() or not target.resolve().is_relative_to(episode):
            raise ValueError('Refusing symlink/outside artifact')
    frames=sorted((episode/'images').glob('*.jpg'))
    if not frames:
        raise ValueError('No final frame; preserve trial for diagnosis')
    final=episode/'final.jpg'
    shutil.copy2(frames[-1],final)
    report=dict(complete=False, video_required=video_required, original_frame_count=len(frames),
                final_image_source=frames[-1].name,
                final_image_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                deleted=['images','rosbag'], retained=['final.jpg','result.json','observations.jsonl','frames.jsonl','policy_actions.jsonl','sync_audit.json'])
    if video_required:
        video=episode/'rollout_1x.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),
                        '-start_number',str(int(frames[0].stem)),'-i',str(episode/'images/%06d.jpg'),
                        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(video)],check=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                                                '-show_entries','stream=nb_read_frames,width,height','-of','json',str(video)],text=True))['streams'][0]
        if int(info['nb_read_frames']) != len(frames):
            raise RuntimeError('Video frame count mismatch; retaining originals')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-f','null','-'],check=True)
        report.update(video='rollout_1x.mp4',video_frames=int(info['nb_read_frames']),
                      video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),video_speed='1x simulation time, nominal10Hz')
    marker.write_text(json.dumps(report,indent=2)+'\n')
    for target in targets:
        if target.exists():shutil.rmtree(target)
    report['complete']=True
    marker.write_text(json.dumps(report,indent=2)+'\n')
    return report
