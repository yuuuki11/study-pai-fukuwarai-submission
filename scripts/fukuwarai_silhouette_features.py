"""Opt-in current RGB silhouette cues, with no object-state input."""
from functools import lru_cache
import json
from pathlib import Path
import cv2
import numpy as np
from fukuwarai_image_pose import MouthImagePose

@lru_cache(maxsize=1)
def estimator():
    return MouthImagePose(json.loads((Path(__file__).resolve().parents[1]/'config/fukuwarai.json').read_text()))

def silhouette_features(rgb):
    rgb=np.asarray(rgb)
    if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
        raise ValueError('Expected original640x480 uint8 RGB')
    h,s,v=cv2.split(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
    mask=(((h<=17)|(h>=168))&(s>=90)&(v>=70)).astype(np.uint8)*255
    mask[:200]=0;mask[350:]=0;mask[:,:230]=0;mask[:,410:]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    valid=[i for i in range(1,count) if 450<=stats[i,4]<=4500 and min(stats[i,2:4])>=16 and .25<=stats[i,2]/stats[i,3]<=4]
    if not valid:return np.zeros(5,dtype=np.float32)
    label=max(valid,key=lambda i:stats[i,4])
    try:pose=estimator().estimate(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(labels==label).astype(np.uint8)*255)
    except RuntimeError:return np.zeros(5,dtype=np.float32)
    if pose['confidence']<.8:return np.zeros(5,dtype=np.float32)
    x,y=pose['pixel'];angle=-pose['yaw_rad']
    # Same five dimensions as existing cues, but versioned center semantics.
    return np.array([(x-320)/100,(y-270)/100,np.sin(2*angle),np.cos(2*angle),1],dtype=np.float32)

# Store only image digests and five floats, never full training images.
_FEATURE_CACHE = {}
def cached_silhouette_features(rgb):
    import hashlib
    image=np.asarray(rgb)
    key=hashlib.sha256(image.tobytes()).digest()
    if key not in _FEATURE_CACHE:
        if len(_FEATURE_CACHE)>=60000:_FEATURE_CACHE.clear()
        _FEATURE_CACHE[key]=silhouette_features(image)
    return _FEATURE_CACHE[key].copy()
