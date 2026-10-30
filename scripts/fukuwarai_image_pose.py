"""Image-only silhouette registration against the authored mouth contour."""
import math
import cv2
import numpy as np
from fukuwarai_success import load_contour


class MouthImagePose:
    def __init__(self, config):
        camera=config['observation_camera']
        focal=camera['width']/(2*math.tan(camera['horizontal_fov_rad']/2))
        self.scale=focal/(camera['pose_m_rad'][2]-config['board_surface_z_m']-config['mouth_size_m'][2])
        self.xy=load_contour()[:,:2]

    def estimate(self, image, mask):
        # Fit only to observed red pixels. No simulator pose or target labels.
        ys,xs=np.nonzero(mask)
        if len(xs)<300:raise RuntimeError('Insufficient visible mouth pixels for registration')
        x0,x1=max(0,int(xs.min())-14),min(image.shape[1],int(xs.max())+15)
        y0,y1=max(0,int(ys.min())-14),min(image.shape[0],int(ys.max())+15)
        roi=(mask[y0:y1,x0:x1]>0).astype(np.float32)
        best=None
        def fit(deg):
            angle=math.radians(deg);c,s=math.cos(angle),math.sin(angle)
            xy=self.xy @ np.array([[c,s],[-s,c]])
            uv=np.c_[-xy[:,1],-xy[:,0]]*self.scale
            lo=np.floor(uv.min(axis=0)).astype(int)-1
            hi=np.ceil(uv.max(axis=0)).astype(int)+1
            template=np.zeros((hi[1]-lo[1]+1,hi[0]-lo[0]+1),np.float32)
            cv2.fillPoly(template,[np.rint(uv-lo).astype(np.int32)],1.)
            if min(np.array(roi.shape)-np.array(template.shape))<0:return None
            response=cv2.matchTemplate(roi,template,cv2.TM_CCORR_NORMED)
            _,value,_,location=cv2.minMaxLoc(response)
            peak=np.array(location,dtype=float)
            for axis in (0,1):
                ix,iy=location
                if axis==0 and 0<ix<response.shape[1]-1:
                    left,mid,right=map(float,response[iy,ix-1:ix+2])
                elif axis==1 and 0<iy<response.shape[0]-1:
                    left,mid,right=map(float,response[iy-1:iy+2,ix])
                else:continue
                denominator=left-2*mid+right
                if denominator < -1e-9:
                    peak[axis]+=float(np.clip(.5*(left-right)/denominator,-.5,.5))
            return value, peak+[x0,y0]-lo, float(deg)
        for deg in range(-180,180,5):
            r=fit(deg)
            if r is not None and (best is None or r[0]>best[0]):best=r
        if best is None:raise RuntimeError('Mouth silhouette did not fit image ROI')
        for deg in np.arange(best[2]-5,best[2]+5.01,.5):
            r=fit(float(deg))
            if r is not None and r[0]>best[0]:best=r
        if best[0]<.7:raise RuntimeError('Mouth silhouette registration confidence too low')
        return {'pixel':best[1].tolist(),'yaw_rad':math.radians((best[2]+180)%360-180),'confidence':best[0]}


def rear_contact_distance(footprint, direction, offset):
    """Distance behind an offset force line origin to its polygon intersection."""
    polygon=cv2.convexHull(np.asarray(footprint,dtype=np.float32)).reshape(-1,2).astype(float)
    direction=np.asarray(direction,dtype=float)
    normal=np.array([-direction[1],direction[0]])
    offset=np.asarray(offset,dtype=float)
    lateral=float(offset@normal)
    along=polygon@direction; across=polygon@normal
    intersections=[]
    for i in range(len(polygon)):
        j=(i+1)%len(polygon)
        if abs(across[j]-across[i])<1e-12:continue
        fraction=(lateral-across[i])/(across[j]-across[i])
        if 0 <= fraction <= 1:
            intersections.append(float(along[i]+fraction*(along[j]-along[i])))
    if not intersections:raise RuntimeError('Offset push line misses mouth collision footprint')
    return float(offset@direction)-min(intersections)
