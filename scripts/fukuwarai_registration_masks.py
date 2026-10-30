"""Static-artwork/local moving-lip masks for image-only face registration."""
import cv2
import numpy as np


def registration_masks(artwork, image):
    h,w=artwork.shape[:2]
    artwork_mask=np.full((h,w),255,np.uint8)
    # Exclude movable lip/guide/nose neighborhood, retain static cheeks/jaw/hair.
    artwork_mask[int(h*.52):int(h*.72),int(w*.30):int(w*.70)]=0
    hue,saturation,value=cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
    red=(((hue<10)|(hue>170))&(saturation>85)&(value>100)).astype(np.uint8)*255
    image_mask=np.full(image.shape[:2],255,np.uint8)
    image_mask[cv2.dilate(red,np.ones((11,11),np.uint8))>0]=0
    return artwork_mask,image_mask
