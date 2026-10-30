import unittest,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
@unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV required')
class MaskTest(unittest.TestCase):
    def test_masks_local_mutable_area_without_discarding_static_outline(self):
        import numpy as np
        from fukuwarai_registration_masks import registration_masks
        art=np.zeros((100,100,3),np.uint8);im=np.full((100,100,3),200,np.uint8)
        a,b=registration_masks(art,im)
        self.assertEqual(a[60,50],0);self.assertEqual(a[60,10],255);self.assertEqual(a[20,50],255)
        self.assertTrue(np.all(b==255))
    def test_red_piece_and_descriptor_margin_excluded_dark_hair_kept(self):
        import numpy as np
        from fukuwarai_registration_masks import registration_masks
        im=np.full((100,100,3),200,np.uint8);im[40:45,40:45]=[80,80,200];im[10:20,10:20]=[29,37,52]
        a,b=registration_masks(im,im)
        self.assertEqual(b[42,42],0);self.assertEqual(b[37,42],0);self.assertEqual(b[15,15],255)
        self.assertEqual(b[80,80],255)
