import io
import unittest

import numpy as np
from PIL import Image

from data.dataset import letterbox, restore
from progress import Progress


class ProgressInputTests(unittest.TestCase):
    def test_native_detail_is_unchanged(self):
        pixels = np.random.default_rng(7).integers(0,256,(73,125),dtype=np.uint8)
        mask = (pixels>127).astype(np.uint8)*255
        canvas,target,valid,g = letterbox(Image.fromarray(pixels),Image.fromarray(mask),
            size=256,upscale=False,resize_filter='lanczos')
        crop = np.asarray(canvas)[g['top']:g['top']+73,g['left']:g['left']+125]
        np.testing.assert_array_equal(crop,pixels)
        np.testing.assert_array_equal(restore(np.asarray(target),g,mask=True),mask)
        self.assertEqual(valid.sum(),73*125)
        self.assertEqual((g['width'],g['height']),(125,73))

    def test_downscale_preserves_aspect_and_binary_mask(self):
        image = Image.fromarray(np.full((150,300),100,np.uint8))
        mask = Image.fromarray(np.pad(np.ones((50,100),np.uint8)*255,((50,50),(100,100))))
        canvas,target,valid,g = letterbox(image,mask,size=128,upscale=False,resize_filter='lanczos')
        self.assertEqual((g['width'],g['height']),(128,64))
        self.assertTrue(set(np.unique(target)) <= {0,255})
        self.assertEqual(restore(np.asarray(canvas),g).shape,(150,300))

    def test_legacy_default_upscales(self):
        _,_,_,g = letterbox(Image.new('L',(64,32)),size=128)
        self.assertEqual((g['width'],g['height']),(128,64))

    def test_progress_logs_loss_and_completes(self):
        stream = io.StringIO()
        with Progress(3,'Train 1/1',every=2,stream=stream) as bar:
            for i in range(3): bar.update(loss=f'{1/(i+1):.3f}',lr='3e-4')
        output = stream.getvalue()
        self.assertIn('3/3 100.0%',output)
        self.assertIn('loss=0.333',output)
        self.assertIn('ETA=',output)
        self.assertEqual(len(output.splitlines()),3)

    def test_disabled_progress_is_silent(self):
        stream = io.StringIO()
        with Progress(1,'Val',enabled=False,stream=stream) as bar: bar.update(dice='1')
        self.assertEqual(stream.getvalue(),'')


if __name__ == '__main__': unittest.main()
