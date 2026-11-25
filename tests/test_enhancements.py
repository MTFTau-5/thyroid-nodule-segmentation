import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from data.augmentation import UltrasoundAugmentation
from engine import checkpoint_model, loader, seed_everything, validate
from inference import predict_probability
from losses import SegmentationLoss, dice_bce
from models import ThyroidSegmentation
from models.context_refinement import BoundaryRefinement


class SpatialPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, image):
        self.shapes.append(image.shape[-2:])
        position = torch.linspace(-2, 2, image.shape[-1], device=image.device)[None, None, None, :]
        return {'segmentation': image*3 + position}


class EnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_multiscale_depth_and_all_gradients(self):
        for steps in (2, 3):
            model = ThyroidSegmentation(base=8, heads=2, depth=3, window=3, snake_kernel=3,
                                       multiscale_context=True, refinement_steps=steps)
            grids = []
            handle = model.mid_refine.register_forward_pre_hook(
                lambda module, inputs: grids.append(inputs[0].shape[-2:]))
            image = torch.rand(1, 1, 65, 97)
            target = (image > .5).float()
            output = model(image)
            handle.remove()
            self.assertEqual(len(model.global_context.layers.layers), 3)
            self.assertEqual(grids, [(9, 13)])
            self.assertEqual(set(output), {'segmentation', 'coarse', 'boundary', 'intermediate_coarse'})
            self.assertTrue(all(value.shape == image.shape for value in output.values()))
            loss, parts = SegmentationLoss()(output, target, torch.ones_like(target))
            expected = sum(dice_bce(output[key], target, torch.ones_like(target))
                           for key in ('coarse', 'intermediate_coarse'))/2
            torch.testing.assert_close(parts['coarse'], expected.detach())
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(model.mid_head.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.mid_refine.context.q.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.refine.x.offset.weight.grad.abs().sum().item(), 0)

    def test_refinement_feeds_previous_output_and_shares_weights(self):
        operation = BoundaryRefinement(8, mode='snake', kernel=3, steps=3)
        single = BoundaryRefinement(8, mode='snake', kernel=3)
        single.load_state_dict(operation.state_dict())
        image, coarse = torch.rand(1, 8, 8, 9), torch.rand(1, 1, 8, 9)
        expected = image
        for _ in range(3):
            expected, expected_boundary = single(expected, coarse)
        actual, boundary = operation(image, coarse)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(boundary, expected_boundary)
        self.assertEqual(sum(parameter.numel() for parameter in operation.parameters()),
                         sum(parameter.numel() for parameter in single.parameters()))

    def test_legacy_checkpoint_and_enhanced_checkpoint_roundtrip(self):
        for extra in ({}, {'depth':3, 'multiscale_context':True, 'refinement_steps':2}):
            config = {'model': {'base':8, 'heads':2, 'window':3, 'snake_kernel':3, **extra}}
            model = ThyroidSegmentation(**config['model']).eval()
            image = torch.rand(1, 1, 64, 64)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory)/'model.pt'
                torch.save({'model':model.state_dict(), 'config':config}, checkpoint)
                restored, _ = checkpoint_model(checkpoint, torch.device('cpu'))
                with torch.inference_mode():
                    torch.testing.assert_close(restored(image)['segmentation'], model(image)['segmentation'])
                if not extra:
                    self.assertIsNone(restored.mid_refine)
                    self.assertEqual(restored.refine.steps, 1)

    def test_invalid_model_settings(self):
        for kwargs in ({'depth':0}, {'refinement_steps':0}, {'refinement_steps':1.5},
                       {'multiscale_context':True, 'refinement':'none'}):
            with self.assertRaises(ValueError):
                ThyroidSegmentation(base=8, **kwargs)

    def test_full_config_enables_enhancements(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/full.json').read_text())
        self.assertEqual(config['model']['depth'], 3)
        self.assertTrue(config['model']['multiscale_context'])
        self.assertEqual(config['model']['refinement_steps'], 3)
        UltrasoundAugmentation(**config['augmentation'])

    def test_elastic_alignment_and_binary_masks(self):
        pixels = np.zeros((64, 96), np.uint8)
        pixels[16:48, 24:72] = 255
        image = Image.fromarray(pixels)
        operation = UltrasoundAugmentation(elastic_prob=1, elastic_alpha=.08,
                                          speckle_prob=0, gain_prob=0, bright_spot_prob=0)
        seed_everything(43)
        transformed, target = operation(image, image)
        self.assertEqual(transformed.size, image.size)
        self.assertTrue(set(np.unique(target)) <= {0, 255})
        self.assertGreater(np.mean((np.asarray(transformed) > 127) == (np.asarray(target) > 0)), .99)
        self.assertFalse(np.array_equal(np.asarray(target), pixels))

    def test_intensity_augmentation_reproducible_and_mask_unchanged(self):
        image = Image.fromarray(np.full((48, 64), 128, np.uint8))
        mask = Image.fromarray(np.pad(np.ones((24, 32), np.uint8)*255, ((12, 12), (16, 16))))
        operation = UltrasoundAugmentation(elastic_prob=0, speckle_prob=1, gain_prob=1, bright_spot_prob=1)
        seed_everything(11)
        first, target = operation(image, mask)
        seed_everything(11)
        second, _ = operation(image, mask)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(target, mask)
        self.assertFalse(np.array_equal(first, image))
        self.assertEqual(np.asarray(first).dtype, np.uint8)

    def test_disabled_augmentations_are_identity(self):
        pixels = np.arange(64*64, dtype=np.uint8).reshape(64, 64)
        image = Image.fromarray(pixels)
        operation = UltrasoundAugmentation(elastic_prob=0, speckle_prob=0, gain_prob=0, bright_spot_prob=0)
        transformed, target = operation(image, image)
        np.testing.assert_array_equal(transformed, pixels)
        np.testing.assert_array_equal(target, pixels)

    def test_invalid_augmentation(self):
        for kwargs in ({'elastic_prob':1.1}, {'speckle_std':-1}, {'elastic_sigma':0},
                       {'gain_range':[2, 1]}, {'attenuation_max':float('nan')}):
            with self.assertRaises(ValueError):
                UltrasoundAugmentation(**kwargs)

    def test_tta_noop_and_cached_output(self):
        model, image = SpatialPredictor(), torch.rand(2, 1, 65, 97)
        output = model(image)
        actual = predict_probability(model, image, output=output)
        torch.testing.assert_close(actual, output['segmentation'].sigmoid())
        self.assertEqual(len(model.shapes), 1)
        torch.testing.assert_close(predict_probability(model, image), actual)
        self.assertEqual(len(model.shapes), 2)

    def test_tta_inverse_flip_resize_and_probability_average(self):
        model, image = SpatialPredictor(), torch.rand(2, 1, 65, 97)
        actual = predict_probability(model, image, scales=(.5, 1., 1.5), horizontal_flip=True)
        self.assertEqual(model.shapes, [(32, 64)]*2 + [(65, 97)]*2 + [(96, 160)]*2)
        expected = []
        for shape in ((32, 64), (65, 97), (96, 160)):
            resized = F.interpolate(image, size=shape, mode='bilinear', align_corners=False)
            position = torch.linspace(-2, 2, shape[-1])[None, None, None, :]
            for probability in ((resized*3+position).sigmoid(), (resized*3+position.flip(-1)).sigmoid()):
                expected.append(F.interpolate(probability, size=image.shape[-2:], mode='bilinear', align_corners=False))
        torch.testing.assert_close(actual, torch.stack(expected).mean(0))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_invalid_tta_scales(self):
        for scales in ((), (0,), (-1,), (float('nan'),), (float('inf'),)):
            with self.assertRaises(ValueError):
                predict_probability(SpatialPredictor(), torch.rand(1, 1, 32, 32), scales)

    def test_loader_and_validation_with_tta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((40, 60), 128, np.uint8)).save(root/'image.png')
            Image.fromarray(np.zeros((40, 60), np.uint8)).save(root/'mask.png')
            row = dict(image='image.png', mask='mask.png', patient_id='patient', centre='centre')
            (root/'val.jsonl').write_text(json.dumps(row)+'\n')
            config = dict(data_root=directory, splits=directory, size=64, seed=7, batch_size=1,
                          augmentation={'elastic_prob':1})
            batches = loader(config, 'val')
            self.assertIsNone(batches.dataset.ultrasound_augmentation)
            self.assertIsNotNone(loader(config, 'val', training=True).dataset.ultrasound_augmentation)
            model = ThyroidSegmentation(base=8, heads=2, window=3, snake_kernel=3,
                                       multiscale_context=True, refinement_steps=2)
            metrics, rows = validate(model, batches, torch.device('cpu'), criterion=SegmentationLoss(),
                                     progress=False, tta_scales=(.5, 1.), tta_flip=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(metrics['tta'], {'scales':[.5, 1.], 'horizontal_flip':True})
            self.assertTrue(np.isfinite(metrics['losses']['loss']))


if __name__ == '__main__':
    unittest.main()
