import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from data.prepare import prepare
from data.dataset import SegmentationDataset, assert_disjoint, letterbox, load_manifest, restore
from evaluation.metrics import segmentation_metrics
from losses import SegmentationLoss
from models import ThyroidSegmentation
from models.dsconv import DSConv
from models.context_refinement import LocalCrossAttention


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_snake_zero_offset_matches_straight_convolution(self):
        for axis in ('x','y'):
            op=DSConv(2,8,5,axis)
            x=torch.randn(2,2,7,11)
            sampled=F.grid_sample(x,op.sampling_grid(x),padding_mode='border',align_corners=False)
            got=op.aggregate(sampled)
            weight=op.aggregate.weight
            if axis=='x':
                expected=F.conv2d(F.pad(x,(2,2,0,0),mode='replicate'),weight)
            else:
                expected=F.conv2d(F.pad(x,(0,0,2,2),mode='replicate'),weight.transpose(-1,-2))
            torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-5)

    def test_snake_offsets_are_continuous_and_differentiable(self):
        op=DSConv(2,8,5,'x')
        with torch.no_grad(): op.offset.bias.fill_(.2)
        x=torch.randn(1,2,8,9,requires_grad=True)
        offset=op.cumulative_offsets(x)
        self.assertTrue(torch.equal(offset[:,2],torch.zeros_like(offset[:,2])))
        self.assertTrue(torch.all(offset[:,0]>offset[:,1]))
        self.assertTrue(torch.all(offset[:,4]>offset[:,3]))
        op(x).square().mean().backward()
        for index in (0,1,3,4):
            self.assertGreater(float(op.offset.weight.grad[index].abs().sum()),0.)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_local_attention_odd_dimensions_and_gradient(self):
        op=LocalCrossAttention(8,heads=2,window=9)
        x=torch.randn(1,8,7,13,requires_grad=True)
        out=op(x,x,torch.rand(1,3,7,13))
        self.assertEqual(out.shape,x.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(op.role_embed.weight.grad.abs().sum()),0.)

    def test_all_ablations_forward_backward_and_missing_foreground(self):
        for mode in ('none','cnn','snake','context'):
            for global_att in (False,True):
                model=ThyroidSegmentation(base=8,heads=2,refinement=mode,global_attention=global_att,window=3,snake_kernel=5)
                x=torch.rand(2,1,64,96)
                target=torch.zeros(2,1,64,96);target[0,0,15:40,25:55]=1
                output=model(x)
                self.assertTrue(all(o.shape==target.shape for o in output.values()))
                loss,_=SegmentationLoss()(output,target,torch.ones_like(target))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(float(model.heads.segmentation.weight.grad.abs().sum()),0.)

    def test_geometry_and_metrics(self):
        image=Image.fromarray(np.full((101,161),127,dtype=np.uint8))
        mask=np.zeros((101,161),np.uint8);mask[30:70,50:90]=255
        _,p,valid,g=letterbox(image,Image.fromarray(mask),64)
        restored=restore(np.asarray(p),g,mask=True)>0
        result=segmentation_metrics(restored,mask>0)
        self.assertGreater(result['dice'],.95)
        self.assertEqual(restore(np.asarray(p),g).shape,mask.shape)
        same=segmentation_metrics(mask,mask)
        self.assertEqual(same['hd95_px'],0)
        self.assertEqual(same['surface_dice'],1.)
        missing=segmentation_metrics(np.zeros_like(mask),mask)
        self.assertIsNone(missing['hd95_px'])
        self.assertEqual(missing['dice'],0.)

    def test_patient_split_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'data'
            for patient in range(12):
                for view in range(2):
                    rel=Path('centre_A')/'images'/f'P{patient}'/f'{patient}_{view}.png'
                    path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                    # patient 0 and 1 deliberately share exact pixels.
                    pixels=np.full((40,64),1+max(0,patient-1)*2+view,dtype=np.uint8)
                    Image.fromarray(pixels).save(path)
                    mask=root/Path(str(rel).replace('images','masks'))
                    mask.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.zeros_like(pixels)).save(mask)
            out=Path(d)/'split'
            audit=prepare(root,out)
            manifests=[load_manifest(out/f'{s}.jsonl') for s in ('train','val','test')]
            assert_disjoint(*manifests)
            for rows in manifests:
                patients={r['patient_id'] for r in rows}
                self.assertEqual('centre_A/P0' in patients,'centre_A/P1' in patients)
            sample=SegmentationDataset(root,out/'train.jsonl',64,True)[0]
            self.assertEqual(sample['image'].shape,(1,64,64))
            self.assertTrue(set(torch.unique(sample['mask']).tolist())<={0.,1.})


if __name__=='__main__': unittest.main()
