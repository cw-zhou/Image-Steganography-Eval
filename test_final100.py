import io
import json
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path
from PIL import Image
from core.baseline_reference import BaselineReferenceCore
from ecc_payload_candidate import ECCPayloadCore
from run_header_ablation import RecordingBackend
from run_final100 import run_jobs,collect,case_dir,read_case,archive_partial,commit_case,load_plan


class EchoBackend:
    def __init__(self):self.generations=0
    def generate(self,noise,context):
        self.generations+=1;self.noise=noise.copy()
        return Image.new('RGB',(512,512),'gray')
    def invert(self,image,context):return self.noise.copy()


class FinalRunTests(unittest.TestCase):
    def test_resume_skips_completed_methods_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);backend=EchoBackend()
            # Author mapping imports torch but its NumPy-only ours_mapping path does not use it.
            with patch.dict(sys.modules,{'torch':types.ModuleType('torch')}):
                baseline=BaselineReferenceCore(RecordingBackend(backend))
            ecc=ECCPayloadCore.from_backend(RecordingBackend(backend),device='cpu')
            cores={'baseline':baseline,'payload_ecc':ecc}
            plan=dict(prompts=['test one','test two'],seed_start=20261101)
            with redirect_stdout(io.StringIO()):
                run_jobs(plan,out,cores,lambda image,a:image,lambda image:3.5,lambda **kw:80.,lambda seed:None,1)
            first=(case_dir(out,0,'payload_ecc')/'result.json').read_bytes()
            self.assertEqual(backend.generations,2)
            self.assertEqual(len(collect(out)),2)
            summary=json.loads((out/'summary.json').read_text())
            self.assertFalse(summary['complete_100_pairs'])
            self.assertIsNone(summary['methods']['payload_ecc']['provisional_score'])
            with redirect_stdout(io.StringIO()):
                run_jobs(plan,out,cores,lambda image,a:image,lambda image:3.5,lambda **kw:80.,lambda seed:None,1)
            self.assertEqual(backend.generations,4)
            self.assertEqual(len(collect(out)),4)
            self.assertEqual(first,(case_dir(out,0,'payload_ecc')/'result.json').read_bytes())
            rows=collect(out)
            self.assertTrue(all(r['attacks']['clean']['exact_match'] for r in rows))

    def test_completed_artifact_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            commit_case(d,dict(case_index=0,method='baseline'))
            (d/'result.json').write_text('{}')
            with self.assertRaises(ValueError):read_case(d)

    def test_partial_case_is_archived_without_deleting_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'case_000_payload_ecc';d.mkdir()
            (d/'evidence.txt').write_text('interrupted')
            self.assertIsNone(read_case(d))
            archive_partial(d)
            self.assertFalse(d.exists())
            found=list(Path(tmp).glob('incomplete_*/evidence.txt'))
            self.assertEqual(len(found),1)
            self.assertEqual(found[0].read_text(),'interrupted')

    def test_failed_completed_case_is_still_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            commit_case(d,dict(case_index=0,method='payload_ecc',generation_error='failed'))
            self.assertEqual(read_case(d)['generation_error'],'failed')

    def test_changed_prompt_file_fails_before_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'data/prompts';p.mkdir(parents=True)
            (p/'final100_coco_frozen.txt').write_text('changed\n')
            (p/'final100_coco_manifest.json').write_text('{}')
            with self.assertRaises(ValueError):load_plan(root)


if __name__=='__main__':unittest.main()
