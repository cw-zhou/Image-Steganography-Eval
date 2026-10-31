import json
import tempfile
import unittest
from pathlib import Path
from run_payload_ecc_batch2 import load_controls


class Batch2ControlsTests(unittest.TestCase):
    def fixture(self, root, seed):
        header=root/'header';baseline=root/'baseline'
        header.mkdir();baseline.mkdir()
        prompts=['prompt '+str(i) for i in range(5)]
        provenance=dict(model_files={},packages={},attacks={},source_sha256={})
        hm=dict(seed=seed,cases=5,prompts=prompts,provenance=provenance)
        bm=dict(seed=seed,cases=5,prompts=prompts,**provenance)
        for folder,manifest,method in [(header,hm,'header_randomized'),(baseline,bm,'baseline')]:
            (folder/'manifest.json').write_text(json.dumps(manifest))
            rows=[dict(method=method,seed=seed+i,prompt=prompts[i],generation_error=None,niqe=3.) for i in range(5)]
            (folder/'results.json').write_text(json.dumps(rows))
        return header,baseline

    def test_loads_second_batch_controls_without_reusing_first_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            header,baseline=self.fixture(root,74005)
            manifest,rows=load_controls(header,baseline,5,root)
            self.assertEqual(manifest['seed'],74005)
            self.assertEqual({r['seed'] for r in rows},set(range(74005,74010)))
            self.assertEqual(len(rows),10)
            self.assertTrue(all(r['reused_control'] for r in rows))

    def test_rejects_first_batch_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            header,baseline=self.fixture(root,74000)
            with self.assertRaises(ValueError):load_controls(header,baseline,5,root)


if __name__=='__main__':unittest.main()
