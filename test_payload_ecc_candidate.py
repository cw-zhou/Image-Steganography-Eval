import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
import numpy as np
from PIL import Image
from core.randomized_header import RandomizedHeaderCore,RandomizedHeaderMapping
from ecc_payload_candidate import ECCPayloadMapping,ECCPayloadCore
from run_payload_ecc_pilot import run_new,load_controls


class MappingTests(unittest.TestCase):
    def test_variable_lengths_trailing_zeros_and_fresh_decoder(self):
        for n in (1,31,1024,8192):
            rng=random.Random(n)
            bits=''.join(str(rng.randrange(2)) for _ in range(n-1))+'0'
            noise=ECCPayloadMapping(padding_seed=123).encode(bits)
            self.assertEqual(ECCPayloadMapping().decode(noise),bits)

    def test_only_filler_changes_before_orthogonal_transform(self):
        bits='0101'*2048
        base=RandomizedHeaderMapping(padding_seed=2).bits_to_tensor(bits).ravel()
        ecc=ECCPayloadMapping(padding_seed=2).bits_to_tensor(bits).ravel()
        np.testing.assert_array_equal(base[:1984+8192],ecc[:1984+8192])
        self.assertTrue(np.any(base[1984+8192:]!=ecc[1984+8192:]))

    def test_soft_errors_corrected_without_mutating_input(self):
        mapping=ECCPayloadMapping(padding_seed=4)
        bits='01'*4096
        tensor=mapping.bits_to_tensor(bits)
        flat=tensor.ravel()
        for pos in (1984,1984+25,1984+555):
            flat[pos]=.49 if flat[pos]>.5 else .51
        original=tensor.copy()
        self.assertEqual(mapping.tensor_to_bits(tensor),bits)
        np.testing.assert_array_equal(tensor,original)
        self.assertEqual(mapping.last_diagnostics['changed_payload_bits'],3)

    def test_invalid_payload_or_header_fails(self):
        mapping=ECCPayloadMapping()
        with self.assertRaises(ValueError):mapping.encode('0'*8193)
        tensor=mapping.bits_to_tensor('0101')
        tensor.ravel()[:1984]=mapping._header_mask
        with self.assertRaises(ValueError):mapping.tensor_to_bits(tensor)
        self.assertIsNone(mapping.last_diagnostics)

    def test_context_json_and_old_core_rejection(self):
        context=json.loads(json.dumps(dict(ECCPayloadCore.DEFAULT_CONTEXT,prompt='test')))
        self.assertEqual(ECCPayloadCore._validate_context(context),context)
        with self.assertRaises(ValueError):RandomizedHeaderCore._validate_context(context)
        context['payload_code']='unknown'
        with self.assertRaises(ValueError):ECCPayloadCore._validate_context(context)


class EchoBackend:
    def generate(self,noise,context):
        self.noise=noise.copy()
        return Image.new('RGB',(512,512),'gray')
    def invert(self,image,context):
        return self.noise.copy()


class RunnerTests(unittest.TestCase):
    def test_new_case_uses_saved_controls_and_writes_diagnostics(self):
        rng=random.Random(74000)
        bits=''.join(str(rng.randrange(2)) for _ in range(8192))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source';case=source/'case_00_header_randomized'
            case.mkdir(parents=True)
            (case/'context.json').write_text(json.dumps(dict(RandomizedHeaderCore.DEFAULT_CONTEXT,prompt='test')))
            controls=[]
            for method in ('baseline','header_randomized'):
                controls.append(dict(method=method,seed=74000,prompt='test',generation_error=None,niqe=3.5,
                                     bpp=.0625 if method=='baseline' else .03125,payload_sha256=hashlib.sha256(bits.encode()).hexdigest(),
                                     source_run=str(source),attacks={a:dict(acc=1.,decoded=True,length_match=True) for a in ('clean','jpeg','awgn','mblur')}))
            out=root/'out';out.mkdir()
            core=ECCPayloadCore.from_backend(EchoBackend(),device='cpu')
            with redirect_stdout(io.StringIO()):
                status=run_new({'prompts':['test']},controls,core,lambda image,a:image,lambda image:3.5,
                               lambda **kwargs:80.,out,lambda seed:None)
            self.assertEqual(status,0)
            rows=json.loads((out/'results.json').read_text())
            self.assertEqual(len(rows),3)
            candidate=rows[-1]
            self.assertEqual(candidate['payload_bits'],8192)
            self.assertEqual(candidate['bpp'],.03125)
            self.assertTrue(all(d['converged_blocks']==8 for d in candidate['payload_decoder_diagnostics']))
            self.assertTrue((out/'case_00_payload_ecc/awgn_raw_recovered.txt').is_file())
            self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
