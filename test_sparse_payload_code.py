import unittest
import numpy as np
from sparse_payload_code import SparsePayloadCode


class CodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code=SparsePayloadCode()
    def test_capacity_and_parity(self):
        msg=np.random.default_rng(1).integers(0,2,(8,1024),dtype=np.uint8)
        word=self.code.encode(msg)
        self.assertEqual(word.size,14400)
        self.assertEqual(word.size-msg.size,6208)
        np.testing.assert_array_equal(word[:,:1024],msg)
        self.assertFalse(self.code.syndrome(word).any())
    def test_exact_roundtrip_extreme_and_random_messages(self):
        msg=np.random.default_rng(2).integers(0,2,(3,1024),dtype=np.uint8)
        msg[0]=0;msg[1]=1;msg[2,-15:]=0
        word=self.code.encode(msg)
        decoded,meta=self.code.decode(20*(1-2*word.astype(float)))
        np.testing.assert_array_equal(decoded,msg)
        self.assertTrue(meta['converged'].all())
    def test_low_confidence_errors_are_corrected(self):
        msg=np.random.default_rng(3).integers(0,2,(2,1024),dtype=np.uint8)
        llr=15*(1-2*self.code.encode(msg).astype(float))
        llr[:,[0,25,555]] *= -.01
        decoded,meta=self.code.decode(llr)
        self.assertTrue(meta['converged'].all())
        np.testing.assert_array_equal(decoded,msg)
    def test_zero_budget_uses_raw_fallback_without_reference(self):
        msg=np.zeros((1,1024),dtype=np.uint8)
        llr=np.full((1,1800),10.)
        llr[0,0]=-10
        decoded,meta=self.code.decode(llr,max_iterations=0)
        self.assertTrue(meta['raw_fallback'][0])
        self.assertEqual(decoded[0,0],1)
    def test_input_validation_and_local_rng(self):
        np.random.seed(123);before=np.random.get_state()
        other=SparsePayloadCode()
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        np.testing.assert_array_equal(other.a_checks,self.code.a_checks)
        with self.assertRaises(ValueError):self.code.encode(np.full((1,1024),2))
        with self.assertRaises(ValueError):self.code.decode(np.full((1,1800),np.nan))


if __name__=='__main__':unittest.main()
