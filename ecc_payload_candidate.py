"""Opt-in payload ECC candidate. Existing baseline and cores are unchanged."""
import numpy as np
from PIL import Image
from core.randomized_header import RandomizedHeaderMapping, RandomizedHeaderCore
from sparse_payload_code import SparsePayloadCode

CODE_ID = 'ra8_k1024_p776_seed20261031_nms08_i60_v1'


class ECCPayloadMapping(RandomizedHeaderMapping):
    def __init__(self, **kwargs):
        kwargs.setdefault('bit_length',8192)
        super().__init__(**kwargs)
        if self.latent_shape!=(1,4,64,64) or self.max_payload_bits>8192:
            raise ValueError('ECC candidate supports default shape and 1..8192 payload bits')
        self.code=SparsePayloadCode(seed=20261031)
        self.last_diagnostics=None

    def bits_to_tensor(self, secret_bits):
        tensor=super().bits_to_tensor(secret_bits)
        flat=tensor.reshape(-1)
        # Short messages retain random systematic padding; actual length stays in the protected header.
        systematic=flat[self.header_bits:self.header_bits+8192].astype(np.uint8).reshape(8,1024)
        word=self.code.encode(systematic)
        flat[self.header_bits+8192:]=word[:,1024:].reshape(-1)
        return tensor

    def tensor_to_bits(self, tensor, threshold=.5):
        self.last_diagnostics=None
        if threshold!=.5:
            raise ValueError('ECC candidate uses fixed threshold .5')
        # Existing header unmasking, mean decision, CRC and capacity checks run first.
        raw=super().tensor_to_bits(tensor,threshold=threshold)
        flat=self._as_array(tensor).reshape(-1)
        llr=1-2*flat[self.header_bits:]
        llr=np.concatenate((llr[:8192].reshape(8,1024),llr[8192:].reshape(8,776)),axis=1)
        decoded,meta=self.code.decode(llr,max_iterations=60,normalization=.8)
        result=''.join(str(int(x)) for x in decoded.reshape(-1)[:len(raw)])
        self.last_diagnostics=dict(raw_recovered_bits=raw,
                                   converged_blocks=int(meta['converged'].sum()),
                                   fallback_blocks=int(meta['raw_fallback'].sum()),
                                   iterations=meta['iterations'].tolist(),
                                   changed_payload_bits=sum(a!=b for a,b in zip(raw,result)))
        return result


class ECCPayloadCore(RandomizedHeaderCore):
    DEFAULT_CONTEXT=dict(RandomizedHeaderCore.DEFAULT_CONTEXT,payload_code=CODE_ID)

    @classmethod
    def _validate_context(cls, context):
        result=super()._validate_context(context)
        if result['payload_code']!=CODE_ID:
            raise ValueError('Unsupported payload ECC identity')
        return result

    @staticmethod
    def _mapping(context, *, padding_seed=None):
        return ECCPayloadMapping(seed_kernel=context['seed_kernel'],seed_shuffle=context['seed_shuffle'],
                                 header_mask_seed=context['header_mask_seed'],padding_seed=padding_seed)

    def extract(self, image, *, context=None):
        if not isinstance(image,Image.Image) or image.size!=(512,512):
            raise ValueError('Expected a 512x512 PIL image')
        context=self._validate_context(self.get_context() if context is None else context)
        mapping=self._mapping(context)
        if not hasattr(self,'extraction_diagnostics'):
            self.extraction_diagnostics=[]
        try:
            noise=self._backend.invert(image.convert('RGB'),context)
            result=mapping.decode(noise)
        except Exception as exc:
            self.extraction_diagnostics.append(dict(error=str(exc)))
            raise
        self.extraction_diagnostics.append(dict(mapping.last_diagnostics,error=None))
        return result
