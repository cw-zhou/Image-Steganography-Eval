"""无模型测试：验证接口衔接和图像转换，不代表真实隐写已跑通。"""

import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from core.diffusion_backend import StableDiffusionBackend, array_to_image, image_to_array
from core.mapping import SecretMapping
from core.steganography import SteganographyCore


class CoreInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.backend_patch = patch("core.steganography.StableDiffusionBackend")
        self.factory = self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.backend = self.factory.return_value
        self.backend.generate.return_value = Image.new("RGB", (512, 512), "gray")
        self.core = SteganographyCore("unused.ckpt", "unused_clip", "unused.yaml", device="cpu")

    def test_embed_passes_message_as_noise_and_returns_pil(self):
        image = self.core.embed("01011000", "a photo of a cat")
        noise, context = self.backend.generate.call_args.args
        self.assertEqual(SecretMapping(packet_version=2).decode(noise), "01011000")
        self.assertEqual(context["prompt"], "a photo of a cat")
        self.assertIs(image, self.backend.generate.return_value)
        self.assertEqual(set(self.core.get_context()), set(SteganographyCore.DEFAULT_CONTEXT) | {"prompt"})

    def test_extract_reads_backend_output_instead_of_original_message(self):
        image = self.core.embed("11111111", "a cat")
        self.backend.invert.return_value = SecretMapping(packet_version=2).encode("001000")
        self.assertEqual(self.core.extract(image), "001000")
        self.assertEqual(self.backend.invert.call_args.args[1]["prompt"], "a cat")

    def test_context_survives_json_and_can_be_used_by_fresh_instance(self):
        image = self.core.embed("0100", "a cat")
        context = json.loads(json.dumps(self.core.get_context()))
        self.backend.invert.return_value = SecretMapping(packet_version=2).encode("0100")
        fresh = SteganographyCore("unused.ckpt", "unused_clip", "unused.yaml", device="cpu")
        self.assertEqual(fresh.extract(image, context=context), "0100")
        with self.assertRaises(RuntimeError):
            fresh.get_context()

    def test_explicit_context_keeps_older_image_prompt(self):
        image = self.core.embed("01", "a cat")
        cat_context = self.core.get_context()
        self.core.embed("10", "a dog")
        self.backend.invert.return_value = SecretMapping(packet_version=2).encode("01")
        self.core.extract(image, context=cat_context)
        self.assertEqual(self.backend.invert.call_args.args[1]["prompt"], "a cat")
        self.assertEqual(self.core.get_context()["prompt"], "a dog")

    def test_context_is_a_copy_and_failed_embed_does_not_replace_it(self):
        self.core.embed("01", "a cat")
        context = self.core.get_context()
        context["prompt"] = "changed"
        self.assertEqual(self.core.get_context()["prompt"], "a cat")
        self.backend.generate.side_effect = RuntimeError("model failed")
        with self.assertRaises(RuntimeError):
            self.core.embed("10", "a dog")
        self.assertEqual(self.core.get_context()["prompt"], "a cat")

    def test_invalid_input_fails_before_model_calls(self):
        for bits, prompt in (("012", "a cat"), ("", "a cat"), ("01", " "), ("1" * 16353, "a cat")):
            with self.subTest(bits=bits[:5], prompt=prompt):
                with self.assertRaises(ValueError):
                    self.core.embed(bits, prompt)
        self.backend.generate.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.core.extract(Image.new("RGB", (512, 512)))
        with self.assertRaises(ValueError):
            self.core.extract(Image.new("RGB", (256, 256)))

    def test_invalid_context_and_invalid_backend_output_fail(self):
        self.core.embed("01", "a cat")
        for key, value in (("width", 256), ("seed_kernel", True), ("guidance_scale", float("nan")),
                           ("generation_steps", 1), ("secret_bits", "01")):
            context = self.core.get_context()
            context[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.core.extract(Image.new("RGB", (512, 512)), context=context)
        self.backend.invert.assert_not_called()
        self.backend.generate.return_value = Image.new("RGB", (256, 256))
        with self.assertRaises(RuntimeError):
            self.core.embed("01", "a dog")


    def test_legacy_context_still_decodes_old_images(self):
        context = dict(self.core.DEFAULT_CONTEXT, prompt="legacy", version=1)
        self.backend.invert.return_value = SecretMapping(packet_version=1).encode("00100")
        image = Image.new("RGB", (512, 512))
        self.assertEqual(self.core.extract(image, context=context), "00100")

    def test_unsupported_version_is_rejected(self):
        context = dict(self.core.DEFAULT_CONTEXT, prompt="test", version=4)
        with self.assertRaises(ValueError):
            self.core.extract(Image.new("RGB", (512, 512)), context=context)
        self.backend.invert.assert_not_called()


class ImageAndLoaderTests(unittest.TestCase):
    def test_real_image_normalization_and_round_trip(self):
        pixels = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        normalized = image_to_array(Image.fromarray(pixels))
        self.assertEqual(normalized.shape, (1, 3, 512, 512))
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(np.asarray(array_to_image(normalized)), pixels)

    def test_conversion_clips_and_handles_grayscale(self):
        array = np.full((1, 3, 512, 512), 2, dtype=np.float32)
        self.assertEqual(array_to_image(array).getpixel((0, 0)), (255, 255, 255))
        self.assertTrue(np.all(image_to_array(Image.new("L", (512, 512), 0)) == -1))
        with self.assertRaises(ValueError):
            array_to_image(np.full((1, 3, 512, 512), np.nan))

    def test_missing_weights_fail_without_torch_or_network(self):
        missing = Path(__file__).parent / "weights" / "__test_missing_model__.ckpt"
        with self.assertRaises(FileNotFoundError):
            StableDiffusionBackend(missing, "unused_clip", "unused.yaml")


class ArrayTensor:
    """仅在测试中代替张量的数据容器；不执行扩散模型计算。"""

    def __init__(self, array):
        self.array = array

    def to(self, device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class BackendFlowTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(StableDiffusionBackend)
        self.backend.torch = SimpleNamespace(
            no_grad=nullcontext, from_numpy=Mock(side_effect=ArrayTensor)
        )
        self.backend.device = SimpleNamespace(type="cpu")
        self.backend.model = Mock()
        self.backend.sampler = Mock()
        self.context = dict(SteganographyCore.DEFAULT_CONTEXT, prompt="a cat")
        self.context["generation_steps"] = 30
        self.context["inversion_steps"] = 40

    def test_generation_uses_forward_sampler_and_image_decoder(self):
        latent = ArrayTensor(np.zeros((1, 4, 64, 64), dtype=np.float32))
        self.backend.sampler.sample.return_value = (latent, None)
        self.backend.model.decode_first_stage.return_value = ArrayTensor(
            np.zeros((1, 3, 512, 512), dtype=np.float32)
        )
        noise = np.ones((1, 4, 64, 64), dtype=np.float32)
        image = self.backend.generate(noise, self.context)
        arguments = self.backend.sampler.sample.call_args.kwargs
        self.assertEqual(arguments["steps"], 30)
        self.assertFalse(arguments["DPMencode"])
        self.assertTrue(arguments["DPMdecode"])
        self.assertEqual(arguments["shape"], (4, 64, 64))
        np.testing.assert_array_equal(arguments["x_T"].array, noise)
        self.backend.model.decode_first_stage.assert_called_once_with(latent)
        self.assertEqual(image.getpixel((0, 0)), (128, 128, 128))

    def test_inversion_encodes_image_before_reverse_sampler(self):
        latent = ArrayTensor(np.zeros((1, 4, 64, 64), dtype=np.float32))
        noise = np.ones((1, 4, 64, 64), dtype=np.float32)
        self.backend.model.get_first_stage_encoding.return_value = latent
        self.backend.sampler.sample.return_value = (ArrayTensor(noise), None)
        recovered = self.backend.invert(Image.new("RGB", (512, 512), "black"), self.context)
        arguments = self.backend.sampler.sample.call_args.kwargs
        self.assertEqual(arguments["steps"], 40)
        self.assertTrue(arguments["DPMencode"])
        self.assertIs(arguments["x_T"], latent)
        self.backend.model.get_first_stage_encoding.assert_called_once_with(
            self.backend.model.encode_first_stage.return_value
        )
        encoded_pixels = self.backend.model.encode_first_stage.call_args.args[0].array
        self.assertEqual(encoded_pixels.shape, (1, 3, 512, 512))
        self.assertTrue(np.all(encoded_pixels == -1))
        np.testing.assert_array_equal(recovered, noise)


if __name__ == "__main__":
    unittest.main(verbosity=2)
