import unittest
import numpy as np
from core.mapping import SecretMapping
from diagnose_attack import recover_symbols, compare_reference


class DiagnosticTests(unittest.TestCase):
    def test_inverse_matches_original_decoder(self):
        mapping = SecretMapping(packet_version=2)
        noise = mapping.encode('101000')
        symbols = recover_symbols(mapping, noise)
        self.assertEqual(mapping.tensor_to_bits(symbols), mapping.decode(noise))
        stats = compare_reference(mapping, symbols, '101000')
        self.assertEqual(stats['raw_header_errors'], 0)
        self.assertEqual(stats['diagnostic_payload_errors'], 0)

    def test_separates_header_failure_from_payload_errors(self):
        mapping = SecretMapping(packet_version=2)
        symbols = mapping.bits_to_tensor('01010000')
        flat = symbols.reshape(-1)
        header = flat[:mapping.header_bits].reshape(9, 64)
        header[:5, 31] = 1 - header[:5, 31]
        flat[mapping.header_bits] = 1
        with self.assertRaises(ValueError):
            mapping.tensor_to_bits(symbols)
        stats = compare_reference(mapping, symbols, '01010000')
        self.assertEqual(stats['majority_length_errors'], 1)
        self.assertEqual(stats['majority_crc_errors'], 0)
        self.assertEqual(stats['diagnostic_payload_errors'], 1)
        self.assertEqual(stats['diagnostic_payload_acc_known_length'], 7 / 8)
        self.assertEqual(stats['raw_header_errors'], 5)

    def test_wrong_reference_does_not_change_decoding(self):
        mapping = SecretMapping(packet_version=2)
        symbols = mapping.bits_to_tensor('0000')
        stats = compare_reference(mapping, symbols, '1111')
        self.assertEqual(stats['diagnostic_payload_acc_known_length'], 0)
        self.assertEqual(mapping.tensor_to_bits(symbols), '0000')


if __name__ == '__main__':
    unittest.main()
