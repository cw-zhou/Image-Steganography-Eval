"""Run with the project's Python: .venv\Scripts\python.exe check_gpu.py."""

import sys


def main():
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    try:
        import numpy as np
        import scipy
        import torch
        import torchvision
    except (ImportError, OSError) as error:
        print("FAIL: a dependency is missing or could not be loaded:", error)
        print("See LOCAL_SETUP.md and use the Python executable inside .venv.")
        return 1

    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    if torch.__version__.split("+")[0] != "2.1.0":
        print("FAIL: this coursework targets PyTorch 2.1.0.")
        return 1
    if torchvision.__version__.split("+")[0] != "0.16.0":
        print("FAIL: torchvision 0.16.0 is required for this PyTorch version.")
        return 1
    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable to this Python environment.")
        print("Check the PyTorch CUDA build and NVIDIA driver.")
        return 1

    try:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        print("GPU:", properties.name)
        print("VRAM: {:.2f} GiB".format(properties.total_memory / 1024 ** 3))

        # Verify actual GPU computation and both directions of the NumPy bridge.
        source = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
        expected = source @ source.T
        tensor = torch.from_numpy(source).to(device)
        with torch.inference_mode():
            actual = (tensor @ tensor.T).cpu().numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            half = tensor.half()
            actual_half = (half @ half.T).float().cpu().numpy()
            np.testing.assert_allclose(actual_half, expected, rtol=1e-3, atol=1e-3)
        torch.cuda.synchronize()
        print("PASS: FP32 and FP16 matrix multiplication on CUDA.")
        print("PASS: NumPy -> GPU tensor -> NumPy conversion.")
    except (RuntimeError, AssertionError) as error:
        print("FAIL: GPU computation did not pass:", error)
        return 1

    print("GPU environment is ready. Model generation and peak VRAM are not tested here.")
    print("Local development uses this Python; final validation still requires Ubuntu/Python 3.8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
