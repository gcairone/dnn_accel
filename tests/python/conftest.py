import pathlib
import sys

# The built extension lands in <repo>/python/fastmlp/_fastmlp*.so (see
# bindings/python/CMakeLists.txt's LIBRARY_OUTPUT_DIRECTORY); make that
# package importable without requiring an editable/pip install first.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PYTHON_DIR = str(_REPO_ROOT / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
