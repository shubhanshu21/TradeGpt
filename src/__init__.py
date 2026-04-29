import sys
import importlib

# Legacy Redirect: Ensures old .pkl files (saved when preprocess was at top-level) still load.
try:
    from data import preprocess as _preprocess
    sys.modules['preprocess'] = _preprocess
except Exception:
    pass
