import sys
from data import preprocess
# Legacy Redirect: Ensures old .pkl files (saved when preprocess was in root) still load.
sys.modules['preprocess'] = preprocess
