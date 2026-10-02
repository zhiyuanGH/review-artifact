"""Original frozen runner plus connection guard outside measured workflows. No retries."""
import inspect,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'mobile_tools/scripts'))
import run_selected10_supplement as base
source=inspect.getsource(base.execute)
needle='                for job in batch:\n                    prefix='
replacement="                for job in batch:\n                    identity=shell('getprop ro.serialno').stdout.strip()\n                    if identity != device: raise RuntimeError('Connection identity mismatch before next workflow')\n                    prefix="
assert source.count(needle)==1
source=source.replace(needle,replacement)
exec(compile(source,__file__,'exec'),base.__dict__)
if __name__=='__main__':base.main()
