import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("command")

args = parser.parse_args()

def explain():
    msg = '''NO MXNET LIBRARY FOUND

MXNet.cr requires the MXNet (https://mxnet.incubator.apache.org/) deep
learning library be installed. On most platforms, you can either build
the MXNet library from source or you can install the library, along
with Python language bindings, with a tool like "pip". In any case, no
MXNet library can be found.

See the MXNet.cr README (https://github.com/toddsundsted/mxnet.cr) for
more guidance on installing MXNet for your platform.

---

'''
    sys.stderr.write(msg)

if sys.version_info.major >= 3:
    from importlib import util
    spec = util.find_spec('mxnet')
    if spec is None:
        explain()
        sys.exit(1)
    mxnet_dir = os.path.dirname(spec.origin)
else:
    import imp
    try:
        spec = imp.find_module('mxnet')
        mxnet_dir = spec[1]
    except ImportError:
        explain()
        sys.exit(1)

if args.command == "version":
    import mxnet
    print(mxnet.__version__)

if args.command == "library":
    curr_path = mxnet_dir
    api_path = os.path.join(curr_path, '../../lib/')
    build_path = os.path.join(curr_path, '../../build/Release/')
    dll_path = [curr_path, api_path, build_path]

    if os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(':')])
    elif os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['DYLD_FALLBACK_LIBRARY_PATH'].split(':')])

    dll_path.append('../../../')
    dll_path = [os.path.join(p, 'libmxnet.so') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) > 0:
        print(lib_path[0])
