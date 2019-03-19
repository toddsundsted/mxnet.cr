# Deep Learning for Crystal

[MXNet](https://mxnet.incubator.apache.org/) is a library for deep
learning, written in C++. It provides bindings for many popular
languages (Python, Scala, R, etc.). [MXNet.cr](https://github.com/toddsundsted/mxnet.cr)
provides bindings for [Crystal](https://crystal-lang.org/).

MXNet.cr follows the design of the Python bindings, albeit with
Crystal syntax. The following code:

```crystal
require "mxnet"
a = MXNet::NDArray.array([[1, 2], [3, 4]])
b = MXNet::NDArray.array([1, 0])
puts a * b
```

outputs:

```
[[1, 0], [3, 0]]
<NDArray 2x2 int32 cpu(0)>
```

# Installation

MXNet.cr requires MXNet.

Build MXNet from source (including Python language bindings) or
install the library from prebuilt packages using the Python package
manager *pip*, per the MXNet installation instructions:

https://mxnet.incubator.apache.org/install/index.html

And add the following to your application's *shard.yml*:

```yaml
dependencies:
  mxnet:
    github: toddsundsted/mxnet.cr
```

# Troubleshooting

MXNet.cr relies on the Python library to find the installed MXNet
shared library ("libmxnet.so"). You can verify MXNet is installed with
the following Python code:

```python
import mxnet as mx
a = mx.ndarray.array([[1, 2], [3, 4]])
b = mx.ndarray.array([1, 0])
print(a * b)
```

which outputs:

```
[[1. 0.]
 [3. 0.]]
<NDArray 2x2 @cpu(0)>
```

On OSX, you may need to give the compiled Crystal executable a hint
about the location of the shared library. If you see an error message
like the following:

```
dyld: Library not loaded: lib/libmxnet.so
  Referenced from: /Users/homedirectory/.cache/crystal/crystal-run-eval.tmp
  Reason: image not found
```

you need to either: 1) manually set the `DYLD_FALLBACK_LIBRARY_PATH`
environment variable to point to the directory containing the shared
library, or 2) move or copy the library into a well-known location.

The Crystal build step does not currently set the path to the library
explicitly and OSX does not currently look for shared libraries in
non-standard locations without guidance.
