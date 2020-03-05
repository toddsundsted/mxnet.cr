# Deep Learning for Crystal

[![GitHub Release](https://img.shields.io/github/release/toddsundsted/mxnet.cr.svg)](https://github.com/toddsundsted/mxnet.cr/releases)
[![Build Status](https://travis-ci.org/toddsundsted/mxnet.cr.svg?branch=master)](https://travis-ci.org/toddsundsted/mxnet.cr)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://toddsundsted.github.io/mxnet.cr/)

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

If you want to see what MXNet.cr can do, check out
[toddsundsted/deep-learning](https://github.com/toddsundsted/deep-learning).
It's a collection of problems and solutions from [Deep Learning - The
Straight Dope](https://gluon.mxnet.io/), a set of notebooks teaching
deep learning using MXNet.

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

## Troubleshooting

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

## OSX

On OSX, you may need to give your program a hint about the location of
the MXNet shared library (*libmxnet.so*). If you build and run your
program and see an error message like the following:

```
dyld: Library not loaded: lib/libmxnet.so
  Referenced from: /Users/homedirectory/.cache/crystal/crystal-run-eval.tmp
  Reason: image not found
```

you need to either: 1) explicitly set the `DYLD_FALLBACK_LIBRARY_PATH`
environment variable to point to the directory containing *libmxnet.so*,
or 2) move or copy *libmxnet.so* into a well-known location (such as
the project's own *lib* directory).

Alternatively, and more permanently, you can modify the *libmxnet.so*
shared library so that it knows where it's located at runtime (you
will modify the library's LC\_ID\_DYLIB information):

```
LIBMXNET=/Users/homedirectory/mxnet-1.5.1/lib/python3.6/site-packages/mxnet/libmxnet.so # the full path
install_name_tool -id $LIBMXNET $LIBMXNET
```

# Status

MXNet.cr currently implements a subset of
[Gluon](https://gluon.mxnet.io/), and supports most of the basic
arithmetic operations on arrays and symbols, with support for symbolic
evaluation and some support for automatic differentiation thrown
in. Almost all operations in the library are exposed, however, via the
automatically generated `Ops`, `Sparse`, `Linalg`, etc. modules but
documentation and guidance are nonexistent at this time.

Implemented functionality:
* MXNet
  * Autograd
  * Context
  * Executor
  * Optimizer
  * NDArray
  * Symbol
  * Gluon
    * Block
    * HybridBlock
    * Sequential
    * HybridSequential
    * SymbolBlock
    * Dense
    * Pooling
    * Conv1D
    * Conv2D
    * Conv3D
    * MaxPool1D
    * MaxPool2D
    * MaxPool3D
    * Flatten
    * L1Loss
    * L2Loss
    * SoftmaxCrossEntropyLoss
    * Activation
    * Trainer
    * Parameter
    * Constant
