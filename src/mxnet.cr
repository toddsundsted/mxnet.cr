module MXNet
  VERSION = "0.1.0"

  class MXNetException < Exception
  end

  private module ClassMethods
    delegate cpu, gpu, to: MXNet::Context
  end

  extend ClassMethods
end

require "./mxnet/libmxnet"
require "./mxnet/name/manager"
require "./mxnet/autograd"
require "./mxnet/context"
require "./mxnet/random"
require "./mxnet/executor"
require "./mxnet/initializer"
require "./mxnet/base"
require "./mxnet/ndarray"
require "./mxnet/symbol"
