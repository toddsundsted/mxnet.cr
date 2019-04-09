module MXNet
  VERSION = "0.1.0"

  {% begin %}
    {% version = `python '#{__DIR__}/mxnet/libmxnet.py' version`.stringify.chomp %}
    puts "MXNet.cr version #{MXNet::VERSION} built with MXNet version " + {{ version }}
    Internal::MXNET_VERSION = {{ version }}
  {% end %}

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
