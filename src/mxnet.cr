# [MXNet](https://mxnet.incubator.apache.org/) for
# [Crystal](https://crystal-lang.org/).
#
# This library is built on top of the core packages `NDArray` and
# `Symbol`.
#
# `NDArray` works with arrays in an imperative fashion, i.e. you
# define how arrays will be transformed to get to an end
# result. `Symbol` works with arrays in a declarative fashion,
# i.e. you define the end result that is required (via a symbolic
# graph) and the MXNet engine will use various optimizations to
# determine the steps required to obtain this. With `NDArray` you have
# a great deal of flexibility when composing operations, and you can
# easily step through your code and inspect the values of arrays,
# which helps with debugging. Unfortunately, this flexibility comes at
# a performance cost when compared to `Symbol`, which can perform
# optimizations on the symbolic graph.
#
module MXNet
  VERSION = "0.1.0"

  {% begin %}
    {% version = `python '#{__DIR__}/mxnet/libmxnet.py' version`.stringify.chomp %}
    puts "MXNet.cr version #{MXNet::VERSION} built with MXNet version " + {{ version }}
    Internal::MXNET_VERSION = {{ version }}
  {% end %}

  class MXNetException < Exception
  end

  # Returns a CPU context.
  #
  # This function is equivalent to `MXNet::Context.cpu`.
  #
  # ### Parameters
  # * *device_id* (`Int32`, default = 0)
  #   Device id of the device. Not required for the CPU
  #   context. Included to make the interface compatible with GPU
  #   contexts.
  #
  def self.cpu(device_id : Int32 = 0)
    MXNet::Context.cpu(device_id)
  end

  # Returns a GPU context.
  #
  # This function is equivalent to `MXNet::Context.gpu`.
  #
  # ### Parameters
  # * *device_id* (`Int32`, default = 0)
  #   Device id of the device. Required for the GPU contexts.
  #
  def self.gpu(device_id : Int32 = 0)
    MXNet::Context.gpu(device_id)
  end
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
