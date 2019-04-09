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
