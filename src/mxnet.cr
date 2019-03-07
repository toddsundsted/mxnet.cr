module MXNet
  # Get the MXNet library version.
  def self.version
    Internal.libcall(MXGetVersion, out version)
    version
  end

  private module ClassMethods
    delegate cpu, gpu, to: MXNet::Context
  end

  extend ClassMethods
end

require "./mxnet/libmxnet"
require "./mxnet/name/manager"
require "./mxnet/operations"
require "./mxnet/context"
require "./mxnet/executor"
require "./mxnet/ndarray"
require "./mxnet/symbol"
