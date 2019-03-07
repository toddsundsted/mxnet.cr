module MXNet
  # Get the MXNet library version.
  def self.version
    Internal.libcall(MXGetVersion, out version)
    version
  end
end

require "./mxnet/libmxnet"
require "./mxnet/name/manager"
require "./mxnet/operations"
require "./mxnet/context"
require "./mxnet/executor"
require "./mxnet/ndarray"
require "./mxnet/symbol"
