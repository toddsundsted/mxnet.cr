module MXNet
  # Get the MXNet library version.
  def self.version
    Internal.libcall(MXGetVersion, out version)
    version
  end
end

require "./mxnet/libmxnet"
require "./mxnet/context"
require "./mxnet/ndarray"
require "./mxnet/symbol"
