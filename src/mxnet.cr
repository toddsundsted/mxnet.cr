module MXNet
  # Get the MXNet library version.
  def self.version
    libcall(MXGetVersion, out version)
    version
  end
end

require "./mxnet/libmxnet"
