require "spec"
require "../src/mxnet"

def gpu_enabled?
  begin
    {% if compare_versions(MXNet::Internal::MXNET_VERSION, "1.3.0") >= 0 %}
      MXNet::Context.num_gpus > 0
    {% else %}
      MXNet::NDArray.array([0], ctx: MXNet.gpu(0)) && true
    {% end %}
  rescue MXNet::Internal::LibraryException | MXNet::MXNetException
    false
  end
end
