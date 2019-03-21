require "spec"
require "../src/mxnet"

class MXNet::NDArray
  def raw
    previous_def
  end
end

module Spec
  struct EqualExpectation(T)
    def match(actual_value)
      expected_value = @expected_value
      if actual_value.is_a?(MXNet::NDArray) && expected_value.is_a?(MXNet::NDArray)
        return false unless expected_value.dtype == actual_value.dtype
        return false unless expected_value.shape == actual_value.shape
        return false unless expected_value.raw == actual_value.raw
        return true
      end
      previous_def
    end

    def failure_message(actual_value)
      expected_value = @expected_value
      if actual_value.is_a?(MXNet::NDArray) && expected_value.is_a?(MXNet::NDArray)
        expected_value = expected_value.to_s.split("\n")
        actual_value = actual_value.to_s.split("\n")
        return <<-MSG
          Expected: #{expected_value[0]}
                    #{expected_value[1]}
               got: #{actual_value[0]}
                    #{actual_value[1]}
          MSG
      end
      previous_def
    end
  end
end

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
