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

  struct CloseExpectation(T, D)
    def match(actual_value)
      expected_value = @expected_value
      if actual_value.is_a?(MXNet::NDArray) && expected_value.is_a?(MXNet::NDArray)
        return false unless expected_value.dtype == actual_value.dtype
        return false unless expected_value.shape == actual_value.shape
        e_raw_t = expected_value.raw
        a_raw_t = actual_value.raw
        case a_raw_t
        when Array(Float32)
          return false if e_raw_t.zip(a_raw_t).any? { |e, a| (e - a).abs > @delta }
        when Array(Float64)
          return false if e_raw_t.zip(a_raw_t).any? { |e, a| (e - a).abs > @delta }
        when Array(UInt8)
          return false if e_raw_t.zip(a_raw_t).any? { |e, a| (e - a).abs > @delta }
        when Array(Int32)
          return false if e_raw_t.zip(a_raw_t).any? { |e, a| (e - a).abs > @delta }
        when Array(Int8)
          return false if e_raw_t.zip(a_raw_t).any? { |e, a| (e - a).abs > @delta }
        when Array(Int64)
          return false if e_raw_t.zip(a_raw_t).any? { |e, a| (e - a).abs > @delta }
        end
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
          Expected: #{expected_value[0]} to be within #{@delta}
                    #{expected_value[1]}
                of: #{actual_value[0]}
                    #{actual_value[1]}
          MSG
      end
      previous_def
    end
  end
end
