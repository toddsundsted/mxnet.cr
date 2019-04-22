require "./operations"

module MXNet
  # Base class for `NDArray` and `Symbol`.
  #
  class Base
    private macro inherited
      include MXNet::Operations
      extend MXNet::Operations
      extend MXNet::Util
    end

    # :nodoc:
    DT2T = {
      0 => :float32,
      1 => :float64,
      3 => :uint8,
      4 => :int32,
      5 => :int8,
      6 => :int64
    }
    # :nodoc:
    T2DT = {
      :float32 => 0,
      :float64 => 1,
      :uint8 => 3,
      :int32 => 4,
      :int8 => 5,
      :int64 => 6
    }
  end
end
