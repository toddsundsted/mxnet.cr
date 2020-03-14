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

    private macro bifunc_helper(op, arg1, arg2, fn_array, fn_scalar, fn_lscalar, fn_rscalar)
      def self.{{op}}({{arg1}} : self | Number, {{arg2}} : self | Number)
        if {{arg1}}.is_a?(self)
          if {{arg2}}.is_a?(self)
            {{fn_array}}({{arg1}}, {{arg2}})
          else
            {{fn_rscalar}}({{arg1}}, scalar: {{arg2}})
          end
        else
          if {{arg2}}.is_a?(self)
            {{fn_lscalar}}({{arg2}}, scalar: {{arg1}})
          else
            # this case is handled in a separate specialized method
            raise "should never happen"
          end
        end
      end
      # :nodoc:
      def self.{{op}}({{arg1}} : Number, {{arg2}} : Number)
        {% if fn_scalar.is_a?(If) %}
          {{fn_scalar}}
        {% elsif fn_scalar.is_a?(SymbolLiteral) %}
          {{arg1}}.{{fn_scalar.id}}({{arg2}})
        {% else %}
          {% raise "not supported: #{fn_scalar}"%}
        {% end %}
      end
    end
  end
end
