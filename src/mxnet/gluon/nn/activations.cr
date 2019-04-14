require "../nn"

module MXNet
  module Gluon
    module NN
      # Applies an activation function to input.
      #
      class Activation < MXNet::Gluon::HybridBlock
        # Creates a new instance.
        #
        # ### Parameters
        # * *activation* (`String` or `::Symbol`)
        #   Name of activation function to use.
        #
        def initialize(@activation : String | ::Symbol, **kwargs)
          super(**kwargs)
        end

        # :nodoc:
        def hybrid_forward(inputs : Array(T), params : Hash(String, T)) : Array(T) forall T
          [T.activation(inputs.first, act_type: @activation, name: "fwd")]
        end

        def to_s(io)
          io << "Activation(" << @activation << ")"
        end
      end
    end
  end
end
