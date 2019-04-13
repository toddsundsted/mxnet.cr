require "../gluon"

module MXNet
  module Gluon
    # Base class for all neural network layers and models. Your models
    # should subclass this class.
    #
    class Block
      # Calls `#forward`.
      #
      # Only accepts positional arguments.
      #
      # ### Parameters
      # * *inputs* (`Array(NDArray)`)
      #   Input tensors.
      #
      def call(inputs : Array(T)) : Array(T) forall T
        forward(inputs)
      end

      # Override to implement forward computation using `NDArray`.
      #
      # Only accepts positional arguments.
      #
      # ### Parameters
      # * *inputs* (`Array(NDArray)`)
      #   Input tensors.
      #
      def forward(inputs : Array(T)) : Array(T) forall T
        raise NotImplementedError.new(
          "#forward must be implemented in a subclass"
        )
      end
    end

    # `HybridBlock` supports forwarding with both `Symbol` and
    # `NDArray`.
    #
    class HybridBlock < Block
      # Defines the forward computation.
      #
      # ### Parameters
      # * *inputs* (`Array(Symbol)` or `Array(NDArray)`)
      #   Input tensors.
      #
      def forward(inputs : Array(T)) : Array(T) forall T
        hybrid_forward(inputs)
      end

      # Override to construct symbolic graph for this `HybridBlock`.
      #
      # ### Parameters
      # * *inputs* (`Array(Symbol)` or `Array(NDArray)`)
      #   Input tensors.
      #
      def hybrid_forward(inputs : Array(T), params : Hash(String, T) = {} of String => T) : Array(T) forall T
        raise NotImplementedError.new(
          "#hybrid_forward must be implemented in a subclass"
        )
      end
    end
  end
end
