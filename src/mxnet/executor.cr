module MXNet
  # `Executor` provides efficient symbolic graph
  # execution and optimization.
  class Executor
    @handle : MXNet::Internal::LibMXNet::ExecutorHandle

    def initialize(handle)
      @handle = handle
    end

    # :nodoc:
    def handle
      @handle
    end

    # Calculate the outputs specified by the bound symbol.
    #
    # ### Parameters
    # * *is_train* (`Bool`, default `false`)
    #   Whether this `#forward` call is for training. If `true`, a
    #   `#backward` call is expected to follow.
    #
    def forward(is_train : Bool = false)
      MXNet::Internal.libcall(
        MXExecutorForward,
        @handle,
        is_train ? 1 : 0
      )
      MXNet::Internal.libcall(
        MXExecutorOutputs,
        @handle,
        out num_outputs,
        out outputs
      )
      num_outputs.times.map { |i| NDArray.new(outputs[i]) }
    end

    # Do backward pass to calculate the gradients.
    #
    # ### Parameters
    # * *out_grads* (`MXNet::NDArray`, optional)
    #   Gradients on the outputs to be propagated back. This parameter
    #   is only needed when `MXNet::Symbol#bind` is called on outputs
    #   that are not a loss function.
    # * *is_train* (`Bool`, default `true`)
    #   Whether this `#backward` call is for training. Note, in rare
    #   cases you may want to call `#backward` with `is_train: false`
    #   to calculate gradients during inference.
    #
    def backward(out_grads : Array(MXNet::NDArray) = [] of MXNet::NDArray, is_train : Bool = true)
      MXNet::Internal.libcall(
        MXExecutorBackwardEx,
        @handle,
        out_grads.size,
        out_grads.map(&.handle),
        is_train ? 1 : 0
      )
    end

    def finalize
      MXNet::Internal.libcall(MXExecutorFree, @handle)
    end
  end
end
