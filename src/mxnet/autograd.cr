module MXNet
  # Autograd for MXNet.
  #
  # ```
  # x = MXNet::NDArray.array([1, 2, 3, 4], dtype: :float64)
  # g = MXNet::NDArray.array([0, 0, 0, 0], dtype: :float64)
  # MXNet::Autograd.mark_variables(x, g)
  # y = MXNet::Autograd.record do
  #   x * x + 1
  # end
  # MXNet::Autograd.backward(y)
  # ```
  #
  class Autograd
    private def self.with_state(record_mode = nil, train_mode = nil, &block : -> _)
      old_record_mode = -1
      old_train_mode = -1
      begin
        unless record_mode.nil?
          MXNet::Internal.libcall(
            MXAutogradSetIsRecording,
            record_mode ? 1 : 0,
            pointerof(old_record_mode)
          )
        end
        unless train_mode.nil?
          MXNet::Internal.libcall(
            MXAutogradSetIsTraining,
            train_mode ? 1 : 0,
            pointerof(old_train_mode)
          )
        end
        yield
      ensure
        if !record_mode.nil? && old_record_mode != record_mode
          MXNet::Internal.libcall(
            MXAutogradSetIsRecording,
            old_record_mode,
            out _
          )
        end
        if !train_mode.nil? && old_train_mode != train_mode
          MXNet::Internal.libcall(
            MXAutogradSetIsTraining,
            old_train_mode,
            out _
          )
        end
      end
    end

    # Gets status of recording/not recording.
    #
    def self.is_recording
      MXNet::Internal.libcall(MXAutogradIsRecording, out current)
      current
    end

    # Gets status of training/predicting.
    #
    def self.is_training
      MXNet::Internal.libcall(MXAutogradIsTraining, out current)
      current
    end

    # Creates a scope context for code that needs gradients to be
    # calculated.
    #
    # When forwarding with `train_mode = false`, the corresponding
    # `.backward` should also use `train_mode = false`, otherwise
    # the gradient is undefined.
    #
    # ### Parameters
    # * *train_mode* (`Bool`, default = true)
    #   Whether the forward pass is in training or predicting
    #   mode. This controls the behavior of some layers such as
    #   Dropout and BatchNorm.
    #
    def self.record(train_mode = true, &block : -> _)
      with_state(record_mode: true, train_mode: train_mode, &block)
    end

    # Creates a scope context for code that does not need gradients to
    # be calculated.
    #
    # ### Parameters
    # * *train_mode* (`Bool`, default = true)
    #   Whether the forward pass is in training or predicting
    #   mode.
    #
    def self.pause(train_mode = false, &block : -> _)
      with_state(record_mode: false, train_mode: train_mode, &block)
    end

    # Creates a scope context in which forward pass behavior is set to
    # training mode, without changing the recording mode.
    #
    def self.train_mode(&block : -> _)
      with_state(train_mode: true, &block)
    end

    # Creates a scope context in which forward pass behavior is set to
    # inference mode, without changing the recording mode.
    #
    def self.predict_mode(&block : -> _)
      with_state(train_mode: false, &block)
    end

    # :nodoc:
    GRAD_REQ_MAP = {
      null: 0_u32,
      write: 1_u32,
      add: 3_u32
    }

    # Mark arrays as variables to compute gradients for autograd.
    #
    # ### Parameters
    # * *variables* (`MXNet::NDArray` or `Enumerable(MXNet::NDArray)`)
    # * *gradients* (`MXNet::NDArray` or `Enumerable(MXNet::NDArray)`)
    # * *grad_reqs* (`Symbol` or `Enumerable(Symbol)`, default `:write`)
    #   * `:write`: gradient will be overwritten on every backward pass
    #   * `:add`: gradient will be added to existing value on every backward pass
    #   * `:null`: do not compute gradient
    #
    def self.mark_variables(variables, gradients, grad_reqs = :write)
      variables = variables.responds_to?(:each) ? variables.map(&.handle).to_a : [variables.handle]
      gradients = gradients.responds_to?(:each) ? gradients.map(&.handle).to_a : [gradients.handle]
      if grad_reqs.is_a?(::Symbol)
        grad_reqs = [GRAD_REQ_MAP[grad_reqs]] * variables.size
      else
        grad_reqs = grad_reqs.map { |gr| GRAD_REQ_MAP[gr] }
      end

      MXNet::Internal.libcall(
        MXAutogradMarkVariables,
        variables.size.to_i32,
        variables,
        grad_reqs,
        gradients
      )
    end

    # Compute the gradients with respect to previously marked variables.
    #
    # ### Parameters
    # * *outputs* (`MXNet::NDArray` or `Enumerable(MXNet::NDArray)`)
    #   Output arrays.
    # * *gradients* (`MXNet::NDArray` or `Enumerable(MXNet::NDArray)`)
    #   Gradients with respect to outputs.
    # * *retain_graph* (`Bool`, default false)
    #   Whether to keep computation graph to differentiate again,
    #   instead of clearing history and releasing memory.
    # * *train_mode* (`Bool`, default true)
    #   Whether the backward pass is in training or predicting mode.
    #
    def self.backward(outputs, gradients = nil, retain_graph = false, train_mode = true)
      unless outputs.nil?
        outputs =
          outputs.responds_to?(:each) ? outputs.map(&.handle).to_a : [outputs.handle]
      end
      unless gradients.nil?
        gradients =
          gradients.responds_to?(:each) ? gradients.map(&.handle).to_a : [gradients.handle]
      end

      MXNet::Internal.libcall(
        MXAutogradBackwardEx,
        outputs.size.to_i32,
        outputs, gradients,
        0,
        nil,
        retain_graph ? 1 : 0,
        0,
        train_mode ? 1 : 0,
        nil,
        nil
      )
    end
  end
end
