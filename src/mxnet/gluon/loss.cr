require "../gluon"

module MXNet
  module Gluon
    # Base class for loss.
    #
    class Loss < MXNet::Gluon::HybridBlock
      # Creates a new instance.
      #
      # ### Parameters
      # * *weight* (`Float` or `nil`)
      #   Global scalar weight for loss.
      # * *batch_axis* (`Int`)
      #   The axis that represents the mini-batch.
      #
      def initialize(@weight : Float64?, @batch_axis : Int32, **kwargs)
        super(**kwargs)
      end

      # Calculates the mean absolute error between prediction and label.
      #
      # Inputs "prediction" and "label" can have arbitrary shape as long
      # as they have the same number of elements.
      #
      class L1Loss < Loss
        # Creates a new instance.
        #
        # ### Parameters
        # * *weight* (`Float` or `nil`, default = `nil`)
        #   Global scalar weight for loss.
        # * *batch_axis* (`Int`, default 0)
        #   The axis that represents the mini-batch.
        #
        def initialize(weight = nil, batch_axis = 0, **kwargs)
          super(weight, batch_axis, **kwargs)
        end

        # :nodoc:
        def hybrid_forward(inputs : Array(T), params : Hash(String, T)) : Array(T) forall T
          prediction, label = inputs
          sample_weight = params["sample_weight"]?
          label = T.reshape_like(label, prediction)
          loss = T.abs(prediction - label)
          loss = apply_weighting(T, loss, @weight, sample_weight)
          [T.mean(loss, axis: @batch_axis, exclude: true)]
        end
      end

      # Calculates the mean squared error between prediction and label.
      #
      # Inputs "prediction" and "label" can have arbitrary shape as long
      # as they have the same number of elements.
      #
      class L2Loss < Loss
        # Creates a new instance.
        #
        # ### Parameters
        # * *weight* (`Float` or `nil`, default = 1.0)
        #   Global scalar weight for loss.
        # * *batch_axis* (`Int`, default 0)
        #   The axis that represents the mini-batch.
        #
        def initialize(weight = 1.0, batch_axis = 0, **kwargs)
          super(weight, batch_axis, **kwargs)
        end

        # :nodoc:
        def hybrid_forward(inputs : Array(T), params : Hash(String, T)) : Array(T) forall T
          prediction, label = inputs
          sample_weight = params["sample_weight"]?
          label = T.reshape_like(label, prediction)
          loss = T.square(prediction - label)
          loss = apply_weighting(T, loss, @weight.try(&./(2)), sample_weight)
          [T.mean(loss, axis: @batch_axis, exclude: true)]
        end
      end

      # Computes the softmax cross-entropy loss.
      #
      # If *sparse_label* is `true` (default), labels should contain
      # integer category indicators. The labels' shape should be the
      # predictions' shape with the *axis* dimension removed --
      # i.e. for predictions with shape `[1, 2, 3, 4]` and `axis: 2`,
      # labels' shape should be `[1, 2, 4]`.
      #
      # If *sparse_label* is `false`, labels should contain
      # probability distributions and labels' shape should be the same
      # as predictions' shape.
      #
      class SoftmaxCrossEntropyLoss < Loss
        # Creates a new instance.
        #
        # ### Parameters
        # * *axis* (`Int`, default = -1)
        #   The axis to sum over when computing softmax and entropy.
        # * *sparse_label* (`Bool`, default = `true`)
        #   Whether label is an integer array instead of probability
        #   distribution.
        # * *from_logits* (`Bool`, default = `false`)
        #   Whether prediction is a log probability (usually from
        #   `#log_softmax`) instead of unnormalized numbers.
        # * *weight* (`Float` or `nil`, default = `nil`)
        #   Global scalar weight for loss.
        # * *batch_axis* (`Int`, default 0)
        #   The axis that represents the mini-batch.
        #
        def initialize(@axis = -1, @sparse_label = true, @from_logits = false,
                       weight = nil, batch_axis = 0, **kwargs)
          super(weight, batch_axis, **kwargs)
        end

        # :nodoc:
        def hybrid_forward(inputs : Array(T), params : Hash(String, T)) : Array(T) forall T
          prediction, label = inputs
          sample_weight = params["sample_weight"]?
          unless @from_logits
            prediction = T.log_softmax(prediction, axis: @axis)
          end
          if @sparse_label
            loss = -T.pick(prediction, label, axis: @axis, keepdims: true)
          else
            label = T.reshape_like(label, prediction)
            loss = -T.sum(prediction * label, axis: @axis, keepdims: true)
          end
          loss = apply_weighting(T, loss, @weight, sample_weight)
          [T.mean(loss, axis: @batch_axis, exclude: true)]
        end
      end

      # Apply weighting to loss.
      #
      # ### Parameters
      # * *loss* (`Symbol` or `NDArray`)
      #   The loss to be weighted.
      # * *weight* (`Float` or `nil`)
      #   Global scalar weight for loss.
      # * *sample_weight* (`Symbol`, `NDArray` or +nil+)
      #   Per sample weighting. Must be broadcastable to the same
      #   shape as loss. For example, if loss has shape `[64, 10]` and
      #   you want to weight each sample in the batch separately,
      #   *sample_weight* should have shape `[64, 1]`.
      #
      protected def apply_weighting(clazz, loss, weight = nil, sample_weight = nil)
        unless sample_weight.nil?
          loss = clazz.broadcast_mul(loss, sample_weight)
        end
        unless weight.nil?
          loss *= weight
        end
        loss
      end
    end
  end
end
