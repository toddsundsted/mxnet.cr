require "../gluon"

module MXNet
  module Gluon
    # Applies an `Optimizer` on a set of `Parameter`s.
    #
    # `Trainer` should be used together with `Autograd`.
    #
    class Trainer
      @contexts : Array(Context)
      @optimizer : Optimizer

      # Creates a new instance.
      #
      # ### Parameters
      # * *params* (`ParameterDict`)
      #   The set of parameters to optimize.
      # * *optimizer* (`Optimizer`)
      #   The optimizer to use.
      # * *optimizer_params* (`NamedTuple`)
      #   Key-word arguments to be passed to optimizer
      #   constructor. See each `Optimizer` for a list of additional
      #   supported arguments common to all optimizers.
      #
      def initialize(params, optimizer, **optimizer_params)
        @params = [] of Parameter
        @contexts = check_contexts
        @optimizer = init_optimizer(optimizer, **optimizer_params)
        @scale = optimizer_params[:rescale_grad]? || 1.0
        params = params.values if params.responds_to?(:values)
        params.each do |param|
          param.trainer = self
          @params << param
        end
      end

      def learning_rate
        @optimizer.lr
      end

      def weight_decay
        @optimizer.wd
      end

      # Makes one step of parameter update.
      #
      # This should be called after `Autograd#backward` and outside
      # of `Autograd.record`.
      #
      # ### Parameters
      # * *batch_size* (Int)
      #   Batch size of data processed. Gradient will be normalized by
      #   `1/batch_size`. Set this to 1 if you normalized loss
      #   manually with `loss = mean(loss)`.
      #
      def step(batch_size)
        @optimizer.rescale_grad = @scale / batch_size
        _update
      end

      # Makes one step of parameter update.
      #
      # This should be called after `Autograd#backward` and outside
      # of `Autograd.record`.
      #
      # ### Parameters
      # * *batch_size* (Int)
      #   Batch size of data processed. Gradient will be normalized by
      #   `1/batch_size`. Set this to 1 if you normalized loss
      #   manually with `loss = mean(loss)`.
      #
      def update(batch_size)
        @optimizer.rescale_grad = @scale / batch_size
        _update
      end

      private def check_contexts
        contexts = nil
        @params.each do |param|
          unless contexts.nil? || contexts == param.list_ctx
            raise Exception.new(
                  "All Parameters must be initialized on the same set of contexts, " \
                  "but Parameter '#{param.name}' is initialized on '#{param.list_ctx.try(&.map(&.to_s))}' " \
                  "while previous Parameters are initialized on '#{contexts.map(&.to_s)}'."
            )
          end
          contexts = param.list_ctx
        end
        contexts.nil? ? [] of Context : contexts
      end

      private def init_optimizer(optimizer, **optimizer_params)
        MXNet::Optimizer.create(optimizer, **optimizer_params)
      end

      private def _update
        @params.each.with_index do |param, i|
          param.list_data.zip(param.list_grad) do |data, grad|
            @optimizer.update(i, data, grad, nil)
          end
        end
      end
    end
  end
end
