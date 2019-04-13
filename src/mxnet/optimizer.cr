module MXNet
  # The base class inherited by all optimizers.
  #
  # Custom optimizers can be created by subclassing `Optimizer` and
  # implementing the required function `#update`. By default, the
  # created optimizer will be registered under its simplified class
  # name (`class.name.split("::").last.downcase`) but it may also be
  # registered under another name by calling `#register`.
  #
  #     class MyOptimizer < MXNet::Optimizer
  #       register :myopt
  #       def update(index, weight, gradient, state)
  #         weight
  #       end
  #     end
  #
  abstract class Optimizer
    @@registry = Hash(String, Optimizer.class).new

    protected def self.register_optimizer(name, optimizer)
      @@registry[name.to_s] = optimizer
    end

    private macro inherited
      MXNet::Optimizer.register_optimizer({{@type.name.split("::").last.downcase}}, {{@type}})
    end

    protected def self.register(name)
      MXNet::Optimizer.register_optimizer(name, self)
    end

    def self.create(optimizer, **kwargs)
      case optimizer
      when ::String, ::Symbol
        @@registry[optimizer.to_s].new(**kwargs)
      when .responds_to?(:new)
        optimizer.new(**kwargs)
      else
        optimizer
      end
    end

    # Creates a new instance.
    #
    # ### Parameters
    # * *rescale_grad* (`Float`, optional)
    #   Before updating, multiply the gradient by
    #   *rescale_grad*. Often chosen to be *1.0 / batch_size*.
    # * *clip_gradient* (`Float`, optional)
    #   Clip the gradient by projecting onto the box
    #   `[-clip_gradient, clip_gradient]`.
    # * *lr* (`Float`, optional)
    #   The initial learning rate.
    # * *wd* (`Float`, optional)
    #   The weight decay (or L2 regularization) coefficient. Modifies
    #   the objective by adding a penalty for having large weights.
    #
    def initialize(@rescale_grad = 1.0, @clip_gradient = -1.0, @lr = 0.01, @wd = 0.0)
      @lr_mult = {} of Int32 => Float64
      @wd_mult = {} of Int32 => Float64
    end

    property :rescale_grad

    getter :lr, :wd

    # Sets an individual learning rate multiplier for each parameter.
    #
    # If you specify a learning rate multiplier for a parameter, then
    # the learning rate for that parameter will be set as the product
    # of the global learning rate and its multiplier.
    #
    # ### Parameters
    # * *lr_mult* (`Hash(Int, Float)`)
    #   For each of the entries, the learning rate multipler for the
    #   parameter specified will be set as the given value.
    #
    def set_lr_mult(lr_mult)
      @lr_mult = lr_mult
    end

    # Sets an individual weight decay multiplier for each parameter.
    #
    # If you specify a weight decay multiplier for a parameter, then
    # the weight decay for that parameter will be set as the product
    # of the global weight decay and its multiplier.
    #
    # ### Parameters
    # * *wd_mult* (`Hash(Int, Float)`)
    #   For each of the entries, the weight decay multipler for the
    #   parameter specified will be set as the given value.
    #
    def set_wd_mult(wd_mult)
      @wd_mult = wd_mult
    end

    # Updates the given parameter using the corresponding gradient and
    # state.
    #
    # ### Parameters
    # * *index* (`Int`)
    #   The unique index of the parameter into the individual learning
    #   rates and weight decays. Learning rates and weight decay may
    #   be set via `#set_lr_mult` and `#set_wd_mult`, respectively.
    # * *weight* (`NDArray`)
    #   The parameter to be updated.
    # * *gradient* (`NDArray`)
    #   The gradient of the objective with respect to this parameter.
    # * *state* (any)
    #   The state returned by `#create_state`.
    #
    abstract def update(index, weight, gradient, state)

    # Creates auxiliary state for a given weight.
    #
    # Some optimizers require additional states (e.g. momentum) in
    # addition to gradients in order to update weights. This function
    # creates state for a given weight which will be used in
    # update. This function is called only once for each weight.
    #
    # ### Parameters
    # * *index* (`Int`)
    #   A unique index to identify the weight.
    # * *weight* (`NDArray`)
    #   The weight.
    #
    def create_state(index, weight)
    end

    # The SGD optimizer with momentum and weight decay.
    #
    # Updates are calculated by:
    #     rescaled_grad = lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
    #     state = momentum * state + rescaled_grad
    #     weight = weight - state
    #
    class SGD < Optimizer
      # Creates a new instance.
      #
      # This optimizer accepts the following parameters in addition to
      # those accepted by `Optimizer`.
      #
      # ### Parameters
      # * *momentum* (`Float`, optional)
      #   The momentum value.
      #
      def initialize(@momentum = 0.0, **kwargs)
        super(**kwargs)
      end

      # See: `MXNet::Optimizer#update`
      def update(index, weight, gradient, state)
        lr = get_lr(index)
        wd = get_wd(index)
        kwargs = {lr: lr, wd: wd, rescale_grad: @rescale_grad, clip_gradient: @clip_gradient, out: weight}
        if state
          MXNet::NDArray.sgd_mom_update(weight, gradient, state, **kwargs.merge({momentum: @momentum}))
        else
          MXNet::NDArray.sgd_update(weight, gradient, **kwargs)
        end
      end

      # See: `MXNet::Optimizer#create_state`
      def create_state(index, weight)
        if @momentum != 0
          MXNet::NDArray.zeros(weight.shape, weight.context, dtype: weight.dtype)
        else
          nil
        end
      end
    end

    # Gets the learning rate given the index of the weight.
    #
    # ### Parameters
    # * *index* (`Int`)
    #   The index corresponding to the weight.
    #
    protected def get_lr(index)
      lr = @lr
      lr *= @lr_mult[index] if @lr_mult.has_key?(index)
      lr
    end

    # Gets weight decay for index.
    #
    # ### Parameters
    # * *index* (`Int`)
    #   The index corresponding to the weight.
    #
    protected def get_wd(index)
      wd = @wd
      wd *= @wd_mult[index] if @wd_mult.has_key?(index)
      wd
    end
  end
end
