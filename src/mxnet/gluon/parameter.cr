require "../gluon"

module MXNet
  module Gluon
    # Error for unfinished deferred initializations.
    #
    class DeferredInitializationError < Exception
    end

    # A Container holding parameters (weights) of `Block`s.
    #
    # `Parameter` holds a copy of the parameter on each `Context`
    # after it is initialized with `#init`. If *grad_req* is not
    # `:null`, it also holds a gradient array on each `Context`.
    #
    class Parameter
      @name : String
      @shape : Array(Int32) = [] of Int32
      @dtype : ::Symbol = :float32
      @data : Array(NDArray)?
      @grad : Array(NDArray)?
      @init : Initializer?
      @ctx : Array(Context)?
      @trainer : Trainer?
      @var : Symbol?

      # :nodoc:
      alias InitType = String | ::Symbol | Initializer.class | Initializer
      # :nodoc:
      record DeferredInit, ctx : Array(Context), init : InitType, data : NDArray?
      @deferred_init : DeferredInit?

      # Creates a new instance.
      #
      # ### Parameters
      # * *name* (`String`)
      #   Name of this parameter.
      # * *shape* (`Int` | `Array(Int)`, optional)
      #   Shape of this parameter.  By default, shape is inferred.
      # * *dtype* (`Symbol`, default `:float32`)
      #   Data type of this parameter.
      # * *init* (`Initializer`, optional)
      #   The initializer to use.
      # * *allow_deferred_init* (`Bool`, default = `false`)
      #   Is deferred initialization allowed.
      # * *grad_req* (`Symbol`, default  `:write`)
      #   * `:write`: updated is written to the gradient
      #   * `:add`: update is added to the existing gradient.
      #   * `:null`: gradient is not supported on this parameter
      #
      def initialize(@name, shape = nil, dtype = nil,
                     @init = nil, @allow_deferred_init = false,
                     @grad_req = :write)
        shape = shape.is_a?(Int) ? [shape] : shape
        self.shape = shape if shape
        self.dtype = dtype if dtype
      end

      getter :name, :shape, :dtype

      def shape=(shape)
        if @shape.empty?
          @shape = shape
        else
          unless @shape.size == shape.size &&
                 @shape.zip(shape).all? { |i, j| i == j || i == 0 }
            raise Exception.new(
              "Expected shape #{shape} is incompatible " \
              "with given shape #{@shape}."
            )
          end
          @shape = shape
        end
      end

      def dtype=(dtype)
        @dtype = dtype
      end

      property :trainer

      # Initializes parameter and gradient arrays. Only used with
      # `NDArray` API.
      #
      # ### Parameters
      # * *init* (`Initializer`, default = `nil`)
      #   The initializer to use. Overrides both *init*, set when this
      #   instance was created, and *default_init* in this call.
      # * *ctx* (`Context` | `Array(Context)`, default = `nil`)
      #   Initialize `Parameter` on given `Context`s. A copy will be
      #   created for each context. Note: copies are independent
      #   arrays. The programmer is responsible for keeping values
      #   consistent when updating. Normally `Trainer` does this for
      #   you.
      # * *default_init* (`Initializer`, default = `:uniform`)
      #   Default initializer.
      # * *force_reinit* (`Bool`, default = `false`)
      #   Whether to force re-initialization if parameter is already
      #   initialized.
      #
      #     weight = MXNet::Gluon::Parameter.new('weight', shape: [2, 2])
      #     weight.init(ctx: MXNet.cpu)
      #     weight.data # => [[0.0068339, 0.0129982],...
      #     weight.grad # => [[0, 0],...
      #
      def init(init = nil, ctx = nil, default_init = :uniform, force_reinit = false)
        unless @data.nil? || force_reinit
          return self
        end
        init = (@init || default_init) if init.nil?
        ctx = [MXNet::Context.current] if ctx.nil?
        ctx = [ctx] if ctx.is_a?(MXNet::Context)
        @ctx = ctx
        @data = @grad = nil
        if @shape.empty? || @shape.flatten.product <= 0
          unless @allow_deferred_init
            raise Exception.new(
              "Cannot initialize Parameter '#{@name}' because it has " \
              "invalid shape: #{@shape}."
            )
          end
          @deferred_init = DeferredInit.new(ctx, init, nil)
          return self
        end
        @deferred_init = DeferredInit.new(ctx, init, nil)
        _finish_deferred_init
        self
      end

      # Returns a list of contexts this parameter is initialized on.
      #
      def list_ctx
        if @data.nil?
          @deferred_init.try(&.ctx) ||
            raise Exception.new(
              "Parameter '#{@name}' has not been initialized."
            )
        else
          @ctx
        end
      end

      # Returns a symbol representing this parameter.
      #
      def var
        @var ||= MXNet::Symbol.var(@name, shape: @shape, dtype: @dtype)
      end

      # Returns a copy of this parameter on one context. Must have been
      # initialized on this context before.
      #
      # ### Parameters
      # * *ctx* (`Context`, optional)
      #   Desired context.
      #
      def data(ctx = nil)
        check_and_get(@data, ctx).first
      end

      # Returns copies of this parameter on all contexts, in the same
      # order as creation.
      #
      def list_data
        check_and_get(@data, :all)
      end

      # Sets this parameter's value on all contexts.
      #
      def set_data(data)
        @shape = data.shape
        if @data
          check_and_get(@data, :all).each do |arr|
            arr[0..-1] = data
          end
        else
          if @deferred_init.nil?
            raise Exception.new(
              "Parameter '#{@name}' has not been initialized."
            )
          end
          @deferred_init = @deferred_init.try do |deferred_init|
            DeferredInit.new(
              deferred_init.ctx,
              deferred_init.init,
              data
            )
          end
        end
      end

      # Returns a gradient buffer for this parameter on one context.
      # Must have been initialized on this context before.
      #
      # ### Parameters
      # * *ctx* (`Context`, optional)
      #   Desired context.
      #
      def grad(ctx = nil)
        if !@grad && @data
          raise Exception.new(
            "Cannot get gradient buffer for Parameter '#{name}' " \
            "because grad_req = :null."
          )
        end
        check_and_get(@grad, ctx).first
      end

      # Returns gradient buffers on all contexts, in the same order as
      # creation.
      #
      def list_grad
        if !@grad && @data
          raise Exception.new(
            "Cannot get gradient buffers for Parameter '#{name}' " \
            "because grad_req = :null."
          )
        end
        check_and_get(@grad, :all)
      end

      # Sets gradient buffer to zero on all contexts.
      #
      def zero_grad
        if @grad
          check_and_get(@grad, :all).each do |arr|
            arr[0..-1] = 0
          end
        else
          if @deferred_init.nil?
            raise Exception.new(
              "Parameter '#{@name}' has not been initialized."
            )
          end
        end
      end

      # Writes this object to an `IO`.
      #
      def to_s(io)
        io << "Parameter #{@name} (shape=#{@shape}, dtype=#{@dtype})"
      end

      def ==(other : self)
        self.name == other.name && self.shape == other.shape
      end

      # Reduce data from multiple contexts to CPU.
      #
      private def reduce
        data = list_data.map { |d| d.copy_to(MXNet.cpu) }
        MXNet::NDArray.add_n(*data) / data.size
      end

      private def check_and_get(arr_list, ctx)
        unless arr_list.nil?
          if ctx == :all
            return arr_list
          elsif ctx.nil?
            if arr_list.size == 1
              return arr_list
            end
            ctx = MXNet::Context.current
          end
          data = arr_list.select { |arr| arr.context == ctx }
          if data.size > 0
            return data
          end
          raise Exception.new(
            "Parameter '#{@name}' was not initialized on context #{ctx}."
          )
        end
        unless @deferred_init.nil?
          raise DeferredInitializationError.new(
            "Parameter '#{@name}' has not been initialized yet because " \
            "initialization was deferred. Actual initialization happens " \
            "during the first forward pass. Please pass one batch of " \
            "data through the network before accessing Parameters."
          )
        end
        raise Exception.new(
          "Parameter '#{@name}' has not been initialized. You should " \
          "initialize parameters and create a Trainer with #collect_params " \
          "instead of #params because the later does not include Parameters " \
          "of nested child Blocks."
        )
      end

      private def load_and_init(ctx, data)
        ctx = ctx.is_a?(MXNet::Context) ? [ctx] : ctx
        if @data
          set_data(data)
        else
          init_impl(ctx, data)
        end
      end

      # :nodoc:
      def _finish_deferred_init
        return unless deferred_init = @deferred_init
        ctx = deferred_init.ctx
        default_init = deferred_init.init
        data = deferred_init.data
        @deferred_init = nil
        if @shape.empty? || @shape.flatten.product <= 0
          raise Exception.new(
            "Cannot initialize Parameter '#{@name}' because it has " \
            "invalid shape: #{@shape}."
          )
        end
        MXNet::Autograd.pause do
          unless data
            data = MXNet::NDArray.zeros(@shape, dtype: @dtype, ctx: MXNet.cpu)
            MXNet::Initializer.create(default_init).init_array(data)
          end
          init_impl(ctx, data)
        end
      end

      private def init_impl(ctx, data)
        @data = ctx.map { |c| data.copy_to(c) }
        grad = MXNet::NDArray.zeros(@shape, dtype: @dtype, ctx: MXNet.cpu)
        init_grad(ctx, grad)
      end

      private def init_grad(ctx, grad)
        if @grad_req == :null
          @grad = nil
          return
        end
        @grad = ctx.map { |c| grad.copy_to(c) }
        MXNet::Autograd.mark_variables(
          check_and_get(@data, :all),
          check_and_get(@grad, :all),
          grad_reqs: @grad_req
        )
      end
    end

    # A constant parameter for holding immutable tensors.
    #
    # `Constant`s are ignored by `Autograd` and `Trainer`, thus their
    # values will not change during training. But you can still update
    # their values manually with the `set_data` method.
    #
    class Constant < Parameter
      # :nodoc:
      class ConstantParameterInitializer < MXNet::Initializer
        @value : MXNet::NDArray

        def initialize(value = nil, **kwargs)
          @value = value || MXNet::NDArray.zeros(0)
          super(**kwargs)
        end

        def init_array(array)
          @value.copy_to(array)
        end
      end

      # Creates a new instance.
      #
      # ### Parameters
      # * *name* (`String`, required)
      #    Name of the constant.
      # * *value* (`Array` | `NDArray`, required)
      #   Initial value for the constant.
      #
      def initialize(name, value)
        unless value.is_a?(MXNet::NDArray)
          value = MXNet::NDArray.array(value)
        end
        super(
          name,
          init: ConstantParameterInitializer.new(value),
          grad_req: :null,
          shape: value.shape,
          dtype: value.dtype
        )
      end

      # Writes this object to an `IO`.
      #
      def to_s(io)
        io << "Constant #{@name} (shape=#{@shape}, dtype=#{@dtype})"
      end
    end

    # A dictionary managing a set of `Parameter`s.
    #
    class ParameterDict
      include Enumerable({String, Parameter})

      @prefix : String
      @shared : ParameterDict?
      @params = {} of String => Parameter

      # Creates a new instance.
      #
      # ### Parameters
      # * *prefix* (`String`, default "")
      #   The prefix to be prepended to all `Parameter`s' names
      #   created by this dict.
      # * *shared* (`ParameterDict`, optional)
      #   If not `nil`, when this dict's `#get` method creates a new
      #   parameter, it will first try to retrieve it from *shared*
      #   dict. Usually used for sharing parameters with another
      #   `Block`.
      #
      def initialize(@prefix = "", @shared = nil)
      end

      def each
        @params.each do |k, v|
          yield({k, v})
        end
      end

      def keys
        @params.keys
      end

      def has_key?(key)
        @params.has_key?(key)
      end

      def values
        @params.values
      end

      def has_value?(value)
        @params.has_value?(value)
      end

      # Prefix of this dict. It will be prepended to a `Parameter`'s
      # name created with `#get`.
      #
      def prefix
        @prefix
      end

      # Retrieves a `Parameter` with name "<prefix><name>". If not
      # found, `#get` will first try to retrieve it from *shared*
      # dict. If still not found, `#get` will create a new `Parameter`
      # with key-word arguments and both store and return it.
      #
      # ### Parameters
      # * *name* (`String`)
      #   Name of the desired `Parameter`. It will be prepended with
      #   this dict's *prefix*.
      #
      def get(name, **kwargs)
        name = @prefix + name
        unless param = _get(name)
          param = @params[name] = Parameter.new(name, **kwargs)
        end
        param
      end

      # Copies all `Parameter`s in *other* into this dict.
      #
      # ### Parameters
      # * *other* (`Enumerable({String, Parameter})`)
      #   Dict to copy from.
      #
      def update(other : Enumerable({String, Parameter}))
        other.each do |k, v|
          if @params[k]? && @params[k] != v
            raise ArgumentError.new(
              "Cannot update because keys have different values: #{@params[k]}, #{v}"
            )
          end
        end
        other.each do |k, v|
          @params[k] = v
        end
      end

      # Initializes all `Parameter`s managed by this dict for use with
      # `NDArray` API. It has no effect when using `Symbol` API.
      #
      # ### Parameters
      # * *init* (`Initializer`, optional)
      #   The initializer to use.
      # * *ctx* (`Context` | `Array(Context)`, optional)
      #   Desired contexts. Initializes `Parameter` on given contexts.
      # * *force_reinit* (`Boolean`, default = `false`)
      #    Whether to force re-initialization if parameters are
      #    already initialized.
      #
      def init(init = nil, ctx = nil, force_reinit = false)
        values.each do |v|
          v.init(init: init, ctx: ctx, force_reinit: force_reinit)
        end
      end

      # Writes this object to an `IO`.
      #
      def to_s(io)
        io << "ParameterDict (\n"
        self.each { |_, v| io << "  #{v}\n" }
        io << ")"
      end

      def ==(other : self)
        self.prefix == other.prefix && self.to_a == other.to_a
      end

      protected def _get(name)
        if value = @params[name]?
          value
        elsif @shared && (value = @shared.try(&._get(name)))
          @params[name] = value
          value
        end
      end
    end
  end
end
