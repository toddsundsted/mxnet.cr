module MXNet
  class SymbolException < Exception
  end

  class Symbol
    extend MXNet::Operations

    alias SymbolHandle = MXNet::Internal::LibMXNet::SymbolHandle

    @handle : SymbolHandle

    def initialize(handle)
      @handle = handle
    end

    # :nodoc:
    def handle
      @handle
    end

    # Gets name of the symbol.
    #
    # This function only works for a non-grouped symbol. It returns
    # `nil` for a grouped symbol.
    #
    def name
      MXNet::Internal.libcall(MXSymbolGetName, @handle, out name, out success)
      success != 0 ? String.new(name) : nil
    end

    # Lists all the arguments of the symbol.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a * b
    # c.list_arguments # => ["a", "b"]
    # ```
    #
    def list_arguments
      MXNet::Internal.libcall(MXSymbolListArguments, @handle, out size, out str_array)
      str_array.to_slice(size).map { |u| String.new(u) }.to_a
    end

    # Lists all the outputs of the symbol.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b
    # c.last_outputs # => ["_plus12_output"]
    # ```
    #
    def list_outputs
      MXNet::Internal.libcall(MXSymbolListOutputs, @handle, out size, out str_array)
      str_array.to_slice(size).map { |u| String.new(u) }.to_a
    end

    # Binds the current symbol to an executor and returns the executor.
    #
    # First, declare the computation and then bind to the data to
    # evaluate.  This function returns an executor which provides an
    # `Executor#forward()` method for evaluation.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b # => "<Symbol broadcast_add>"
    # e = c.bind(args: {"a" => MXNet::NDArray.ones([2, 3]), "b" => MXNet::NDArray.ones([2, 3])}, ctx: MXNet.cpu)
    # e.forward.first # => [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    #                 #    <NDArray 2x3 float32 cpu(0)>
    # ```
    #
    # ### Parameters
    # * *args* (`Array(MXNet::NDArray)` or `Hash(String, MXNet::NDArray)`, default `[]`)
    #   Input arguments.
    #   * If the input type is `Array(MXNet::NDArray)`, the order should be same as the order returned by `#list_arguments`.
    #   * If the input type is `Hash(String, MXNet::NDArray)`, the arguments map to those returned by `#list_arguments`.
    # * *ctx* (`Context`, default is current context)
    #   The device context the executor is to evaluate on.
    #
    def bind(args : Array(MXNet::NDArray) | Hash(String, MXNet::NDArray) = [] of MXNet::NDArray, ctx : Context = MXNet::Context.current)
      arguments = list_arguments
      if args.is_a?(Array(MXNet::NDArray))
        if arguments.size != args.size
          raise SymbolException.new("wrong number of arguments (expected #{arguments.size}, given #{args.size})")
        end
      elsif args.is_a?(Hash(String, MXNet::NDArray))
        args = arguments.map { |a| args[a]?.as(MXNet::NDArray | Nil) }.compact
        if arguments.size != args.size
          raise SymbolException.new("wrong number of arguments (expected #{arguments.size}, matched #{args.size})")
        end
      end

      arg_grad_store = Pointer(MXNet::Internal::LibMXNet::NDArrayHandle).malloc(args.size)
      grad_req_type = Pointer(UInt32).malloc(args.size, 1_u32)

      MXNet::Internal.libcall(
        MXExecutorBindEX,
        @handle,
        *ctx.device,
        0, [] of UInt8*, [] of Int32, [] of Int32,
        args.size, args.map(&.handle), arg_grad_store, grad_req_type,
        0, [] of MXNet::Internal::LibMXNet::NDArrayHandle,
        nil,
        out exec_handle
      )
      MXNet::Executor.new(exec_handle)
    end

    # Evaluates a symbol given arguments.
    #
    # The `#eval` method combines a call to `#bind` (which returns an
    # `Executor`) with a call to `Executor#forward`. For the common
    # use case, where you might repeatedly evaluate with the same
    # arguments, `#eval` is slow. In that case, you should call `#bind`
    # once and then repeatedly call `Executor#forward`. This function
    # allows simpler syntax for less cumbersome introspection.
    #
    # Returns an array of `MXNet::NDArray` corresponding to the values
    # taken by each symbol when evaluated on the given arguments. When
    # called on a single symbol (not a group), the result will be an
    # array with one element.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b # => "<Symbol broadcast_add>"
    # c.eval(a: MXNet::NDArray.ones([2, 3]), b: MXNet::NDArray.ones([2, 3])) # => [<NDArray 2x3 int32 @cpu(0)>]
    # c.eval(MXNet::NDArray.ones([2, 3]), MXNet::NDArray.ones([2, 3])) # => [<NDArray 2x3 int32 @cpu(0)>]
    # ```
    #
    # ### Parameters
    # * *ctx* (`Context`, default is current context)
    #   The device context the executor is to evaluate on.
    # * *ndargs* (`MXNet::NDArray`)
    #   Input arguments. All the arguments must be provided.
    #
    def eval(*ndargs : MXNet::NDArray, ctx : Context = MXNet::Context.current)
      args = ndargs.to_a
      bind(args: args, ctx: ctx).forward
    end

    # ditto
    def eval(ctx : Context = MXNet::Context.current, **ndargs : MXNet::NDArray)
      args = ndargs.map { |k, v| {k.to_s, v} }.to_h
      bind(ctx: ctx, args: args).forward
    end

    # ditto
    def eval(ctx : Context = MXNet::Context.current)
      bind(ctx: ctx).forward
    end

    macro arithmetic(op, array_mod, scalar_mod)
      def {{ op.id }}(other : self | Number)
        if other.is_a?(self)
          Symbol::{{ array_mod }}(self, other)
        else
          Symbol::{{ scalar_mod }}(self, scalar: other)
        end
      end
    end

    # Performs element-wise addition (without broadcasting).
    arithmetic(:+, Internal._plus, Internal._plus_scalar)

    # Performs element-wise subtraction (without broadcasting).
    arithmetic(:-, Internal._minus, Internal._minus_scalar)

    # Performs element-wise multiplication (without broadcasting).
    arithmetic(:*, Internal._mul, Internal._mul_scalar)

    # Performs element-wise division (without broadcasting).
    arithmetic(:/, Internal._div, Internal._div_scalar)

    # Returns the result of the first array elements raised to powers
    # from the second array (or scalar), element-wise with broadcasting.
    arithmetic(:**, Internal._power, Internal._power_scalar)

    # Reshapes the input array.
    #
    # Returns a copy of the array with a new shape without altering any data.
    #
    # ```
    # MXNet::NDArray.array([1, 2, 3, 4]).reshape(shape: [2, 2]) # => MXNet::NDArray.array([[1, 2], [3, 4]])
    # ```
    #
    # Some dimensions of the shape can take special values from the
    # set `{0, -1, -2, -3, -4}`. The significance of each is explained
    # below:
    #
    # Given `a = MXNet::NDArray.zeros([2, 3, 4])`
    #
    # * `0` copies this dimension from the input to the output shape:
    #     MXNet::Symbol.var("a").reshape([4, 0, 2]).eval(a).first.shape # => [4, 3, 2]
    #     MXNet::Symbol.var("a").reshape([2, 0, 0]).eval(a).first.shape # => [2, 3, 4]
    # * `-1` infers the dimension of the output shape by using the
    #   remainder of the input dimensions, keeping the size of the
    #   new array the same as that of the input array. At most one
    #   dimension can be `-1`:
    #     MXNet::Symbol.var("a").reshape([6, 1, -1]).eval(a).first.shape # => [6, 1, 4]
    #     MXNet::Symbol.var("a").reshape([3, -1, 8]).eval(a).first.shape # => [3, 1, 8]
    #     MXNet::Symbol.var("a").reshape([-1]).eval(a).first.shape # => [24]
    # * `-2` copies all/the remainder of the input dimensions to the
    #   output shape:
    #     MXNet::Symbol.var("a").reshape([-2]).eval(a).first.shape # => [2, 3, 4]
    #     MXNet::Symbol.var("a").reshape([2, -2]).eval(a).first.shape # => [2, 3, 4]
    #     MXNet::Symbol.var("a").reshape([-2, 1, 1]).eval(a).first.shape # => [2, 3, 4, 1, 1]
    # * `-3` uses the product of two consecutive dimensions of the
    #   input shape as the output dimension:
    #     MXNet::Symbol.var("a").reshape([-3, 4]).eval(a).first.shape # => [6, 4]
    #     MXNet::Symbol.var("a").reshape([-3, -3]).eval(a).first.shape # => [6, 20]
    #     MXNet::Symbol.var("a").reshape([0, -3]).eval(a).first.shape # => [2, 12]
    #     MXNet::Symbol.var("a").reshape([-3, -2]).eval(a).first.shape # => [6, 4]
    # * `-4` splits one dimension of the input into the two dimensions
    #   passed subsequent to `-4` (which can contain `-1`):
    #     MXNet::Symbol.var("a").reshape([-4, 1, 2, -2]).eval(a).first.shape # => [1, 2, 3, 4]
    #     MXNet::Symbol.var("a").reshape([2, -4, -1, 3, -2]).eval(a).first.shape # => [2, 1, 3, 4]
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The target shape.
    # * *reverse* (`Bool`, optional, default `false`)
    #   If `true` then the special values are inferred from right to left.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def reshape(shape : Int | Array(Int), **kwargs)
      Ops._reshape(self, **kwargs.merge({shape: shape}))
    end

    # Flattens the input array into a 2-D array by collapsing the
    # higher dimensions.
    #
    # For an input array with shape `(d1, d2, ..., dk)`, `#flatten`
    # reshapes the input array into an output array of shape
    # `(d1, d2 * ... * dk)`.
    #
    # Note that the bahavior of this function is different from
    # `Array#flatten`, which behaves similar to `#reshape([-1])`.
    #
    # ```
    # x = MXNet::NDArray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    # MXNet::Symbol.var("x").flatten.eval(x).first.shape # => [2, 6]
    # ```
    #
    # ### Parameters
    # * *out* (`NDArray`, optional)
    #   The output array.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def flatten(**kwargs)
      Ops._flatten(self, **kwargs)
    end

    def to_s(io)
      io << "<Symbol"
      io << " " << name
      io << ">"
    end

    # :nodoc:
    def finalize
      MXNet::Internal.libcall(MXSymbolFree, @handle)
    end

    # Creates a symbolic variable with the specified name.
    #
    # ### Parameters
    # * *name* (`String`)
    #   Variable name.
    #
    def self.var(name : String)
      MXNet::Internal.libcall(MXSymbolCreateVariable, name, out handle)
      new(handle)
    end

    # Draw random samples from a uniform distribution.
    #
    # Samples are uniformly distributed over the half-open interval
    # [low, high) (includes low, but excludes high).
    #
    # ```
    # MXNet::Symbol.random_uniform(0.0, 1.0, [2, 2]).eval.first # => MXNet::NDArray.array([[0.60276335, 0.85794562], [0.54488319, 0.84725171]])
    # ```
    #
    # ### Parameters
    # * *low* (`Float`, default = 0.0)
    #   Lower bound of the distribution.
    # * *high* (`Float`, default = 1.0)
    #   Upper bound of the distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output. The default is `:float32` if the
    #   data type can't be inferred.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_uniform(low : Number = 0.0, high : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[low.class]? || "None"
      Symbol::Internal._random_uniform(**kwargs.merge({low: low, high: high, shape: shape, dtype: dtype, ctx: ctx}))
    end

    # Draw random samples from a normal (Gaussian) distribution.
    #
    # Samples are distributed according to a normal distribution
    # parametrized by loc (mean) and scale (standard deviation).
    #
    # ```
    # MXNet::Symbol.random_normal(0.0, 1.0, [2, 2]).eval.first # => MXNet::NDArray.array([[1.89171135, -1.16881478], [-1.23474145, 1.55807114]])
    # ```
    #
    # ### Parameters
    # * *loc* (`Float`, default = 0.0)
    #   Mean of the distribution.
    # * *scale* (`Float`, default = 1.0)
    #   Standard deviation of the distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output. The default is `:float32` if the
    #   data type can't be inferred.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_normal(loc : Number = 0.0, scale : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[loc.class]? || "None"
      Symbol::Internal._random_normal(**kwargs.merge({loc: loc, scale: scale, shape: shape, dtype: dtype, ctx: ctx}))
    end

    # Draw random samples from a Poisson distribution.
    #
    # Samples are distributed according to a Poisson distribution
    # parametrized by lambda (rate). Samples will always be returned
    # as a floating point data type.
    #
    # ```
    # MXNet::Symbol.random_poisson(4.0, [2, 2]).eval.first # => MXNet::NDArray.array([[5.0, 2.0], [4.0, 6.0]])
    # ```
    #
    # ### Parameters
    # * *lam* (`Float`, default = 1.0)
    #   Lambda parameter (rate) of the Poisson distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output. The default is `:float32` if the
    #   data type can't be inferred.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_poisson(lam : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[lam.class]? || "None"
      Symbol::Internal._random_poisson(**kwargs.merge({lam: lam, shape: shape, dtype: dtype, ctx: ctx}))
    end

    # Draw random samples from an exponential distribution.
    #
    # Samples are distributed according to an exponential distribution
    # parametrized by lambda (rate).
    #
    # ```
    # MXNet::Symbol.random_exponential(4.0, [2, 2]).eval.first # => MXNet::NDArray.array([[0.0097189 , 0.08999364], [0.04146638, 0.31715935]])
    # ```
    #
    # ### Parameters
    # * *lam* (`Float`, default = 1.0)
    #   Lambda parameter (rate) of the exponential distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output. The default is `:float32` if the
    #   data type can't be inferred.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_exponential(lam : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[lam.class]? || "None"
      Symbol::Internal._random_exponential(**kwargs.merge({lam: lam, shape: shape, dtype: dtype, ctx: ctx}))
    end

    # Draw random samples from a gamma distribution.
    #
    # Samples are distributed according to a gamma distribution
    # parametrized by alpha (shape) and beta (scale).
    #
    # ```
    # MXNet::Symbol.random_exponential(9.0, 0.5, [2, 2]).eval.first # => MXNet::NDArray.array([[7.10486984, 3.37695289], [3.91697288, 3.65933681]])
    # ```
    #
    # ### Parameters
    # * *alpha* (`Float`, default = 1.0)
    #   Alpha parameter (shape) of the gamma distribution.
    # * *beta* (`Float`, default = 1.0)
    #   Beta parameter (scale) of the gamma distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output. The default is `:float32` if the
    #   data type can't be inferred.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_gamma(alpha : Number = 1.0, beta : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[alpha.class]? || "None"
      Symbol::Internal._random_gamma(**kwargs.merge({alpha: alpha, beta: beta, shape: shape, dtype: dtype, ctx: ctx}))
    end

    # TODO: cache op handles
    def self.create_symbol(op, *args, **kwargs)
      op = op.to_s
      args = args.to_a
      kwargs = kwargs.to_h

      name = kwargs.delete(:name)
      name = name.is_a?(String) ? name : nil

      MXNet::Internal.libcall(
        NNGetOpHandle,
        op,
        out op_handle
      )
      MXNet::Internal.libcall(
        MXSymbolCreateAtomicSymbol,
        op_handle,
        kwargs.size,
        kwargs.keys.map(&.to_s.as(String).to_unsafe),
        kwargs.values.map(&.to_s.as(String).to_unsafe),
        out sym_handle)
      sym = new(sym_handle)
      MXNet::Internal.libcall(
        NNSymbolCompose,
        sym_handle,
        MXNet::Name::Manager.current.get(name, op.downcase),
        args.size,
        nil,
        args.to_a.map(&.handle.as(SymbolHandle)))
      sym
    end
  end
end

struct Number
  # Performs element-wise addition.
  def +(other : MXNet::Symbol)
    MXNet::Symbol::Internal._plus_scalar(other, scalar: self)
  end

  # Performs element-wise subtraction.
  def -(other : MXNet::Symbol)
    MXNet::Symbol::Internal._rminus_scalar(other, scalar: self)
  end

  # Performs element-wise multiplication.
  def *(other : MXNet::Symbol)
    MXNet::Symbol::Internal._mul_scalar(other, scalar: self)
  end

  # Performs element-wise division.
  def /(other : MXNet::Symbol)
    MXNet::Symbol::Internal._rdiv_scalar(other, scalar: self)
  end

  # Returns the result of this number raised to powers from the array,
  # element-wise with broadcasting.
  def **(other : MXNet::Symbol)
    MXNet::Symbol::Internal._rpower_scalar(other, scalar: self)
  end
end
