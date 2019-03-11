module MXNet
  class NDArrayException < Exception
  end

  class NDArray
    extend MXNet::Operations

    alias NDArrayHandle = MXNet::Internal::LibMXNet::NDArrayHandle

    @handle : NDArrayHandle

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

    # :nodoc:
    INFERRED_TYPES = {
      Array(Float32) => 0,
      Array(Float64) => 1,
      Array(UInt8) => 3,
      Array(Int32) => 4,
      Array(Int8) => 5,
      Array(Int64) => 6
    }

    def initialize(handle)
      @handle = handle
    end

    # :nodoc:
    def handle
      @handle
    end

    def shape
      MXNet::Internal.libcall(MXNDArrayGetShape, @handle, out dim, out pdata)
      pdata.to_slice(dim).to_a
    end

    def context
      MXNet::Internal.libcall(MXNDArrayGetContext, @handle, out dev_type, out dev_id)
      Context.new(dev_type, dev_id)
    end

    def dtype
      MXNet::Internal.libcall(MXNDArrayGetDType, @handle, out dtype)
      DT2T[dtype]
    end

    def copy_to(other : self)
      NDArray.imperative_invoke("_copyto", self, out: other)
    end

    def as_type(dtype : ::Symbol)
      return self if dtype == self.dtype
      NDArray.empty(shape, dtype: dtype).tap do |res|
        copy_to(res)
      end
    end

    # Returns gradient buffer attached to this array.
    #
    def grad
      MXNet::Internal.libcall(MXNDArrayGetGrad, @handle, out grad_handle)
      raise NDArrayException.new("no gradient is attached") if grad_handle.null?
      NDArray.new(grad_handle)
    end

    # Attach a gradient buffer to this array, so that `#backward`
    # can compute gradient with respect to it.
    #
    # ### Parameters
    # * *grad_req* (`Symbol`, default `:write`)
    #   * `:write`: gradient will be overwritten on every backward pass
    #   * `:add`: gradient will be added to existing value on every backward pass
    #   * `:null`: do not compute gradient
    #
    def attach_grad(grad_req = :write)
      MXNet::Autograd.mark_variables(self, Ops._zeros_like(self).first)
      self
    end

    # Compute the gradients of this array with respect to previously
    # marked variables.
    #
    # ### Parameters
    # * *gradient* (`MXNet::NDArray`, optional)
    #   Gradient with respect to this array.
    # * *retain_graph* (`Bool`, default false)
    #   Whether to keep computation graph to differentiate again,
    #   instead of clearing history and releasing memory.
    # * *train_mode* (`Bool`, default true)
    #   Whether the backward pass is in training or predicting mode.
    #
    def backward(gradient = nil, retain_graph = false, train_mode = true)
      MXNet::Autograd.backward(self, gradient, retain_graph, train_mode)
      self
    end

    private macro arithmetic(op, array_mod, scalar_mod)
      def {{ op.id }}(other : self | Number)
        if other.is_a?(self)
          NDArray::{{ array_mod }}(self, other).first
        else
          NDArray::{{ scalar_mod }}(self, scalar: other).first
        end
      end
    end

    # Performs element-wise addition with broadcasting.
    arithmetic(:+, Ops._broadcast_add, Internal._plus_scalar)

    # Performs element-wise subtraction with broadcasting.
    arithmetic(:-, Ops._broadcast_sub, Internal._minus_scalar)

    # Performs element-wise multiplication with broadcasting.
    arithmetic(:*, Ops._broadcast_mul, Internal._mul_scalar)

    # Performs element-wise division with broadcasting.
    arithmetic(:/, Ops._broadcast_div, Internal._div_scalar)

    # Returns the result of the first array elements raised to powers
    # from the second array (or scalar), element-wise with broadcasting.
    arithmetic(:**, Ops._broadcast_power, Internal._power_scalar)

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
    # * `0` copies this dimension from the input to the output shape:
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([4, 0, 2]).shape # => [4, 3, 2]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([2, 0, 0]).shape # => [2, 3, 4]
    # * `-1` infers the dimension of the output shape by using the
    #   remainder of the input dimensions, keeping the size of the
    #   new array the same as that of the input array. At most one
    #   dimension can be `-1`:
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([6, 1, -1]).shape # => [6, 1, 4]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([3, -1, 8]).shape # => [3, 1, 8]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([-1]).shape # => [24]
    # * `-2` copies all/the remainder of the input dimensions to the
    #   output shape:
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([-2]).shape # => [2, 3, 4]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([2, -2]).shape # => [2, 3, 4]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([-2, 1, 1]).shape # => [2, 3, 4, 1, 1]
    # * `-3` uses the product of two consecutive dimensions of the
    #   input shape as the output dimension:
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([-3, 4]).shape # => [6, 4]
    #     MXNet::NDArray.zeros([2, 3, 4, 5]).reshape([-3, -3]).shape # => [6, 20]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([0, -3]).shape # => [2, 12]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([-3, -2]).shape # => [6, 4]
    # * `-4` splits one dimension of the input into the two dimensions
    #   passed subsequent to `-4` (which can contain `-1`):
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([-4, 1, 2, -2]).shape # => [1, 2, 3, 4]
    #     MXNet::NDArray.zeros([2, 3, 4]).reshape([2, -4, -1, 3, -2]).shape # => [2, 1, 3, 4]
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The target shape.
    # * *reverse* (`Bool`, optional, default `false`)
    #   If `true` then the special values are inferred from right to left.
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def reshape(shape : Int | Array(Int), **kwargs)
      Ops._reshape(self, **kwargs.merge({shape: shape})).first
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
    # x.flatten.shape # => [2, 6]
    # ```
    #
    # ### Parameters
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def flatten(**kwargs)
      Ops._flatten(self, **kwargs).first
    end

    private macro method_missing(call)
      {% if call.name == "[]".id %}
        self.[]({{call.args}})
      {% elsif call.name == "[]=".id %}
        self.[]=({{call.args[0..-2]}}, {{call.args.last}})
      {% else %}
        {% raise "no method matches '#{@type}##{call.name}'" %}
      {% end %}
    end

    private def ranges_and_dims(keys, compact = nil)
      shape = self.shape.map(&.to_i32)
      ranges = keys.map_with_index do |k, i|
        if k.is_a?(Int)
          b = k
          e = k + 1
          {b, e}
        else
          b = k.begin
          e = k.end < 0 ? shape[i] + k.end : k.end
          e = k.excludes_end? ? e : e + 1
          {b, e}
        end
      end
      dims = shape.map_with_index do |s, i|
        if keys[i]?
          if keys[i].is_a?(Int)
            compact ? nil : 1
          else
            ranges[i].last - ranges[i].first
          end
        else
          s
        end
      end.compact
      {ranges, dims}
    end

    # Returns a sliced view of this array.
    #
    # This method assumes the key is `Array` of `Int` or `Range(Int, Int)`.
    # A macro is provided that rewrites a key presented as a variable
    # number of `Int` or `Range(Int, Int)` arguments to array syntax.
    #
    # ### Parameters
    # * *keys* (`Array(Int | Range(Int, Int))`)
    #   Indexing key.
    #
    # Using variable argument syntax:
    #
    # ```
    # a = MXNet::NDArray.array([1, 2, 3, 4])
    # a[1] # => MXNet::NDArray.array([2])
    # a[1...3] # => MXNet::NDArray.array([2, 3])
    # a[1..-2] # => MXNet::NDArray.array([2, 3])
    # b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
    # b[1...3, 1] # => MXNet::NDArray.array([[7, 8], [1, 2]])
    # b[1...3, 1...2] # => MXNet::NDArray.array([[[7, 8]], [[1, 2]]])
    # b[1, 1...2] # => MXNet::NDArray.array([[7, 8]])
    # ```
    #
    def [](keys : Array(Int | Range(Int, Int)))
      ranges, dims = ranges_and_dims(keys, compact: true)
      out = Ops._slice(self, begin: ranges.map(&.first), end: ranges.map(&.last)).first
      dims = dims.size > 0 ? out.reshape(dims) : out
    end

    # Sets sliced view of this array to the specified value.
    #
    # This method assumes the key is `Array` of `Int` or `Range(Int, Int)`.
    # A macro is provided that rewrites a key presented as a variable
    # number of `Int` or `Range(Int, Int)` arguments to array syntax.
    #
    # ### Parameters
    # * *keys* (`Array(Int | Range(Int, Int))`)
    #   Indexing key.
    # * *value* (`Number | MXNet::NDArray)`)
    #   The value to set.
    #
    # Using variable argument syntax:
    #
    # ```
    # a = MXNet::NDArray.array([1, 2, 3, 4])
    # a[1] = 99
    # a # => MXNet::NDArray.array([1, 99, 3, 4])
    # a[1..-2] = 98
    # a # => MXNet::NDArray.array([1, 98, 98, 4])
    # a[1...3] = MXNet::NDArray.array([97, 97])
    # a # => MXNet::NDArray.array([1, 97, 97, 4])
    # b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
    # b[1...3, 1] = 99
    # b # => MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [99, 99]]])
    # b[1...3, 1...2] = MXNet::NDArray.array([[[98, 98]], [[98, 98]]])
    # b # => MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [98, 98]], [[9, 0], [98, 98]]])
    # b[1, 1...2] = MXNet::NDArray.array([[97, 97]])
    # b # => MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [97, 97]], [[9, 0], [98, 98]]])
    # ```
    #
    def []=(keys : Array(Int | Range(Int, Int)), value : Number | self)
      ranges, dims = ranges_and_dims(keys, compact: false)
      if value.is_a?(self)
        Internal._slice_assign(
          self,
          value.reshape(dims),
          begin: ranges.map(&.first),
          end: ranges.map(&.last),
          out: self
        )
      else
        Internal._slice_assign_scalar(
          self,
          begin: ranges.map(&.first),
          end: ranges.map(&.last),
          scalar: value,
          out: self
        )
      end
      value
    end

    # Returns a scalar whose value is copied from this array.
    #
    # The array must have shape `[1]`.
    #
    # ```
    # MXNet::NDArray.zeros([1], dtype: :float64).as_scalar # => 0.0
    # ```
    #
    def as_scalar
      unless (ar = raw).size == 1
        raise NDArrayException.new("the array is not scalar")
      end
      ar[0]
    end

    # Returns an `Array` with values copied from this array.
    #
    # Only works for 1-dimensional arrays (`shape.size == 1`).
    #
    # ```
    # MXNet::NDArray.zeros([4], dtype: :float64).to_a # => [0.0, 0.0, 0.0, 0.0]
    # ```
    #
    def to_a
      unless shape.size == 1
        raise NDArrayException.new("the array must have only 1 dimension")
      end
      raw
    end

    def to_s(io)
      data = ["[]"]
      if shape.product > 0
        data = raw
        shape.reverse.each do |dim|
          data = data.in_groups_of(dim).map { |group| "[#{group.join(", ")}]" }
        end
      end
      data.each { |line| io << line << "\n" }
      io << "<NDArray"
      io << " " << shape.join("x") if shape.size > 0
      io << " " << dtype
      io << " " << context
      io << ">"
    end

    protected def raw
      case dtype
      when :float32
        Array(Float32).new(shape.product, 0.0).tap do |array|
          MXNet::Internal.libcall(MXNDArraySyncCopyToCPU, @handle, array, array.size)
        end
      when :float64
        Array(Float64).new(shape.product, 0.0).tap do |array|
          MXNet::Internal.libcall(MXNDArraySyncCopyToCPU, @handle, array, array.size)
        end
      when :uint8
        Array(UInt8).new(shape.product, 0).tap do |array|
          MXNet::Internal.libcall(MXNDArraySyncCopyToCPU, @handle, array, array.size)
        end
      when :int32
        Array(Int32).new(shape.product, 0).tap do |array|
          MXNet::Internal.libcall(MXNDArraySyncCopyToCPU, @handle, array, array.size)
        end
      when :int8
        Array(Int8).new(shape.product, 0).tap do |array|
          MXNet::Internal.libcall(MXNDArraySyncCopyToCPU, @handle, array, array.size)
        end
      when :int64
        Array(Int64).new(shape.product, 0).tap do |array|
          MXNet::Internal.libcall(MXNDArraySyncCopyToCPU, @handle, array, array.size)
        end
      else
        raise "this should never happen"
      end
    end

    # :nodoc:
    def finalize
      MXNet::Internal.libcall(MXNDArrayFree, @handle)
    end

    # Returns an MXNet array filled with all zeros, with the given
    # shape and type.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the array.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output array. The default is `:float32`.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.zeros(shape : Int | Array(Int), dtype : ::Symbol = :float32, ctx : Context = Context.current, **kwargs)
      NDArray::Internal._zeros(**kwargs.merge({shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Returns an MXNet array filled with all ones, with the given
    # shape and type.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the array.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output array. The default is `:float32`.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.ones(shape : Int | Array(Int), dtype : ::Symbol = :float32, ctx : Context = Context.current, **kwargs)
      NDArray::Internal._ones(**kwargs.merge({shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Draw random samples from a uniform distribution.
    #
    # Samples are uniformly distributed over the half-open interval
    # [low, high) (includes low, but excludes high).
    #
    # ```
    # MXNet::NDArray.random_uniform(0.0, 1.0, [2, 2]) # => MXNet::NDArray.array([[0.60276335, 0.85794562], [0.54488319, 0.84725171]])
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
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_uniform(low : Number = 0.0, high : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[low.class]? || "None"
      NDArray::Internal._random_uniform(**kwargs.merge({low: low, high: high, shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Draw random samples from a normal (Gaussian) distribution.
    #
    # Samples are distributed according to a normal distribution
    # parametrized by loc (mean) and scale (standard deviation).
    #
    # ```
    # MXNet::NDArray.random_normal(0.0, 1.0, [2, 2]) # => MXNet::NDArray.array([[1.89171135, -1.16881478], [-1.23474145, 1.55807114]])
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
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_normal(loc : Number = 0.0, scale : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[loc.class]? || "None"
      NDArray::Internal._random_normal(**kwargs.merge({loc: loc, scale: scale, shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Draw random samples from a Poisson distribution.
    #
    # Samples are distributed according to a Poisson distribution
    # parametrized by lambda (rate). Samples will always be returned
    # as a floating point data type.
    #
    # ```
    # MXNet::NDArray.random_poisson(4.0, [2, 2]) # => MXNet::NDArray.array([[5.0, 2.0], [4.0, 6.0]])
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
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_poisson(lam : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[lam.class]? || "None"
      NDArray::Internal._random_poisson(**kwargs.merge({lam: lam, shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Draw random samples from an exponential distribution.
    #
    # Samples are distributed according to an exponential distribution
    # parametrized by lambda (rate).
    #
    # ```
    # MXNet::NDArray.random_exponential(4.0, [2, 2]) # => MXNet::NDArray.array([[0.0097189 , 0.08999364], [0.04146638, 0.31715935]])
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
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_exponential(lam : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[lam.class]? || "None"
      NDArray::Internal._random_exponential(**kwargs.merge({lam: lam, shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Draw random samples from a gamma distribution.
    #
    # Samples are distributed according to a gamma distribution
    # parametrized by alpha (shape) and beta (scale).
    #
    # ```
    # MXNet::NDArray.random_exponential(9.0, 0.5, [2, 2]) # => MXNet::NDArray.array([[7.10486984, 3.37695289], [3.91697288, 3.65933681]])
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
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_gamma(alpha : Number = 1.0, beta : Number = 1.0, shape : Int | Array(Int) = 1, dtype : ::Symbol? = nil, ctx = MXNet::Context.current, **kwargs)
      shape = shape.is_a?(Int32) ? [shape] : shape
      dtype ||= {Float32 => :float32, Float64 => :float64}[alpha.class]? || "None"
      NDArray::Internal._random_gamma(**kwargs.merge({alpha: alpha, beta: beta, shape: shape, dtype: dtype, ctx: ctx})).first
    end

    # Returns an MXNet array of given shape and type, without initializing entries.
    #
    # ### Parameters
    # * *shape* (`UInt32` or `Array(UInt32)`)
    #   The shape of the empty array.
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output array. The default is `:float32`.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    #
    def self.empty(shape : UInt32 | Array(UInt32), dtype : ::Symbol = :float32, ctx : Context = Context.current)
      shape = shape.is_a?(UInt32) ? [shape] : shape
      dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      MXNet::Internal.libcall(MXNDArrayCreateEx, shape, shape.size, *ctx.device, 0, dtype, out handle)
      new(handle)
    end

    # Creates an MXNet array from any enumerable object.
    #
    # ### Parameters
    # * *source* (`Enumerable(T)`)
    #   Any enumerable object, or nested objects.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *dtype* (`Symbol`, optional)
    #   The data type of the output array. If unspecified, the type is
    #   inferred from the source type.
    #
    def self.array(source : Enumerable(T), ctx : Context | Nil = nil, dtype : ::Symbol | Nil = nil) forall T
      source = source.to_a
      shape_and_type = infer_shape_and_type(source)
      inferred_shape = shape_and_type.map(&.first)
      inferred_type = shape_and_type.last.last
      elements = source.flatten

      ctx ||= Context.current

      if dtype
        dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      end
      if inferred_type
        inferred_type = INFERRED_TYPES[inferred_type]? || raise MXNet::NDArrayException.new("type is unsupported: #{inferred_type}")
      else
        raise MXNet::NDArrayException.new("type can't be inferred")
      end

      MXNet::Internal.libcall(MXNDArrayCreateEx, inferred_shape, inferred_shape.size, *ctx.device, 0, inferred_type, out handle)
      MXNet::Internal.libcall(MXNDArraySyncCopyFromCPU, handle, elements, elements.size)

      if dtype && dtype != inferred_type
        empty(inferred_shape, ctx: ctx, dtype: DT2T[dtype]).tap do |res|
          new(handle).copy_to(res)
        end
      else
        new(handle)
      end
    end

    private def self.infer_shape_and_type(array : T) forall T
      # compiler guard to narrow type to array
      raise "this will never happen" unless array.is_a?(Array)
      nested = array.map { |item| item.is_a?(Array) && item.size.to_u32 }
      if nested.none?
        [{array.size.to_u32, T}]
      elsif nested.all?
        unless nested.map { |item| item.is_a?(Bool) ? 0_u32 : item }.sort.uniq.size > 1
          shape = [{array.size.to_u32, nil}] of Tuple(UInt32, T.class | Nil)
          shape + infer_shape_and_type(array.sample)
        else
          raise MXNet::NDArrayException.new("inconsistent dimensions: #{array}")
        end
      else
        raise MXNet::NDArrayException.new("inconsistent nesting: #{array}")
      end
    end

    # TODO: cache op handles
    def self.imperative_invoke(op, *ndargs, **kwargs)
      op = op.to_s
      ndargs = ndargs.to_a
      kwargs = kwargs.to_h
      num_outputs = 0
      outputs = Pointer(NDArrayHandle).null

      out = nil
      if kwargs.has_key?(:out)
        out = kwargs.delete(:out)
        if out.is_a?(NDArray)
          num_outputs = 1
          outputs = Pointer(NDArrayHandle).malloc(1)
          outputs[0] = out.handle
        else
          raise MXNet::NDArrayException.new("out is invalid (must be NDArray): #{out}")
        end
      end

      MXNet::Internal.libcall(
        NNGetOpHandle,
        op,
        out op_handle
      )
      MXNet::Internal.libcall(
        MXImperativeInvoke,
        op_handle,
        ndargs.size,
        ndargs.map(&.handle.as(NDArrayHandle)),
        pointerof(num_outputs),
        pointerof(outputs),
        kwargs.size,
        kwargs.keys.map(&.to_s.as(String).to_unsafe),
        kwargs.values.map(&.to_s.as(String).to_unsafe)
      )
      out ? [out] : num_outputs.times.map { |i| NDArray.new(outputs[i]) }
    end
  end
end

struct Number
  # Performs element-wise addition.
  def +(other : MXNet::NDArray)
    MXNet::NDArray::Internal._plus_scalar(other, scalar: self).first
  end

  # Performs element-wise subtraction.
  def -(other : MXNet::NDArray)
    MXNet::NDArray::Internal._rminus_scalar(other, scalar: self).first
  end

  # Performs element-wise multiplication.
  def *(other : MXNet::NDArray)
    MXNet::NDArray::Internal._mul_scalar(other, scalar: self).first
  end

  # Performs element-wise division.
  def /(other : MXNet::NDArray)
    MXNet::NDArray::Internal._rdiv_scalar(other, scalar: self).first
  end

  # Returns the result of this number raised to powers from the array,
  # element-wise with broadcasting.
  def **(other : MXNet::NDArray)
    MXNet::NDArray::Internal._rpower_scalar(other, scalar: self).first
  end
end
