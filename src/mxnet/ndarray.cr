module MXNet
  class NDArrayException < Exception
  end

  # The `NDArray` API provides imperative tensor operations on
  # CPU/GPU. An `NDArray` represents a multi-dimensional, fixed-size
  # homogeneous array.
  #
  # ```
  # x = MXNet::NDArray.array([[1, 2, 3], [4, 5, 6]], dtype: :float32)
  # x.shape      # [2, 3]
  # y = x + MXNet::NDArray.ones(x.shape) * 3
  # puts y       # [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  #              # <NDArray 2x3 float32 cpu(0)>
  # z = y.as_in_context(MXNet.gpu(0))
  # puts z       # [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  #              # <NDArray 2x3 float32 gpu(0)>
  # ```
  #
  # A detailed (albeit in Python) tutorial is available at
  # [NDArray - Imperative tensor operations on CPU/GPU](https://mxnet.incubator.apache.org/versions/master/tutorials/basic/ndarray.html).
  #
  # Note: `NDArray` provides almost the same routines as `Symbol`.
  # Most routines between these two packages share source code. But
  # `NDArray` differs from `Symbol` in few aspects:
  #
  # * `NDArray` adopts an imperative programming style -- namely
  #   expressions are executed step-by-step so that the results can be
  #   obtained immediately, whereas `Symbol` adopts a declarative
  #   style.
  # * Most binary operators in `NDArray` such as `+` and `>` have
  #   broadcasting enabled by default.
  #
  class NDArray < Base
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

    # :nodoc:
    protected def initialize(@handle)
    end

    # :nodoc:
    def handle
      @handle
    end

    def shape
      MXNet::Internal.libcall(MXNDArrayGetShape, @handle, out dim, out pdata)
      pdata.to_slice(dim).to_a.map(&.to_i32)
    end

    def context
      MXNet::Internal.libcall(MXNDArrayGetContext, @handle, out dev_type, out dev_id)
      Context.new(dev_type, dev_id)
    end

    def dtype
      MXNet::Internal.libcall(MXNDArrayGetDType, @handle, out dtype)
      DT2T[dtype]
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
    # * *grad_req* (`::Symbol`, default = `:write`)
    #   * `:write`: gradient will be overwritten on every backward pass
    #   * `:add`: gradient will be added to existing value on every backward pass
    #   * `:null`: do not compute gradient
    #
    def attach_grad(grad_req = :write)
      MXNet::Autograd.mark_variables(self, Ops._zeros_like(self))
      self
    end

    # Compute the gradients of this array with respect to previously
    # marked variables.
    #
    # ### Parameters
    # * *gradient* (`MXNet::NDArray`, optional)
    #   Gradient with respect to this array.
    # * *retain_graph* (`Bool`, default = `false`)
    #   Whether to keep computation graph to differentiate again,
    #   instead of clearing history and releasing memory.
    # * *train_mode* (`Bool`, default = `true`)
    #   Whether the backward pass is in training or predicting mode.
    #
    def backward(gradient = nil, retain_graph = false, train_mode = true)
      MXNet::Autograd.backward(self, gradient, retain_graph, train_mode)
      self
    end

    # Returns element-wise sum of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs + rhs`. Equivalent to `.broadcast_add` and
    # `.broadcast_plus` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be added.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be added.
    #
    bifunc_helper(
      add,
      lhs, rhs,
      Ops._broadcast_add,
      :+,
      Internal._plus_scalar,
      Internal._plus_scalar
    )

    # Returns element-wise difference of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs - rhs`. Equivalent to `.broadcast_sub` and
    # `.broadcast_minus` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be subtracted.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be subtracted.
    #
    bifunc_helper(
      subtract,
      lhs, rhs,
      Ops._broadcast_sub,
      :-,
      Internal._rminus_scalar,
      Internal._minus_scalar
    )

    # Returns element-wise product of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs * rhs`. Equivalent to `.broadcast_mul` for
    # `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be multiplied.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be multiplied.
    #
    bifunc_helper(
      multiply,
      lhs, rhs,
      Ops._broadcast_mul,
      :*,
      Internal._mul_scalar,
      Internal._mul_scalar
    )

    # Returns element-wise division of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs / rhs`. Equivalent to `.broadcast_div` for
    # `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be divided.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be divided.
    #
    bifunc_helper(
      divide,
      lhs, rhs,
      Ops._broadcast_div,
      :/,
      Internal._rdiv_scalar,
      Internal._div_scalar
    )

    # Returns result of first array elements raised to powers from
    # second array, element-wise.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `base ** exp`. Equivalent to `.broadcast_power`
    # for `NDArray` arguments.
    #
    # ### Parameters
    # * *base* (`NDArray` or `Number`)
    #   The base value.
    # * *exp* (`NDArray` or `Number`)
    #   The exponent value.
    #
    bifunc_helper(
      power,
      base, exp,
      Ops._broadcast_power,
      :**,
      Internal._rpower_scalar,
      Internal._power_scalar
    )

    # Returns element-wise maximum of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `.broadcast_maximum` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      maximum,
      lhs, rhs,
      Ops._broadcast_maximum,
      lhs > rhs ? lhs : rhs,
      Internal._maximum_scalar,
      Internal._maximum_scalar
    )

    # Returns element-wise minimum of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `.broadcast_minimum` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      minimum,
      lhs, rhs,
      Ops._broadcast_minimum,
      lhs < rhs ? lhs : rhs,
      Internal._minimum_scalar,
      Internal._minimum_scalar
    )

    # Returns the result of element-wise equal to (`==`) comparison
    # operation.
    #
    # For each element in input arrays, return 1 (true) if
    # corresponding elements are same, otherwise return 0 (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs == rhs`. Equivalent to `.broadcast_equal` for
    # `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      equal,
      lhs, rhs,
      Ops._broadcast_equal,
      lhs == rhs ? 1.0 : 0.0,
      Internal._equal_scalar,
      Internal._equal_scalar
    )

    # Returns the result of element-wise not equal to (`!=`)
    # comparison operation.
    #
    # For each element in input arrays, return 1 (true) if
    # corresponding elements are different, otherwise return 0
    # (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs != rhs`. Equivalent to `.broadcast_not_equal`
    # for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      not_equal,
      lhs, rhs,
      Ops._broadcast_not_equal,
      lhs == rhs ? 0.0 : 1.0,
      Internal._not_equal_scalar,
      Internal._not_equal_scalar
    )

    # Returns the result of element-wise greater than (`>`) comparison
    # operation.
    #
    # For each element in input arrays, return 1 (true) if *lhs*
    # element is greater than corresponding *rhs* element, otherwise
    # return 0 (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs > rhs`. Equivalent to `.broadcast_greater`
    # for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      greater,
      lhs, rhs,
      Ops._broadcast_greater,
      lhs > rhs ? 1.0 : 0.0,
      Internal._lesser_scalar,
      Internal._greater_scalar
    )

    # Returns the result of element-wise greater than or equal to
    # (`>=`) comparison operation.
    #
    # For each element in input arrays, return 1 (true) if *lhs*
    # element is greater than or equal to *rhs* element, otherwise
    # return 0 (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs >= rhs`. Equivalent to
    # `.broadcast_greater_equal` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      greater_equal,
      lhs, rhs,
      Ops._broadcast_greater_equal,
      lhs >= rhs ? 1.0 : 0.0,
      Internal._lesser_equal_scalar,
      Internal._greater_equal_scalar
    )

    # Returns the result of element-wise less than (`<`) comparison
    # operation.
    #
    # For each element in input arrays, return 1 (true) if *lhs*
    # element is less than corresponding *rhs* element, otherwise
    # return 0 (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs < rhs`. Equivalent to `.broadcast_lesser`
    # for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      lesser,
      lhs, rhs,
      Ops._broadcast_lesser,
      lhs < rhs ? 1.0 : 0.0,
      Internal._greater_scalar,
      Internal._lesser_scalar
    )

    # Returns the result of element-wise less than or equal to (`<=`)
    # comparison operation.
    #
    # For each element in input arrays, return 1 (true) if *lhs*
    # element is less than or equal to *rhs* element, otherwise return
    # 0 (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs <= rhs`. Equivalent to
    # `.broadcast_lesser_equal` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      lesser_equal,
      lhs, rhs,
      Ops._broadcast_lesser_equal,
      lhs <= rhs ? 1.0 : 0.0,
      Internal._greater_equal_scalar,
      Internal._lesser_equal_scalar
    )

    # Performs element-wise addition with broadcasting.
    def +(other)
      self.class.add(self, other)
    end

    # Performs element-wise subtraction with broadcasting.
    def -(other)
      self.class.subtract(self, other)
    end

    # Performs element-wise multiplication with broadcasting.
    def *(other)
      self.class.multiply(self, other)
    end

    # Performs element-wise division with broadcasting.
    def /(other)
      self.class.divide(self, other)
    end

    # Returns the result of the first array elements raised to powers
    # from the second array (or scalar), element-wise with
    # broadcasting.
    def **(other)
      self.class.power(self, other)
    end

    # Performs element-wise equal to (`==`) comparison operation with
    # broadcasting.
    def ==(other)
      self.class.equal(self, other)
    end

    # Performs element-wise not equal to (`!=`) comparison operation
    # with broadcasting.
    def !=(other)
      self.class.not_equal(self, other)
    end

    # Performs element-wise greater than (`>`) comparison operation
    # with broadcasting.
    def >(other)
      self.class.greater(self, other)
    end

    # Performs element-wise greater than or equal to (`>=`) comparison
    # operation with broadcasting.
    def >=(other)
      self.class.greater_equal(self, other)
    end

    # Performs element-wise less than (`<`) comparison operation
    # with broadcasting.
    def <(other)
      self.class.lesser(self, other)
    end

    # Performs element-wise less than or equal to (`<=`) comparison
    # operation with broadcasting.
    def <=(other)
      self.class.lesser_equal(self, other)
    end

    # Performs element-wise numerical negative.
    def -
      NDArray::Internal._mul_scalar(self, scalar: -1)
    end

    # Leaves the values unchanged.
    def +
      self
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
      out = Ops._slice(self, begin: ranges.map(&.first), end: ranges.map(&.last))
      dims.size > 0 ? out.reshape(shape: dims) : out
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
    # * *value* (`Number` or `MXNet::NDArray)`)
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
          value.reshape(shape: dims),
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

    # Copies the values of this array to another array.
    #
    # If *other* is a `NDArray` object, then `other.shape` and
    # `self.shape` must be the same. This method copies the data from
    # *self* to *other*.
    #
    # If *other* is a `Context` object, then a new `NDArray` will be
    # created on the target context, and the method copies the data
    # from *self* to the new array.
    #
    # ### Parameters
    # * *other* (`NDArray` or `Context`)
    #   The destination array or context.
    #
    def copy_to(other : Context | self)
      if other.is_a?(self)
        if self.handle == other.handle
          raise NDArrayException.new("cannot copy an array onto itself")
        end
        Internal._copyto(self, out: other)
      else
        NDArray.empty(shape: shape, dtype: dtype, ctx: other).tap do |res|
          Internal._copyto(self, out: res)
        end
      end
    end

    # Returns a copy of the array after casting to a specified type.
    #
    # ### Parameters
    # * *dtype* (`::Symbol`)
    #   The type of the copy.
    # * *copy* (`Bool`, default = `true`)
    #   By default, `#as_type` always returns a newly allocated array
    #   on the same context. If *copy* is set to `false`, and the
    #   *dtype* requested is the same as this array's dtype, this
    #   array is returned instead of a copy.
    #
    def as_type(dtype : ::Symbol, copy = true)
      return self if !copy && dtype == self.dtype
      NDArray.empty(shape: shape, dtype: dtype, ctx: context).tap do |res|
        copy_to(res)
      end
    end

    # Returns a copy of the array on the target device with the same
    # values as this array.
    #
    # ### Parameters
    # * *context* (`Context`)
    #   The target context.
    # * *copy* (`Bool`, default = `false`)
    #   By default, if the target context is the same as this context,
    #   this array is returned and no copy is made. If *copy* is set
    #   to `true`, and the target context is the same as this context,
    #   a copy is returned instead.
    #
    def as_in_context(context : Context, copy = false)
      return self if !copy && context == self.context
      NDArray.empty(shape: shape, dtype: dtype, ctx: context).tap do |res|
        copy_to(res)
      end
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
      unless shape == [1_u32]
        raise NDArrayException.new("the array is not scalar")
      end
      raw[0]
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
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.zeros(shape : Int | Array(Int), ctx = Context.current, **kwargs)
      Internal._zeros(**kwargs.merge({shape: shape, ctx: ctx}))
    end

    # Returns an MXNet array filled with all ones, with the given
    # shape and type.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the array.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.ones(shape : Int | Array(Int), ctx = Context.current, **kwargs)
      Internal._ones(**kwargs.merge({shape: shape, ctx: ctx}))
    end

    # Draw random samples from a uniform distribution.
    #
    # Samples are uniformly distributed over the half-open interval
    # [low, high) (includes low, but excludes high).
    #
    # ```
    # MXNet::NDArray.random_uniform(0.0, 1.0, shape: [2, 2]) # => [[0.60276335, 0.85794562], [0.54488319, 0.84725171]]
    # ```
    #
    # ### Parameters
    # * *low* (`Float`, default = 0.0)
    #   Lower bound of the distribution.
    # * *high* (`Float`, default = 1.0)
    #   Upper bound of the distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output. If unspecified, the type is
    #   `:float32` unless the type can be inferred from the output
    #   array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_uniform(low : Number = 0.0, high : Number = 1.0, ctx = Context.current, **kwargs)
      Internal._random_uniform(**kwargs.merge({low: low, high: high, ctx: ctx}))
    end

    # Draw random samples from a normal (Gaussian) distribution.
    #
    # Samples are distributed according to a normal distribution
    # parametrized by loc (mean) and scale (standard deviation).
    #
    # ```
    # MXNet::NDArray.random_normal(0.0, 1.0, shape: [2, 2]) # => [[1.89171135, -1.16881478], [-1.23474145, 1.55807114]]
    # ```
    #
    # ### Parameters
    # * *loc* (`Float`, default = 0.0)
    #   Mean of the distribution.
    # * *scale* (`Float`, default = 1.0)
    #   Standard deviation of the distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output. If unspecified, the type is
    #   `:float32` unless the type can be inferred from the output
    #   array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_normal(loc : Number = 0.0, scale : Number = 1.0, ctx = Context.current, **kwargs)
      Internal._random_normal(**kwargs.merge({loc: loc, scale: scale, ctx: ctx}))
    end

    # Draw random samples from a Poisson distribution.
    #
    # Samples are distributed according to a Poisson distribution
    # parametrized by lambda (rate). Samples will always be returned
    # as a floating point data type.
    #
    # ```
    # MXNet::NDArray.random_poisson(4.0, shape: [2, 2]) # => [[5.0, 2.0], [4.0, 6.0]]
    # ```
    #
    # ### Parameters
    # * *lam* (`Float`, default = 1.0)
    #   Lambda parameter (rate) of the Poisson distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output. If unspecified, the type is
    #   `:float32` unless the type can be inferred from the output
    #   array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_poisson(lam : Number = 1.0, ctx = Context.current, **kwargs)
      Internal._random_poisson(**kwargs.merge({lam: lam, ctx: ctx}))
    end

    # Draw random samples from an exponential distribution.
    #
    # Samples are distributed according to an exponential distribution
    # parametrized by lambda (rate).
    #
    # ```
    # MXNet::NDArray.random_exponential(4.0, shape: [2, 2]) # => [[0.0097189 , 0.08999364], [0.04146638, 0.31715935]]
    # ```
    #
    # ### Parameters
    # * *lam* (`Float`, default = 1.0)
    #   Lambda parameter (rate) of the exponential distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output. If unspecified, the type is
    #   `:float32` unless the type can be inferred from the output
    #   array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_exponential(lam : Number = 1.0, ctx = Context.current, **kwargs)
      Internal._random_exponential(**kwargs.merge({lam: lam, ctx: ctx}))
    end

    # Draw random samples from a gamma distribution.
    #
    # Samples are distributed according to a gamma distribution
    # parametrized by alpha (shape) and beta (scale).
    #
    # ```
    # MXNet::NDArray.random_gamma(9.0, 0.5, shape: [2, 2]) # => [[6.2806954, 6.1658335], [4.5625057, 6.479337]]
    # ```
    #
    # ### Parameters
    # * *alpha* (`Float`, default = 1.0)
    #   Alpha parameter (shape) of the gamma distribution.
    # * *beta* (`Float`, default = 1.0)
    #   Beta parameter (scale) of the gamma distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output. If unspecified, the type is
    #   `:float32` unless the type can be inferred from the output
    #   array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    # * *out* (`NDArray`, optional)
    #   The output array.
    #
    def self.random_gamma(alpha : Number = 1.0, beta : Number = 1.0, ctx = Context.current, **kwargs)
      Internal._random_gamma(**kwargs.merge({alpha: alpha, beta: beta, ctx: ctx}))
    end

    # Returns an MXNet array of given shape and type, without
    # initializing entries.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the empty array.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output array.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    #
    def self.empty(shape : Int | Array(Int), dtype = :float32, ctx = nil)
      shape = shape.is_a?(Int) ? [shape] : shape
      dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      ctx ||= Context.current
      MXNet::Internal.libcall(
        MXNDArrayCreateEx,
        shape.map(&.to_u32),
        shape.size,
        *ctx.device,
        0,
        dtype,
        out handle
      )
      new(handle)
    end

    # Creates an MXNet array from any enumerable object.
    #
    # ### Parameters
    # * *source* (`Enumerable(T)`)
    #   Any enumerable object, or nested objects.
    # * *dtype* (`::Symbol`, optional)
    #   The data type of the output array. If unspecified, the type is
    #   inferred from the source type.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    #
    def self.array(source : Enumerable(T), dtype = nil, ctx = nil) forall T
      source = source.to_a
      shape_and_type = infer_shape_and_type(source)
      inferred_shape = shape_and_type.map(&.first)
      inferred_type = shape_and_type.last.last
      elements = source.flatten

      if dtype
        dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      end
      if inferred_type
        inferred_type = INFERRED_TYPES[inferred_type]? || raise MXNet::NDArrayException.new("type is unsupported: #{inferred_type}")
      else
        raise MXNet::NDArrayException.new("type can't be inferred")
      end

      ctx ||= Context.current

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

    # Saves arrays to a file.
    #
    # Examples of filenames:
    # - `/path/to/file`
    # - `s3://my-bucket/path/to/file` (if MXNet is compiled with AWS S3 supports)
    # - `hdfs://path/to/file` (if MXNet is compiled with HDFS supports)
    #
    # ### Parameters
    # * *fname* (`String`)
    #   The filename.
    # * *data* (`NDArray` or `Enumerable({String, NDArray})` or `Enumerable(NDArray)`)
    #   The data to save.
    #
    def self.save(fname, data)
      case data
      when NDArray
        data = [data]
        keys = [] of String
      when Enumerable({String, NDArray})
        keys = data.map(&.first)
        data = data.map(&.last)
      when Enumerable(NDArray)
        data = data.to_a
        keys = [] of String
      else
        raise ArgumentError.new(
          "Data must either be an NDArray, an enumerable of NDArrays, " \
          "or a enumerable of String, NDArray tuples."
        )
      end
      MXNet::Internal.libcall(
        MXNDArraySave,
        fname,
        data.size,
        data.map(&.handle.as(NDArrayHandle)),
        keys.map(&.to_unsafe)
      )
    end

    # Loads arrays from a file.
    #
    # Returns `Array(NDArray)` or `Hash(String, NDArray)`.
    # See `.save` for more detail on format.
    #
    # Examples of filenames:
    # - `/path/to/file`
    # - `s3://my-bucket/path/to/file` (if MXNet is compiled with AWS S3 supports)
    # - `hdfs://path/to/file` (if MXNet is compiled with HDFS supports)
    #
    # ### Parameters
    # * *fname* (`String`)
    #   The filename.
    #
    def self.load(fname)
      MXNet::Internal.libcall(
        MXNDArrayLoad,
        fname,
        out size,
        out arr,
        out name_size,
        out names
      )
      if name_size == 0
        size.times.reduce([] of NDArray) do |array, i|
          value = new(arr[i])
          array << value
        end
      elsif name_size == size
        size.times.reduce({} of String => NDArray) do |hash, i|
          key = String.new(names[i])
          value = new(arr[i])
          hash[key] = value
          hash
        end
      else
        raise Exception.new("invalid file format")
      end
    end

    # TODO: cache op handles
    def self.imperative_invoke(op, *ndargs, out _out : self? = nil, **kwargs)
      ndargs = ndargs.size > 0 ?
        # flatten; reject nil values; obtain handles
        ndargs.to_a.flatten.compact.map { |v| v.handle } :
        [] of NDArrayHandle
      kwargs = kwargs.size > 0 ?
        # stringify; reject entries with empty values and the "name" special key
        kwargs.map { |k, v| [output(k), output(v)] }.reject { |(k, v)| v.empty? || k == "name" }.to_h :
        {} of String => String

      num_outputs = 0
      outputs = Pointer(NDArrayHandle).null
      if _out
        num_outputs = 1
        outputs = Pointer(NDArrayHandle).malloc(1)
        outputs[0] = _out.handle
      end

      # ignore
      kwargs.delete(:name)

      MXNet::Internal.libcall(
        NNGetOpHandle,
        op.to_s,
        out op_handle
      )
      MXNet::Internal.libcall(
        MXImperativeInvoke,
        op_handle,
        ndargs.size,
        ndargs,
        pointerof(num_outputs),
        pointerof(outputs),
        kwargs.size,
        kwargs.keys.map(&.to_unsafe),
        kwargs.values.map(&.to_unsafe)
      )
      _out || new(outputs[0])
    end
  end
end

struct Number
  # Performs element-wise addition.
  def +(other : MXNet::NDArray)
    MXNet::NDArray.add(self, other)
  end

  # Performs element-wise subtraction.
  def -(other : MXNet::NDArray)
    MXNet::NDArray.subtract(self, other)
  end

  # Performs element-wise multiplication.
  def *(other : MXNet::NDArray)
    MXNet::NDArray.multiply(self, other)
  end

  # Performs element-wise division.
  def /(other : MXNet::NDArray)
    MXNet::NDArray.divide(self, other)
  end

  # Returns the result of this number raised to powers from the array,
  # element-wise.
  def **(other : MXNet::NDArray)
    MXNet::NDArray.power(self, other)
  end

  # Performs element-wise equal to (`==`) comparison.
  def ==(other : MXNet::NDArray)
    MXNet::NDArray.equal(self, other)
  end

  # Performs element-wise not equal to (`!=`) comparison.
  def !=(other : MXNet::NDArray)
    MXNet::NDArray.not_equal(self, other)
  end

  # Performs element-wise greater than (`>`) comparison.
  def >(other : MXNet::NDArray)
    MXNet::NDArray.greater(self, other)
  end

  # Performs element-wise greater than or equal to (`>=`) comparison.
  def >=(other : MXNet::NDArray)
    MXNet::NDArray.greater_equal(self, other)
  end

  # Performs element-wise less than (`<`) comparison.
  def <(other : MXNet::NDArray)
    MXNet::NDArray.lesser(self, other)
  end

  # Performs element-wise less than or equal to (`<=`) comparison.
  def <=(other : MXNet::NDArray)
    MXNet::NDArray.lesser_equal(self, other)
  end
end
