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
    include Indexable(self)

    @handle : NDArrayHandle

    # :nodoc:
    protected def initialize(@handle)
    end

    # :nodoc:
    def handle
      @handle
    end

    def shape
      MXNet::Internal.libcall(MXNDArrayGetShape, @handle, out dim, out pdata)
      pdata.to_slice(dim).map(&.to_i32).to_a
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

    # Returns element-wise modulo of the input arrays.
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs % rhs`. Equivalent to `.broadcast_mod` for
    # `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to modulo.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to modulo.
    #
    bifunc_helper(
      modulo,
      lhs, rhs,
      Ops._broadcast_mod,
      :%,
      Internal._rmod_scalar,
      Internal._mod_scalar
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

    # Returns the result of element-wise logical and (`&`) comparison
    # operation.
    #
    # For each element in input arrays, return 1 (true) if *lhs*
    # element and *rhs* element is true (not zero), otherwise return 0
    # (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs & rhs`. Equivalent to
    # `.broadcast_logical_and` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      logical_and,
      lhs, rhs,
      Ops._broadcast_logical_and,
      :&,
      Internal._logical_and_scalar,
      Internal._logical_and_scalar
    )

    # Returns the result of element-wise logical or (`|`) comparison
    # operation.
    #
    # For each element in input arrays, return 1 (true) if *lhs*
    # element or *rhs* element is true (not zero), otherwise return 0
    # (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs | rhs`. Equivalent to
    # `.broadcast_logical_or` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      logical_or,
      lhs, rhs,
      Ops._broadcast_logical_or,
      :|,
      Internal._logical_or_scalar,
      Internal._logical_or_scalar
    )

    # Returns the result of element-wise logical xor (`^`) comparison
    # operation.
    #
    # For each element in input arrays, return 1 (true) if either
    # *lhs* element or *rhs* element is true (not zero) but not both,
    # otherwise return 0 (false).
    #
    # If the corresponding dimensions of two arrays have the same size
    # or one of them has size 1, then the arrays are broadcastable to
    # a common shape.
    #
    # Equivalent to `lhs ^ rhs`. Equivalent to
    # `.broadcast_logical_xor` for `NDArray` arguments.
    #
    # ### Parameters
    # * *lhs* (`NDArray` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`NDArray` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      logical_xor,
      lhs, rhs,
      Ops._broadcast_logical_xor,
      :^,
      Internal._logical_xor_scalar,
      Internal._logical_xor_scalar
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

    # Performs element-wise modulo with broadcasting.
    def %(other)
      self.class.modulo(self, other)
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

    # Performs element-wise logical and (`&`) comparison operation
    # with broadcasting.
    def &(other)
      self.class.logical_and(self, other)
    end

    # Performs element-wise logical or (`|`) comparison operation
    # with broadcasting.
    def |(other)
      self.class.logical_or(self, other)
    end

    # Performs element-wise logical xor (`^`) comparison operation
    # with broadcasting.
    def ^(other)
      self.class.logical_xor(self, other)
    end

    # Performs element-wise numerical negative.
    def -
      NDArray::Internal._mul_scalar(self, scalar: -1)
    end

    # Leaves the values unchanged.
    def +
      self
    end

    # Methods required to implement `Indexable`.

    def size
      shape[0]
    end

    def unsafe_fetch(idx)
      shape = self.shape
      shape.shift
      out = Ops._slice(self, begin: [idx], end: [idx + 1])
      shape.size > 0 ? out.reshape(shape: shape) : out
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
      shape = self.shape
      ranges = keys.map_with_index do |k, i|
        if k.is_a?(Int)
          b = k
          e = k + 1
          {b, e}
        else
          if bk = k.begin
            b = bk
          else
            b = 0
          end
          if tk = k.end
            e = tk < 0 ? shape[i] + tk : tk
            e = k.excludes_end? ? e : e + 1
          else
            e = shape[i]
          end
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
    def [](keys : Array(Int | Range(Int?, Int?) ))
      ranges, dims = ranges_and_dims(keys, compact: true)
      out = Ops._slice(self, begin: ranges.map(&.first), end: ranges.map(&.last))
      dims.size > 0 ? out.reshape(shape: dims) : out.reshape(shape: [1])
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
    def []=(keys : Array(Int | Range(Int?, Int?)), value : Number | self)
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

    # Returns a copy of the array after casting to the specified type.
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
    # Only supports 1-dimensional arrays (`shape.size == 1`).
    #
    # ```
    # MXNet::NDArray.zeros([4], dtype: :float64).to_a # => [0.0, 0.0, 0.0, 0.0]
    # ```
    #
    # The return type of this method is the union of all possible
    # array types (e.g. `Array(Float32) | Array(Float64) | ...`). To
    # return an array and check and restrict the return type in a
    # single operation, see `#to_a(as)`.
    #
    def to_a
      unless shape.size == 1
        raise NDArrayException.new("the array must have only 1 dimension")
      end
      raw
    end

    # Returns an `Array` with values copied from this array.
    #
    # Only supports arrays up to 4 dimensions (`shape.size <= 4`).
    #
    # ```
    # MXNet::NDArray.zeros([4], dtype: :float32).to_a(Float32) # => [0.0, 0.0, 0.0, 0.0]
    # ```
    #
    # To return a 1-dimensional array without checking and restricting
    # the return type, see `#to_a`.
    #
    # ### Parameters
    # * *as* (`Class`)
    #   The class of the contained item. For example, to check and
    #   restrict the return type to `Array(Float32)` specify
    #   `Float32`.
    #
    def to_a(as : Array(Array(Array(T))).class) : Array(Array(Array(Array(T)))) forall T
      {% puts @def %}
      unless shape.size == 4
        raise NDArrayException.new("the array must have 4 dimensions")
      end
      raw
        .in_groups_of(shape[-1], T.zero)
        .in_groups_of(shape[-2], [] of T)
        .in_groups_of(shape[-3], [] of Array(T))
        .as(Array(Array(Array(Array(T)))))
    end

    # :ditto:
    def to_a(as : Array(Array(T)).class) : Array(Array(Array(T))) forall T
      unless shape.size == 3
        raise NDArrayException.new("the array must have 3 dimensions")
      end
      raw
        .in_groups_of(shape[-1], T.zero)
        .in_groups_of(shape[-2], [] of T)
        .as(Array(Array(Array(T))))
    end

    # :ditto:
    def to_a(as : Array(T).class) : Array(Array(T)) forall T
      unless shape.size == 2
        raise NDArrayException.new("the array must have 2 dimensions")
      end
      raw
        .in_groups_of(shape[-1], T.zero)
        .as(Array(Array(T)))
    end

    # :ditto:
    def to_a(as : T.class) : Array(T) forall T
      unless shape.size == 1
        raise NDArrayException.new("the array must have 1 dimension")
      end
      raw
        .as(Array(T))
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

    # :nodoc:
    private INFERRED_TYPES = {
      Float32 => 0,
      Float64 => 1,
      UInt8 => 3,
      Int32 => 4,
      Int8 => 5,
      Int64 => 6
    }

    # Returns a new array of given shape, filled with the given
    # *value*.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the new array.
    # * *value* (`T`)
    #   A fill value of any supported numeric type.
    # * *dtype* (`::Symbol`, optional)
    #   The data type of the output array. If unspecified, the type is
    #   inferred from the value.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    #
    def self.full(shape : Int | Array(Int), value : T, dtype = nil, ctx = nil) forall T
      ctx ||= Context.current

      unless dtype
        dtype = INFERRED_TYPES[T]? || raise MXNet::NDArrayException.new("type is unsupported: #{T}")
        dtype = DT2T[dtype]
      end

      MXNet::NDArray.empty(shape, dtype, ctx).tap do |out|
        out[..] = value
      end
    end

    # Creates an MXNet array from any enumerable object.
    #
    # ### Parameters
    # * *source* (`Enumerable(T)`)
    #   Any enumerable object, or nested enumerable object, whose
    #   elements can be converted to numbers.
    # * *dtype* (`::Symbol`, optional)
    #   The type of the output array. If unspecified, the type is
    #   inferred from the source type.
    # * *ctx* (`Context`, optional)
    #   Device context (default is the current context).
    #
    def self.array(source : Enumerable(T), dtype = nil, ctx = nil) forall T
      source = source.to_a
      inferred_shape = _infer_shape(source)
      source = source.flatten
      inferred_type = _infer_type(source)

      ctx ||= Context.current

      if dtype
        dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      end
      unless inferred_type.empty?
        if inferred_type.size == 1 && (temp = INFERRED_TYPES[inferred_type.first]?)
          size = source.size
          source = source.to_unsafe.as(Pointer(Void))
          inferred_type = temp
        elsif inferred_type.any?(&.<=(Float64))
          size = source.size
          source = source.map(&.to_f64).to_unsafe.as(Pointer(Void))
          inferred_type = INFERRED_TYPES[Float64]
        elsif inferred_type.any?(&.<(Float))
          size = source.size
          source = source.map(&.to_f32).to_unsafe.as(Pointer(Void))
          inferred_type = INFERRED_TYPES[Float32]
        elsif inferred_type.any?(&.<=(Int64))
          size = source.size
          source = source.map(&.to_i64).to_unsafe.as(Pointer(Void))
          inferred_type = INFERRED_TYPES[Int64]
        elsif inferred_type.any?(&.<(Int))
          size = source.size
          source = source.map(&.to_i32).to_unsafe.as(Pointer(Void))
          inferred_type = INFERRED_TYPES[Int32]
        else
          raise MXNet::NDArrayException.new("type is unsupported: #{inferred_type.join(", ")}")
        end
      else
        raise MXNet::NDArrayException.new("type can't be inferred")
      end

      MXNet::Internal.libcall(MXNDArrayCreateEx, inferred_shape, inferred_shape.size, *ctx.device, 0, inferred_type, out handle)
      MXNet::Internal.libcall(MXNDArraySyncCopyFromCPU, handle, source, size)

      if dtype && dtype != inferred_type
        empty(inferred_shape, ctx: ctx, dtype: DT2T[dtype]).tap do |res|
          new(handle).copy_to(res)
        end
      else
        new(handle)
      end
    end

    private def self._infer_type(array : Array(T)) forall T
      {{T.union_types}}
    end

    private def self._infer_shape(array : T) forall T
      nested = array.map { |item| item.is_a?(Array) && item.size.to_u32 }
      if nested.none?
        [array.size.to_u32]
      elsif nested.all?
        unless nested.map { |n| n.is_a?(Bool) ? 0 : n }.sort.uniq.size > 1
          sample = array.sample
          if sample.is_a?(Array)
            [array.size.to_u32] + _infer_shape(sample)
          else
            raise "invalid state"
          end
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

  # Performs element-wise modulo.
  def %(other : MXNet::NDArray)
    MXNet::NDArray.modulo(self, other)
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

  # Performs element-wise logical and (`&`) comparison.
  def &(other : MXNet::NDArray)
    MXNet::NDArray.logical_and(self, other)
  end

  # Performs element-wise logical or (`|`) comparison.
  def |(other : MXNet::NDArray)
    MXNet::NDArray.logical_or(self, other)
  end

  # Performs element-wise logical xor (`^`) comparison.
  def ^(other : MXNet::NDArray)
    MXNet::NDArray.logical_xor(self, other)
  end
end
