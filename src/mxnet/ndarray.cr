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

    macro arithmetic(op, array_mod, scalar_mod)
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
      num_outputs.times.map { |i| NDArray.new(outputs[i]) }
    end
  end
end
