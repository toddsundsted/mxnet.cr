module MXNet
  class NDArrayException < Exception
  end

  class NDArray
    @handle : MXNet::Internal::LibMXNet::NDArrayHandle

    DT2T = {
      0 => :float32,
      1 => :float64,
      3 => :uint8,
      4 => :int32,
      5 => :int8,
      6 => :int64
    }
    T2DT = {
      :float32 => 0,
      :float64 => 1,
      :uint8 => 3,
      :int32 => 4,
      :int8 => 5,
      :int64 => 6
    }

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

    def dtype
      MXNet::Internal.libcall(MXNDArrayGetDType, @handle, out dtype)
      DT2T[dtype]
    end

    def copy_to(other : self)
      NDArray.imperative_invoke("_copyto", self, out: other)
    end

    def as_type(dtype : Symbol)
      return self if dtype == self.dtype
      NDArray.empty(shape, dtype: dtype).tap do |res|
        copy_to(res)
      end
    end

    def to_s(io)
      data = raw
      shape.reverse.each do |dim|
        data = data.in_groups_of(dim).map { |group| "[#{group.join(", ")}]" }
      end
      data.each { |line| io << line << "\n" }
      io << "<NDArray #{shape.join("x")} #{dtype} cpu(0)>"
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

    def self.empty(shape, dtype : Symbol = :float32)
      dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      MXNet::Internal.libcall(MXNDArrayCreateEx, shape, shape.size, 1, 0, 0, dtype, out handle)
      new(handle)
    end

    def self.array(array : Array(T), dtype : Symbol | Nil = nil) forall T
      shape_and_type = infer_shape_and_type(array)
      inferred_shape = shape_and_type.map(&.first)
      inferred_type = shape_and_type.last.last
      elements = array.flatten

      if dtype
        dtype = T2DT[dtype]? || raise MXNet::NDArrayException.new("type is unsupported: #{dtype}")
      end
      if inferred_type
        inferred_type = INFERRED_TYPES[inferred_type]? || raise MXNet::NDArrayException.new("type is unsupported: #{inferred_type}")
      else
        raise MXNet::NDArrayException.new("type can't be inferred")
      end

      MXNet::Internal.libcall(MXNDArrayCreateEx, inferred_shape, inferred_shape.size, 1, 0, 0, inferred_type, out handle)
      MXNet::Internal.libcall(MXNDArraySyncCopyFromCPU, handle, elements, elements.size)

      if dtype && dtype != inferred_type
        empty(inferred_shape, dtype: DT2T[dtype]).tap do |res|
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
      outputs = Pointer(MXNet::Internal::LibMXNet::NDArrayHandle).null

      if kwargs.has_key?(:out)
        out = kwargs.delete(:out)
        if out.is_a?(NDArray)
          num_outputs = 1
          outputs = Pointer(MXNet::Internal::LibMXNet::NDArrayHandle).malloc(1)
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
        ndargs.map(&.handle),
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
