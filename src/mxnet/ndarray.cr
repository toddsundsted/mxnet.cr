module MXNet
  class NDArrayException < Exception
  end

  class NDArray
    @handle : MXNet::Internal::LibMXNet::NDArrayHandle

    T2DT = {
      Array(Float32) => 0,
      Array(Float64) => 1,
      Array(UInt8) => 3,
      Array(Int32) => 4,
      Array(Int8) => 5,
      Array(Int64) => 6
    }
    DT2T = {
      0 => :float32,
      1 => :float64,
      3 => :uint8,
      4 => :int32,
      5 => :int8,
      6 => :int64
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

    def to_s(io)
      data = raw
      shape.reverse.each do |dim|
        data = data.in_groups_of(dim).map { |group| "[#{group.join(", ")}]" }
      end
      data.each { |line| io << line << "\n" }
      io << "<NDArray #{shape.join("x")} #{dtype} cpu(0)>"
    end

    private def raw
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

    def self.array(array : Array(T)) forall T
      shape_and_type = infer_shape_and_type(array)
      inferred_shape = shape_and_type.map(&.first)
      inferred_type = shape_and_type.last.last
      elements = array.flatten
      dtype = T2DT[inferred_type]? || raise MXNet::NDArrayException.new("type is unsupported: #{inferred_type}")
      MXNet::Internal.libcall(MXNDArrayCreateEx, inferred_shape, inferred_shape.size, 1, 0, 0, dtype, out handle)
      MXNet::Internal.libcall(MXNDArraySyncCopyFromCPU, handle, elements, elements.size)
      new(handle)
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
