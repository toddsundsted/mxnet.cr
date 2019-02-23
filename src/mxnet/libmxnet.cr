module MXNet
  class Internal
    class LibraryException < Exception
    end

    macro libcall(expr, *args)
      unless Internal::LibMXNet.{{ expr }}({{ *args }}) == 0
        raise Internal::LibraryException.new(String.new(Internal::LibMXNet.MXGetLastError))
      end
    end

    @[Link("mxnet")]

    lib LibMXNet
      type NDArrayHandle = Void*
      type OpHandle = Void*

      fun MXGetLastError() : UInt8*
      fun MXGetVersion(i : Int32*) : Int32
      fun MXNDArrayCreateEx(
        shape : UInt32*,
        ndim : UInt32,
        dev_type : Int32, dev_id : Int32,
        delay_alloc : Int32,
        dtype : Int32,
        handle : NDArrayHandle*
      ) : Int32
      fun MXNDArraySyncCopyFromCPU(handle : NDArrayHandle, data : Void*, size : LibC::SizeT) : Int32
      fun MXNDArraySyncCopyToCPU(handle : NDArrayHandle, data : Void*, size : LibC::SizeT) : Int32
      fun MXNDArrayGetShape(handle : NDArrayHandle, dim : UInt32*, pdata : UInt32**) : Int32
      fun MXNDArrayGetDType(handle : NDArrayHandle, dtype : UInt32*) : Int32
      fun MXNDArrayFree(handle : NDArrayHandle) : Int32
    end
  end
end
