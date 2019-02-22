module MXNet
  class LibraryException < Exception
  end

  macro libcall(expr, *args)
    unless LibMXNet.{{ expr }}({{ *args }}) == 0
      raise LibraryException.new(String.new(LibMXNet.MXGetLastError))
    end
  end

  @[Link("mxnet")]

  lib LibMXNet
    type NDArrayHandle = Void*
    type OpHandle = Void*

    fun MXGetLastError() : UInt8*
    fun MXGetVersion(i : Int32*) : Int32
  end
end
