require "../mxnet"

module MXNet
  # Cached operator.
  #
  class CachedOp
    include MXNet::Util

    @handle : CachedOpHandle

    def initialize(symbol, flags)
      flags = flags.size > 0 ?
        # stringify
        flags.map { |k, v| [output(k), output(v)] }.to_h :
        {} of String => String

      MXNet::Internal.libcall(
        MXCreateCachedOpEx,
        symbol.handle,
        flags.size,
        flags.keys.map(&.to_unsafe),
        flags.values.map(&.to_unsafe),
        out @handle
      )
    end

    # Invokes the cached operator.
    #
    def call(args, out _out : NDArray? = nil)
      args = args.size > 0 ?
        # obtain handles
        args.to_a.compact.map { |v| v.handle } :
        [] of NDArrayHandle

      num_outputs = 0
      outputs = Pointer(NDArrayHandle).null
      if _out
        num_outputs = 1
        outputs = Pointer(NDArrayHandle).malloc(1)
        outputs[0] = _out.handle
      end

      MXNet::Internal.libcall(
        MXInvokeCachedOpEx,
        @handle,
        args.size,
        args,
        pointerof(num_outputs),
        pointerof(outputs),
        out stypes
      )

      if _out
        [_out]
      else
        num_outputs.times.reduce([] of NDArray) do |arr, i|
          arr << NDArray.new(outputs[i])
        end
      end
    end

    # :nodoc:
    def finalize
      MXNet::Internal.libcall(MXFreeCachedOp, @handle)
    end
  end
end
