module MXNet
  class Symbol
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

    def bind(ctx : Context, args : Array(MXNet::NDArray))
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

    def to_s(io)
      io << "<Symbol #{name}>"
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

    # TODO: cache op handles
    def self.create_symbol(op, *args, **kwargs)
      op = op.to_s
      args = args.to_a
      kwargs = kwargs.to_h

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
        op,
        args.size,
        nil,
        args.to_a.map(&.handle.as(SymbolHandle)))
      sym
    end
  end
end
