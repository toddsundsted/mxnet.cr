module MXNet
  class SymbolException < Exception
  end

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

    # Binds the current symbol to an executor and returns the executor.
    #
    # First, declare the computation and then bind to the data to
    # evaluate.  This function returns an executor which provides an
    # `Executor#forward()` method for evaluation.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b # => "<Symbol broadcast_add>"
    # e = c.bind(args: {"a" => MXNet::NDArray.ones([2, 3]), "b" => MXNet::NDArray.ones([2, 3])}, ctx: MXNet.cpu)
    # e.forward.first # => [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    #                 #    <NDArray 2x3 float32 cpu(0)>
    # ```
    #
    # ### Parameters
    # * *args* (`Array(MXNet::NDArray)` or `Hash(String, MXNet::NDArray)`, default `[]`)
    #   Input arguments.
    #   * If the input type is `Array(MXNet::NDArray)`, the order should be same as the order returned by `#list_arguments`.
    #   * If the input type is `Hash(String, MXNet::NDArray)`, the arguments map to those returned by `#list_arguments`.
    # * *ctx* (`Context`, default is current context)
    #   The device context the executor is to evaluate on.
    #
    def bind(args : Array(MXNet::NDArray) | Hash(String, MXNet::NDArray) = [] of MXNet::NDArray, ctx : Context = MXNet::Context.current)
      arguments = list_arguments
      if args.is_a?(Array(MXNet::NDArray))
        if arguments.size != args.size
          raise SymbolException.new("wrong number of arguments (expected #{arguments.size}, given #{args.size})")
        end
      elsif args.is_a?(Hash(String, MXNet::NDArray))
        args = arguments.map { |a| args[a]?.as(MXNet::NDArray | Nil) }.compact
        if arguments.size != args.size
          raise SymbolException.new("wrong number of arguments (expected #{arguments.size}, matched #{args.size})")
        end
      end

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

    # Evaluates a symbol given arguments.
    #
    # The `#eval` method combines a call to `#bind` (which returns an
    # `Executor`) with a call to `Executor#forward`. For the common
    # use case, where you might repeatedly evaluate with the same
    # arguments, `#eval` is slow. In that case, you should call `#bind`
    # once and then repeatedly call `Executor#forward`. This function
    # allows simpler syntax for less cumbersome introspection.
    #
    # Returns an array of `MXNet::NDArray` corresponding to the values
    # taken by each symbol when evaluated on the given arguments. When
    # called on a single symbol (not a group), the result will be an
    # array with one element.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b # => "<Symbol broadcast_add>"
    # c.eval(a: MXNet::NDArray.ones([2, 3]), b: MXNet::NDArray.ones([2, 3])) # => [<NDArray 2x3 int32 @cpu(0)>]
    # c.eval(MXNet::NDArray.ones([2, 3]), MXNet::NDArray.ones([2, 3])) # => [<NDArray 2x3 int32 @cpu(0)>]
    # ```
    #
    # ### Parameters
    # * *ctx* (`Context`, default is current context)
    #   The device context the executor is to evaluate on.
    # * *ndargs* (`MXNet::NDArray`)
    #   Input arguments. All the arguments must be provided.
    #
    def eval(*ndargs : MXNet::NDArray, ctx : Context = MXNet::Context.current)
      args = ndargs.to_a
      bind(args: args, ctx: ctx).forward
    end

    # ditto
    def eval(ctx : Context = MXNet::Context.current, **ndargs : MXNet::NDArray)
      args = ndargs.map { |k, v| {k.to_s, v} }.to_h
      bind(ctx: ctx, args: args).forward
    end

    # ditto
    def eval(ctx : Context = MXNet::Context.current)
      bind(ctx: ctx).forward
    end

    def to_s(io)
      io << "<Symbol"
      io << " " << name
      io << ">"
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
