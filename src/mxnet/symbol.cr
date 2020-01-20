module MXNet
  class SymbolException < Exception
  end

  # The `Symbol` API provides neural network graphs and
  # auto-differentiation. A symbol represents a multi-output symbolic
  # expression. Symbols are composited by operators, such as simple
  # matrix operations (e.g. “+”), or a neural network layer (e.g.
  # convolution layer). An operator can take several input variables,
  # produce more than one output variable, and have internal state
  # variables. A variable can be either free, which we can bind with
  # values later, or can be an output of another symbol.
  #
  # ```
  # a = MXNet::Symbol.var("a")
  # b = MXNet::Symbol.var("b")
  # c = 2 * a + b
  # e = c.bind({"a" => MXNet::NDArray.array([1, 2]), "b" => MXNet::NDArray.array([2, 3])}, MXNet.cpu)
  # e.forward.first # => [4, 7]
  #                 #    <NDArray 2 int32 cpu(0)>
  # ```
  #
  # A detailed (albeit in Python) tutorial is available at
  # [Symbol - Neural network graphs](https://mxnet.incubator.apache.org/versions/master/tutorials/basic/symbol.html).
  #
  # Note: most operators provided in `Symbol` are similar to those in
  # `NDArray` although there are few differences:
  #
  # * `Symbol` adopts a declarative programming style. In other words,
  #   we need to first compose the computations, and then feed the
  #   computation with data for execution, whereas `NDArray` adopts an
  #   imperative programming style.
  #
  # * Most binary operators in `Symbol` such as `+` and `>` don’t
  #   broadcast. You need to call the broadcast version of the
  #   operator, such as `broadcast_plus`, explicitly.
  #
  class Symbol < Base
    @handle : SymbolHandle

    # :nodoc:
    protected def initialize(@handle)
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

    # Gets the attribute for specified key.
    #
    # This function only works for non-grouped symbols.
    #
    # ```
    # data = MXNet::Symbol.var("data", attr: {mood: "angry"})
    # data.attr("mood") # => "angry"
    # ```
    #
    # ### Parameters
    # * *key* (`String`)
    #   The key corresponding to the desired attribute.
    #
    def attr(key)
      MXNet::Internal.libcall(MXSymbolGetAttr, @handle, key, out value, out success)
      success != 0 ? String.new(value) : nil
    end

    # Gets all attributes.
    #
    # ```
    # data = MXNet::Symbol.var("data", attr: {"mood" => "angry"})
    # data.list_attr # => {"mood" => "angry"}
    # ```
    #
    def list_attr
      MXNet::Internal.libcall(
        MXSymbolListAttrShallow,
        @handle,
        out size,
        out pairs
      )
      Hash(String, String).new.tap do |ret|
        size.times do |i|
          key = String.new(pairs[i * 2])
          value = String.new(pairs[i * 2 + 1])
          ret[key] = value
        end
      end
    end

    # Recursively gets all attributes from the symbol and its
    # children.
    #
    # There is a key in the returned hash for every child with a
    # non-empty set of attributes. For each symbol, the name of the
    # symbol is its key in the hash and the correspond value is that
    # symbol's attribute list.
    #
    # ```
    # a = MXNet::Symbol.var("a", attr: {"a1" => "a2"})
    # b = MXNet::Symbol.var("b", attr: {"b1" => "b2"})
    # c = a + b
    # c.attr_dict # => {"a" => {"a1" => "a2"}, "b" => {"b1" => "b2"}}
    # ```
    #
    def attr_dict
      MXNet::Internal.libcall(
        MXSymbolListAttr,
        @handle,
        out size,
        out pairs
      )
      Hash(String, Hash(String, String)).new.tap do |ret|
        size.times do |i|
          name, key = String.new(pairs[i * 2]).split("$")
          value = String.new(pairs[i * 2 + 1])
          ret[name] ||= Hash(String, String).new
          ret[name][key] = value
        end
      end
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

    # Lists all the auxiliary states of the symbol.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b
    # c.list_auxiliary_states # => []
    # ```
    #
    # Auxiliary states are special states of symbols that do not
    # correspond to an argument, and are not updated by gradient
    # descent. Common examples of auxiliary states include the
    # *moving_mean* and *moving_variance* in `BatchNorm`. Most
    # operators do not have auxiliary states.
    #
    def list_auxiliary_states
      MXNet::Internal.libcall(MXSymbolListAuxiliaryStates, @handle, out size, out str_array)
      str_array.to_slice(size).map { |u| String.new(u) }.to_a
    end

    # Generates "infer_shape..." methods.
    #
    private macro infer_shape_impl(op, name)
      def {{name.id}}(args)
        keys = [] of String
        data = [] of UInt32
        iptr = [0_u32]

        case args
        when Array(Array(Int32) | Nil), Array(Array(Int32))
          args.each do |s|
            if s
              data += s.map(&.to_u32)
            end
            iptr << data.size.to_u32
          end
        when Hash(String, Array(Int32) | Nil), Hash(String, Array(Int32))
          args.each do |k, v|
            keys << k
            if (v)
              data += v.map(&.to_u32)
            end
            iptr << data.size.to_u32
          end
        else
          raise ArgumentError.new(
            "specify arguments either positionally or by name"
          )
        end

        MXNet::Internal.libcall(
          {{op.id}},
          @handle,
          iptr.size - 1,
          keys.map(&.to_unsafe),
          iptr,
          data,
          out arg_shape_size,
          out arg_shape_ndim,
          out arg_shape_data,
          out out_shape_size,
          out out_shape_ndim,
          out out_shape_data,
          out aux_shape_size,
          out aux_shape_ndim,
          out aux_shape_data,
          out complete
        )

        if complete != 0
          arg_shapes = arg_shape_size.times.map do |i|
            l = arg_shape_ndim[i]
            s = arg_shape_data[i].to_slice(l)
            s[0, l].map(&.to_i).to_a
          end
          out_shapes = out_shape_size.times.map do |i|
            l = out_shape_ndim[i]
            s = out_shape_data[i].to_slice(l)
            s[0, l].map(&.to_i).to_a
          end
          aux_shapes = aux_shape_size.times.map do |i|
            l = aux_shape_ndim[i]
            s = aux_shape_data[i].to_slice(l)
            s[0, l].map(&.to_i).to_a
          end
          {arg_shapes.to_a, out_shapes.to_a, aux_shapes.to_a}
        else
          {nil, nil, nil}
        end
      end
    end

    # Infers the shapes of all arguments and all outputs, given the
    # known shapes of some arguments.
    #
    # This function takes the known shapes of arguments either
    # positionally or by name. It returns a tuple of `nil` values if
    # there is not enough information to deduce the missing shapes.
    #
    # Inconsistencies in the known shapes will cause an error to be
    # raised.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b
    # arg_shapes, out_shapes, aux_shapes = c.infer_shape([nil, [3, 3]])
    # arg_shapes # => [[3, 3], [3, 3]]
    # out_shapes # => [[3, 3]]
    # aux_shapes # => []
    # ```
    #
    # ### Parameters
    # * *args* (`Array(Array(Int32) | Nil)` or `Hash(String, Array(Int32) | Nil)`)
    #   Shapes of known arguments. Unknown shapes can be marked as `nil`.
    #
    infer_shape_impl(MXSymbolInferShape, infer_shape)

    # Infers the shapes partially.
    #
    # This functions works the same way as `#infer_shape`, except that
    # this function can return partial results.
    #
    # In the following example, information about "b" is not
    # available. So, `#infer_shape` will return a tuple of `nil`
    # values but this method will return partial values.
    #
    # ```
    # a = MXNet::Symbol.fully_connected(MXNet::Symbol.var("a"), nil, nil, num_hidden: 128)
    # b = MXNet::Symbol.fully_connected(MXNet::Symbol.var("b"), nil, nil, num_hidden: 128)
    # c = a + b
    # arg_shapes, out_shapes, aux_shapes = c.infer_shape_partial([[10, 64]])
    # arg_shapes # => [[10, 64], [128, 64], [128], [], [], []]
    # out_shapes # => [[10, 128]]
    # aux_shapes # => []
    # ```
    #
    infer_shape_impl(MXSymbolInferShapePartial, infer_shape_partial)

    # Generates "infer_dtype..." methods.
    #
    private macro infer_dtype_impl(op, name)
      def {{name.id}}(args)
        {% if compare_versions(MXNet::Internal::MXNET_VERSION, "1.5.0") < 0 %}
          {% if op.stringify == "MXSymbolInferTypePartial" %}
            raise MXNetException.new("not supported on MXNet version #{MXNet::Internal::MXNET_VERSION}: {{name}}")
          {% end %}
        {% end %}

        keys = [] of String
        data = [] of Int32

        case args
        when Array(::Symbol | Nil), Array(::Symbol)
          args.each do |s|
            if s
              data << T2DT[s]
            else
              data << -1
            end
          end
        when Hash(String, ::Symbol | Nil), Hash(String, ::Symbol)
          args.each do |k, v|
            keys << k
            if v
              data << T2DT[v]
            else
              data << -1
            end
          end
        else
          raise ArgumentError.new(
            "specify arguments either positionally or by name"
          )
        end

        MXNet::Internal.libcall(
          {{op.id}},
          @handle,
          data.size,
          keys.map(&.to_unsafe),
          data,
          out arg_type_size,
          out arg_type_data,
          out out_type_size,
          out out_type_data,
          out aux_type_size,
          out aux_type_data,
          out complete
        )

        if complete != 0
          arg_types = arg_type_size.times.map do |i|
            DT2T[arg_type_data[i]]?
          end
          out_types = out_type_size.times.map do |i|
            DT2T[out_type_data[i]]?
          end
          aux_types = aux_type_size.times.map do |i|
            DT2T[aux_type_data[i]]?
          end
          {arg_types.to_a, out_types.to_a, aux_types.to_a}
        else
          {nil, nil, nil}
        end
      end
    end

    # Infers the dtypes of all arguments and all outputs, given the
    # known dtypes of some arguments.
    #
    # This function takes the known dtypes of arguments either
    # positionally or by name. It returns a tuple of `nil` values if
    # there is not enough information to deduce the missing dtypes.
    #
    # Inconsistencies in the known dtypes will cause an error to be
    # raised.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # c = a + b
    # arg_types, out_types, aux_types = c.infer_dtype({"a" => :float32})
    # arg_types # => [:float32, :float32]
    # out_types # => [:float32]
    # aux_types # => []
    # ```
    #
    # ### Parameters
    # * *args* (`Array(::Symbol | Nil)` or `Hash(String, ::Symbol | Nil)`)
    #   Dtypes of known arguments. Unknown dtypes can be marked as `nil`.
    #
    infer_dtype_impl(MXSymbolInferType, infer_dtype)

    # Infers the dtypes partially.
    #
    # This functions works the same way as `#infer_dtype`, except that
    # this function can return partial results.
    #
    # In the following example, information about "b" is not
    # available. So, `#infer_shape` will return a tuple of `nil`
    # values but this method will return partial values.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol::Ops._cast(MXNet::Symbol.var("b"), dtype: :int32)
    # c = a + b
    # arg_types, out_types, aux_types = c.infer_dtype_partial([:int32])
    # arg_types # => [:int32, nil]
    # out_types # => [:int32]
    # aux_types # => []
    #
    # ```
    #
    infer_dtype_impl(MXSymbolInferTypePartial, infer_dtype_partial)

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

    # Returns a shallow copy of this symbol.
    #
    # This method is functionally identical to `#clone`.
    #
    def dup
      MXNet::Internal.libcall(
        MXSymbolCopy,
        self.handle,
        out handle
      )
      self.class.new(handle)
    end

    # Returns a deep copy of this symbol.
    #
    # This method is functionally identical to `#dup`.
    #
    def clone
      MXNet::Internal.libcall(
        MXSymbolCopy,
        self.handle,
        out handle
      )
      self.class.new(handle)
    end

    # Saves symbol to a JSON file.
    #
    # ### Parameters
    # * *fname* (`String`)
    #   The name of the file.
    # * *symbol* (`MXNet::Symbol`)
    #   Symbol to save.
    #
    def self.save(fname, symbol)
      MXNet::Internal.libcall(
        MXSymbolSaveToFile,
        symbol.handle,
        fname
      )
    end

    # Loads symbol from a JSON file.
    #
    # ### Parameters
    # * *fname* (`String`)
    #   The name of the file.

    def self.load(fname)
      MXNet::Internal.libcall(
        MXSymbolCreateFromFile,
        fname,
        out sym_handle
      )
      new(sym_handle)
    end

    # Returns element-wise sum of the input arrays.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs + rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be added.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be added.
    #
    bifunc_helper(
      add,
      lhs, rhs,
      Internal._plus,
      :+,
      Internal._plus_scalar,
      Internal._plus_scalar
    )

    # Returns element-wise difference of the input arrays.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs - rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be subtracted.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be subtracted.
    #
    bifunc_helper(
      subtract,
      lhs, rhs,
      Internal._minus,
      :-,
      Internal._rminus_scalar,
      Internal._minus_scalar
    )

    # Returns element-wise product of the input arrays.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs * rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be multiplied.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be multiplied.
    #
    bifunc_helper(
      multiply,
      lhs, rhs,
      Internal._mul,
      :*,
      Internal._mul_scalar,
      Internal._mul_scalar
    )

    # Returns element-wise division of the input arrays.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs / rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be divided.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be divided.
    #
    bifunc_helper(
      divide,
      lhs, rhs,
      Internal._div,
      :/,
      Internal._rdiv_scalar,
      Internal._div_scalar
    )

    # Returns result of first array elements raised to powers from
    # second array, element-wise.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `base ** exp`.
    #
    # ### Parameters
    # * *base* (`Symbol` or `Number`)
    #   The base value.
    # * *exp* (`Symbol` or `Number`)
    #   The exponent value.
    #
    bifunc_helper(
      power,
      base, exp,
      Internal._power,
      :**,
      Internal._rpower_scalar,
      Internal._power_scalar
    )

    # Returns element-wise maximum of the input arrays.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      maximum,
      lhs, rhs,
      Internal._maximum,
      lhs > rhs ? lhs : rhs,
      Internal._maximum_scalar,
      Internal._maximum_scalar
    )

    # Returns element-wise minimum of the input arrays.
    #
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      minimum,
      lhs, rhs,
      Internal._minimum,
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
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs == rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      equal,
      lhs, rhs,
      Internal._equal,
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
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs != rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      not_equal,
      lhs, rhs,
      Internal._not_equal,
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
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs > rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      greater,
      lhs, rhs,
      Internal._greater,
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
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs >= rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      greater_equal,
      lhs, rhs,
      Internal._greater_equal,
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
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs < rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      lesser,
      lhs, rhs,
      Internal._lesser,
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
    # Both inputs can be a `Symbol` or a scalar number. Broadcasting
    # is not supported.
    #
    # Equivalent to `lhs <= rhs`.
    #
    # ### Parameters
    # * *lhs* (`Symbol` or `Number`)
    #   The first value to be compared.
    # * *rhs* (`Symbol` or `Number`)
    #   The second value to be compared.
    #
    bifunc_helper(
      lesser_equal,
      lhs, rhs,
      Internal._lesser_equal,
      lhs <= rhs ? 1.0 : 0.0,
      Internal._greater_equal_scalar,
      Internal._lesser_equal_scalar
    )

    # Performs element-wise addition (without broadcasting).
    def +(other)
      self.class.add(self, other)
    end

    # Performs element-wise subtraction (without broadcasting).
    def -(other)
      self.class.subtract(self, other)
    end

    # Performs element-wise multiplication (without broadcasting).
    def *(other)
      self.class.multiply(self, other)
    end

    # Performs element-wise division (without broadcasting).
    def /(other)
      self.class.divide(self, other)
    end

    # Returns the result of the first array elements raised to powers
    # from the second array (or scalar), element-wise (without
    # broadcasting).
    def **(other)
      self.class.power(self, other)
    end

    # Performs element-wise equal to (`==`) comparison operation
    # (without broadcasting).
    def ==(other)
      self.class.equal(self, other)
    end

    # Performs element-wise not equal to (`!=`) comparison operation
    # (without broadcasting).
    def !=(other)
      self.class.not_equal(self, other)
    end

    # Performs element-wise greater than (`>`) comparison operation
    # (without broadcasting).
    def >(other)
      self.class.greater(self, other)
    end

    # Performs element-wise greater than or equal to (`>=`) comparison
    # operation (without broadcasting).
    def >=(other)
      self.class.greater_equal(self, other)
    end

    # Performs element-wise less than (`<`) comparison operation
    # (without broadcasting).
    def <(other)
      self.class.lesser(self, other)
    end

    # Performs element-wise less than or equal to (`<=`) comparison
    # operation (without broadcasting).
    def <=(other)
      self.class.lesser_equal(self, other)
    end

    # Performs element-wise numerical negative.
    def -
      Symbol::Internal._mul_scalar(self, scalar: -1)
    end

    # Leaves the values unchanged.
    def +
      self
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

    # Sets attributes of the symbol.
    #
    protected def set_attr(attr)
      attr.each do |key, value|
        MXNet::Internal.libcall(
          MXSymbolSetAttr,
          @handle,
          key.to_s,
          value.to_s
        )
      end
    end

    # Creates a symbolic variable with the specified name.
    #
    # ### Parameters
    # * *name* (`String`)
    #   Variable name.
    # * *attr* (`Enumerable`)
    #   Additional attributes to set on the variable.
    # * *shape* (`Array(Int)`)
    #   The shape of a variable. If specified, it may be used during
    #   the shape inference.
    # * *dtype* (`::Symbol`)
    #   The dtype for input variable. If not specified, this value
    #   will be inferred.
    #
    def self.var(name : String, attr = nil, shape = nil, dtype = nil)
      MXNet::Internal.libcall(MXSymbolCreateVariable, name, out handle)
      new(handle).tap do |ret|
        attr ||= {} of ::Symbol => String
        attr[:__shape__] = shape.to_s if shape
        attr[:__dtype__] = T2DT[dtype].to_s if dtype
        ret.set_attr(attr)
      end
    end

    # Create a symbol representing zeros, with the given
    # shape and type.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the symbol.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the symbol.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.zeros(shape : Int | Array(Int), **kwargs)
      Internal._zeros(**kwargs.merge({shape: shape}))
    end

    # Create a symbol representing ones, with the given
    # shape and type.
    #
    # ### Parameters
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the symbol.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the symbol.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.ones(shape : Int | Array(Int), **kwargs)
      Internal._ones(**kwargs.merge({shape: shape}))
    end

    # Draw random samples from a uniform distribution.
    #
    # Samples are uniformly distributed over the half-open interval
    # [low, high) (includes low, but excludes high).
    #
    # ```
    # MXNet::Symbol.random_uniform(0.0, 1.0, shape: [2, 2]).eval.first # => [[0.60276335, 0.85794562], [0.54488319, 0.84725171]]
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
    #   The data type of the output.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_uniform(low : Number = 0.0, high : Number = 1.0, **kwargs)
      Internal._random_uniform(**kwargs.merge({low: low, high: high}))
    end

    # Draw random samples from a normal (Gaussian) distribution.
    #
    # Samples are distributed according to a normal distribution
    # parametrized by loc (mean) and scale (standard deviation).
    #
    # ```
    # MXNet::Symbol.random_normal(0.0, 1.0, shape: [2, 2]).eval.first # => [[1.89171135, -1.16881478], [-1.23474145, 1.55807114]]
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
    #   The data type of the output.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_normal(loc : Number = 0.0, scale : Number = 1.0, **kwargs)
      Internal._random_normal(**kwargs.merge({loc: loc, scale: scale}))
    end

    # Draw random samples from a Poisson distribution.
    #
    # Samples are distributed according to a Poisson distribution
    # parametrized by lambda (rate). Samples will always be returned
    # as a floating point data type.
    #
    # ```
    # MXNet::Symbol.random_poisson(4.0, shape: [2, 2]).eval.first # => [[5.0, 2.0], [4.0, 6.0]]
    # ```
    #
    # ### Parameters
    # * *lam* (`Float`, default = 1.0)
    #   Lambda parameter (rate) of the Poisson distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_poisson(lam : Number = 1.0, **kwargs)
      Internal._random_poisson(**kwargs.merge({lam: lam}))
    end

    # Draw random samples from an exponential distribution.
    #
    # Samples are distributed according to an exponential distribution
    # parametrized by lambda (rate).
    #
    # ```
    # MXNet::Symbol.random_exponential(4.0, shape: [2, 2]).eval.first # => [[0.0097189 , 0.08999364], [0.04146638, 0.31715935]]
    # ```
    #
    # ### Parameters
    # * *lam* (`Float`, default = 1.0)
    #   Lambda parameter (rate) of the exponential distribution.
    # * *shape* (`Int` or `Array(Int)`)
    #   The shape of the output.
    # * *dtype* (`::Symbol`, default = `:float32`)
    #   The data type of the output.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_exponential(lam : Number = 1.0, **kwargs)
      Internal._random_exponential(**kwargs.merge({lam: lam}))
    end

    # Draw random samples from a gamma distribution.
    #
    # Samples are distributed according to a gamma distribution
    # parametrized by alpha (shape) and beta (scale).
    #
    # ```
    # MXNet::Symbol.random_gamma(9.0, 0.5, shape: [2, 2]).eval.first # => [[6.2806954, 6.1658335], [4.5625057, 6.479337]]
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
    #   The data type of the output.
    # * *name* (`String`, optional)
    #   Name of the resulting symbol.
    #
    def self.random_gamma(alpha : Number = 1.0, beta : Number = 1.0, **kwargs)
      Internal._random_gamma(**kwargs.merge({alpha: alpha, beta: beta}))
    end

    # Creates a symbol that contains a collection of other symbols,
    # grouped together.
    #
    # ```
    # a = MXNet::Symbol.var("a")
    # b = MXNet::Symbol.var("b")
    # MXNet::Symbol.group([a, b]) # => grouped symbol
    # ```
    #
    def self.group(symbols : Array(MXNet::Symbol)) : MXNet::Symbol
      MXNet::Internal.libcall(
        MXSymbolCreateGroup,
        symbols.size,
        symbols.map(&.handle),
        out sym_handle
      )
      new(sym_handle)
    end

    # TODO: cache op handles
    def self.create_symbol(op, *args, name : String? = nil, **kwargs)
      args = args.size > 0 ?
        # flatten; reject nil values; obtain handles
        args.to_a.flatten.compact.map { |v| v.handle } :
        [] of SymbolHandle
      kwargs = kwargs.size > 0 ?
        # stringify; reject entries with empty values and the "out" special key
        kwargs.map { |k, v| [output(k), output(v)] }.reject { |(k, v)| v.empty? || k == "out" }.to_h :
        {} of String => String

      name = MXNet::Name::Manager.current.get(name, op.downcase)

      # ignore
      kwargs.delete(:out)

      MXNet::Internal.libcall(
        NNGetOpHandle,
        op.to_s,
        out op_handle
      )
      MXNet::Internal.libcall(
        MXSymbolCreateAtomicSymbol,
        op_handle,
        kwargs.size,
        kwargs.keys.map(&.to_unsafe),
        kwargs.values.map(&.to_unsafe),
        out sym_handle
      )
      sym = new(sym_handle)
      MXNet::Internal.libcall(
        NNSymbolCompose,
        sym_handle,
        name,
        args.size,
        nil,
        args
      )
      sym
    end
  end
end

struct Number
  # Performs element-wise addition.
  def +(other : MXNet::Symbol)
    MXNet::Symbol.add(self, other)
  end

  # Performs element-wise subtraction.
  def -(other : MXNet::Symbol)
    MXNet::Symbol.subtract(self, other)
  end

  # Performs element-wise multiplication.
  def *(other : MXNet::Symbol)
    MXNet::Symbol.multiply(self, other)
  end

  # Performs element-wise division.
  def /(other : MXNet::Symbol)
    MXNet::Symbol.divide(self, other)
  end

  # Returns the result of this number raised to powers from the array,
  # element-wise.
  def **(other : MXNet::Symbol)
    MXNet::Symbol.power(self, other)
  end

  # Performs element-wise equal to (`==`) comparison.
  def ==(other : MXNet::Symbol)
    MXNet::Symbol.equal(self, other)
  end

  # Performs element-wise not equal to (`!=`) comparison.
  def !=(other : MXNet::Symbol)
    MXNet::Symbol.not_equal(self, other)
  end

  # Performs element-wise greater than (`>`) comparison.
  def >(other : MXNet::Symbol)
    MXNet::Symbol.greater(self, other)
  end

  # Performs element-wise greater than or equal to (`>=`) comparison.
  def >=(other : MXNet::Symbol)
    MXNet::Symbol.greater_equal(self, other)
  end

  # Performs element-wise less than (`<`) comparison.
  def <(other : MXNet::Symbol)
    MXNet::Symbol.lesser(self, other)
  end

  # Performs element-wise less than or equal to (`<=`) comparison.
  def <=(other : MXNet::Symbol)
    MXNet::Symbol.lesser_equal(self, other)
  end
end
