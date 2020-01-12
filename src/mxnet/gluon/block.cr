require "../gluon"

module MXNet
  module Gluon
    # Scope for collecting child `Blocks`.
    #
    class BlockScope
      protected def initialize(@block : Block? = nil)
        @counters = Hash(String, Int32).new { -1 }
      end

      getter :block
      getter :counters

      @@current : self? = nil

      # Creates prefix and params for new `Block`.
      #
      def self.create(prefix, params, hint) : {String, ParameterDict}
        current = @@current ||= BlockScope.new
        if (block = current.block)
          if prefix.nil?
            prefix = "#{hint}#{current.counters[hint] += 1}_"
          end
          if params.nil?
            params = ParameterDict.new(prefix: "#{block.prefix}#{prefix}")
          else
            params = ParameterDict.new(prefix: "#{block.prefix}#{prefix}", shared: params)
          end
          {"#{block.prefix}#{prefix}", params}
        else
          if prefix.nil?
            prefix = "#{hint}#{current.counters[hint] += 1}_"
          end
          if params.nil?
            params = ParameterDict.new(prefix: prefix)
          else
            params = ParameterDict.new(prefix: params.prefix, shared: params)
          end
          {prefix, params}
        end
      end

      def call(block)
        previous, @@current = @@current, block.scope
        yield
      ensure
        @@current = previous
      end
    end

    # Base class for all neural network layers and models. Your models
    # should subclass this class.
    #
    class Block
      # Creates accessors for declared attributes.
      #
      macro attribute(*names)
        {% for name in names %}
          {% if name.is_a?(TypeDeclaration) %}
            def {{name.var.id}}
              get_attr("{{name.var.id}}").as({{name.type}})
            end
            def {{name.var.id}}? : {{name.type}} | Nil
              (value = get_attr("{{name.var.id}}")) ? value.as({{name.type}}) : nil
            end
            def {{name.var.id}}=({{name.var.id}} : {{name.type}} | Nil)
              set_attr("{{name.var.id}}", {{name.var.id}})
            end
          {% elsif name.is_a?(Call) %}
            {% raise "must include a type declaration: #{name}" %}
          {% else %}
            {% raise "must be a type declaration: #{name}" %}
          {% end %}
        {% end %}
      end

      @reg_children = Hash(String, MXNet::Gluon::Block).new
      @reg_parameters = Hash(String, MXNet::Gluon::Parameter).new
      @reg_other = Hash(String, Nil).new

      def initialize(prefix = nil, params = nil)
        @prefix, @params = BlockScope.create(prefix, params, hint)
        @scope = BlockScope.new(self)
      end

      # Scope of this block.
      getter scope

      # Prefix of this block.
      getter prefix

      # Returns this block's parameter dictionary.
      #
      # Does not include its children's parameters.
      #
      getter params

      # Enters a name scope managing block names.
      #
      #     self.with_name_scope do
      #       self.dense = MXNet::Gluon::NN.Dense(20)
      #     end
      #
      def with_name_scope
        unless scope = @scope
          raise Exception.new(
            "Ensure that parent classes are initialized by calling " \
            "`super(...)` in #{self.class}#initialize()."
          )
        end
        scope.call(self) do
          yield
        end
      end

      # Returns this block's registered children.
      #
      def children
        @reg_children.values
      end

      # Returns a `ParameterDict` containing this `Block`'s and all of
      # its children's `Parameter`s. Can also return the `Parameter`s
      # that match some given regular expressions.
      #
      # For example, collect the specified `Parameter`s for
      # "conv1_weight", "conv1_bias", "fc_weight" and "fc_bias":
      #
      #     model.collect_params(/conv1_weight|conv1_bias|fc_weight|fc_bias/)
      #
      # or, alternatively, collect all parameters whose names end with
      # "weight" or "bias":
      #
      #     model.collect_params(/.*weight|.*bias/)
      #
      # ### Parameters
      # * *selector* (`Regex`)
      #   Regular expressions to match parameters.
      #
      def collect_params(selector = nil)
        ret = ParameterDict.new(prefix: @params.prefix)
        if selector
          ret.update(@params.select { |(k, _)| k =~ selector })
        else
          ret.update(@params)
        end
        @reg_children.each_value do |child|
          ret.update(child.collect_params(selector))
        end
        ret
      end

      # Initializes parameters of this block and its children.
      # Equivalent to `self.collect_params.init(...)`.
      #
      # ### Parameters
      # * *init* (`Initializer`, default = `nil`)
      #   The initializer to use.
      # * *ctx* (`Context` or `Array(Context)`, default = `nil`)
      #   Desired contexts.
      # * *force_reinit* (`Bool`, default = `false`)
      #   Whether to force re-initialization if parameter is already
      #   initialized.
      #
      def init(init = nil, ctx = nil, force_reinit = false)
        collect_params.init(init: init, ctx: ctx, force_reinit: force_reinit)
        self
      end

      # Registers block as a child of self. Blocks assigned as
      # attributes will be registered automatically.
      #
      def register_child(block, name = nil)
        name = @reg_children.size.to_s unless name
        @reg_children[name] = block
        block
      end

      # Registers parameter on self. Parameters assigned as attributes
      # will be registered automatically.
      #
      def register_parameter(param, name = nil)
        name = param.name unless name
        @reg_parameters[name] = param
        param
      end

      # Activates or deactivates `HybridBlock` children
      # recursively. Has no effect on non-hybrid blocks.
      #
      # ### Parameters
      # * *active* (`Bool`, default = `true`)
      #   Whether to turn hybridization on or off.
      #
      def hybridize(active = true)
        @reg_children.each_value do |child|
          child.hybridize(active)
        end
      end

      # Calls `#forward`.
      #
      # Only accepts positional arguments.
      #
      # ### Parameters
      # * *inputs* (`Array(NDArray)`)
      #   Input tensors.
      #
      def call(inputs : Array(T)) : Array(T) forall T
        forward(inputs)
      end

      # Override to implement forward computation using `NDArray`.
      #
      # Only accepts positional arguments.
      #
      # ### Parameters
      # * *inputs* (`Array(NDArray)`)
      #   Input tensors.
      #
      def forward(inputs : Array(T)) : Array(T) forall T
        raise NotImplementedError.new(
          "#forward must be implemented in a subclass"
        )
      end

      def set_attr(name : String, value : Block | Parameter | Nil)
        [@reg_children, @reg_parameters, @reg_other].each do |reg|
          if reg.has_key?(name) && reg[name].class != value.class
            raise Exception.new(
              "changing attribute class for #{name} " \
              "from #{reg[name].class} to #{value.class} " \
              "is not allowed"
            )
          end
        end
        case value
        when Block
          register_child(value, name)
        when Parameter
          register_parameter(value, name)
        else
          @reg_other[name] = value
        end
      end

      def get_attr(name : String) : Block | Parameter | Nil
        [@reg_children, @reg_parameters, @reg_other].each do |reg|
          return reg[name] if reg.has_key?(name)
        end
        raise Exception.new("undefined attribute #{name}")
      end

      # Saves parameters to a file.
      #
      # Note that this method only saves parameters, not model
      # structure. If you want to save model structures, use
      # `HybridBlock#export`.
      #
      # For reference see: "Saving and Loading Gluon Models"
      # (https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html).
      #
      # ### Parameters
      # * *fname* (`String`)
      #   Path to file.
      #
      def save_parameters(fname)
        params = collect_params_for_storage.transform_values(&._reduce)
        MXNet::NDArray.save(fname, params)
      end

      # Loads parameters from a file.
      #
      # For reference see: "Saving and Loading Gluon Models"
      # (https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html).
      #
      # ### Parameters
      # * *fname* (`String`)
      #   Path to file.
      # * *ctx* (`Context` or `Array(Context)`, default = cpu)
      #   Context(s) to initialize loaded parameters on.
      # * *allow_missing* (`Bool`, default = `false`)
      #   Whether to silently skip parameters not present
      #   in the file.
      # * *ignore_extra* (`Bool`, default = `false`)
      #   Whether to silently ignore parameters not present
      #   in this block.
      #
      def load_parameters(fname, ctx = MXNet.cpu, allow_missing = false, ignore_extra = false)
        loaded = MXNet::NDArray.load(fname)
        unless loaded.is_a?(Hash(String, NDArray))
          raise Exception.new(
            "Can't load from format #{loaded.class}."
          )
        end
        params = collect_params_for_storage
        unless allow_missing
          params.each do |key, value|
            unless loaded.has_key?(key)
              raise Exception.new(
                "Parameter '#{key}' is missing from file. " \
                "Set `allow_missing` to `true` to ignore."
              )
            end
          end
        end
        loaded.each do |key, value|
          unless params.has_key?(key)
            unless ignore_extra
              raise Exception.new(
                "Value '#{key}' from file is not present in block. " \
                "Set `ignore_extra` to `true` to ignore."
              )
            end
            next
          end
          params[key]._load_init(ctx, value)
        end
      end

      protected def collect_params_for_storage(prefix = "")
        prefix += "." unless prefix.empty?
        Hash(String, Parameter).new.tap do |hash|
          @reg_parameters.each do |key, param|
            hash[prefix + key] = param
          end
          @reg_children.each do |key, child|
            hash.merge!(child.collect_params_for_storage(prefix + key))
          end
        end
      end

      private def hint
        self.class.name.split("::").last.downcase
      end
    end

    # Encapsulates caching symbolized operations.
    #
    module CachedGraph
      @flags = {} of String => String

      @graph : Tuple(Array(MXNet::Symbol), Array(MXNet::Symbol))?
      @cached_op : Tuple(Array(MXNet::Gluon::Parameter), Array(MXNet::Symbol), MXNet::CachedOp)?

      def initialize(**kwargs)
        super(**kwargs)
        clear_cache
      end

      def clear_cache
        @graph = nil
        @cached_op = nil
      end

      def infer_shape(args)
        inputs, outputs = get_graph(args)
        output = MXNet::Symbol.group(outputs)
        arg_attrs, _, aux_attrs =
          output.infer_shape(inputs.zip(args).reduce({} of String => Array(Int32)) { |a, (i, j)| a[i.name.not_nil!] = j.shape ; a })
        shapes =
          output.list_arguments.zip(arg_attrs.not_nil!).to_h.merge(output.list_auxiliary_states.zip(aux_attrs.not_nil!).to_h)
        collect_params.values.each do |value|
          value.shape = shapes[value.name]
        end
      end

      def infer_dtype(args)
        inputs, outputs = get_graph(args)
        output = MXNet::Symbol.group(outputs)
        arg_attrs, _, aux_attrs =
          output.infer_dtype(inputs.zip(args).reduce({} of String => ::Symbol) { |a, (i, j)| a[i.name.not_nil!] = j.dtype ; a })
        dtypes =
          output.list_arguments.zip(arg_attrs.not_nil!).to_h.merge(output.list_auxiliary_states.zip(aux_attrs.not_nil!).to_h)
        collect_params.values.each do |value|
          value.dtype = dtypes[value.name]
        end
      end

      private def get_graph(args)
        @graph ||=
          begin
            inputs =
              if args.size > 1
                (0...args.size).map do |i|
                  MXNet::Symbol.var("data#{i}")
                end
              else
                [MXNet::Symbol.var("data")]
              end
            params = @reg_parameters.reduce({} of String => MXNet::Symbol) do |acc, (i, j)|
              acc[i] = j.var
              acc
            end
            {inputs, hybrid_forward(inputs, params)}
          end
      end

      private def get_cached_op(args)
        @cached_op ||=
          begin
            params = collect_params.values
            _, outputs = get_graph(args)
            symbol = outputs.size > 1 ? MXNet::Symbol.group(outputs) : outputs.first
            {params, outputs, MXNet::CachedOp.new(symbol, @flags)}
          end
      end

      protected def call_cached(inputs)
        params, _, cached_op = get_cached_op(inputs)
        loop do
          pdata = params.map(&.data)
          return cached_op.call(inputs + pdata)
        rescue MXNet::Gluon::DeferredInitializationError
          infer_shape(inputs)
          infer_dtype(inputs)
          params.each do |param|
            param._finish_deferred_init
          end
        end
      end
    end

    # `HybridBlock` supports forwarding with both `Symbol` and
    # `NDArray`.
    #
    class HybridBlock < Block
      def register_child(block, name = nil)
        unless block.is_a?(MXNet::Gluon::HybridBlock)
          raise Exception.new(
            "Children of HybridBlock must also be HybridBlocks, " \
            "but #{block} has type #{block.class}. If you are using " \
            "Sequential, please try HybridSequential instead."
          )
        end
        super
      end

      # Defines the forward computation.
      #
      # ### Parameters
      # * *inputs* (`Array(Symbol)` or `Array(NDArray)`)
      #   Input tensors.
      #
      def forward(inputs : Array(T)) : Array(T) forall T
        hybrid_forward(inputs)
      end

      # Override to construct symbolic graph for this `HybridBlock`.
      #
      # ### Parameters
      # * *inputs* (`Array(Symbol)` or `Array(NDArray)`)
      #   Input tensors.
      #
      def hybrid_forward(inputs : Array(T), params : Hash(String, T) = {} of String => T) : Array(T) forall T
        raise NotImplementedError.new(
          "#hybrid_forward must be implemented in a subclass"
        )
      end
    end
  end
end
