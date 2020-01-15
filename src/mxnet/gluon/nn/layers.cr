require "../nn"

module MXNet
  module Gluon
    module NN
      # Stacks blocks sequentially.
      #
      # ```
      # net = MXNet::Gluon::NN::Sequential.new
      # net.with_name_scope do
      #   net.add(MXNet::Gluon::NN::Dense.new(10, activation: :relu))
      #   net.add(MXNet::Gluon::NN::Dense.new(20))
      # end
      # ```
      #
      class Sequential < MXNet::Gluon::Block
        # Creates a new instance.
        #
        def initialize(**kwargs)
          super(**kwargs)
        end

        # Adds blocks on top of the stack.
        #
        def add(*blocks)
          blocks.each { |block| register_child(block) }
        end

        # Returns the number of blocks in the sequential stack.
        #
        def size
          children.size
        end

        # Returns the block at the specified index.
        #
        # Raises `IndexError` if the index is out of bounds.
        #
        def [](index)
          children[index]
        end

        # Returns the block at the specified index.
        #
        # Returns `nil` if the index is out of bounds.
        #
        def []?(index)
          children[index]?
        end

        # Runs a forward pass on all child blocks.
        #
        # ### Parameters
        # * *inputs* (`Array(Symbol)` or `Array(NDArray)`)
        #   Input tensors.
        #
        def forward(inputs : Array(T)) : Array(T) forall T
          children.reduce(inputs) { |inputs, child| child.call(inputs) }
        end
      end

      # Stacks HybridBlocks sequentially.
      #
      # ```
      # net = MXNet::Gluon::NN::HybridSequential.new
      # net.with_name_scope do
      #   net.add(MXNet::Gluon::NN::Dense.new(10, activation: :relu))
      #   net.add(MXNet::Gluon::NN::Dense.new(20))
      # end
      # net.hybridize
      # ```
      #
      class HybridSequential < MXNet::Gluon::HybridBlock
        # Creates a new instance.
        #
        def initialize(**kwargs)
          super(**kwargs)
        end

        # Adds blocks on top of the stack.
        #
        def add(*blocks)
          blocks.each { |block| register_child(block) }
        end

        # Returns the number of blocks in the sequential stack.
        #
        def size
          children.size
        end

        # Returns the block at the specified index.
        #
        # Raises `IndexError` if the index is out of bounds.
        #
        def [](index)
          children[index]
        end

        # Returns the block at the specified index.
        #
        # Returns `nil` if the index is out of bounds.
        #
        def []?(index)
          children[index]?
        end

        # Runs a forward pass on all child blocks.
        #
        def hybrid_forward(inputs : Array(T), params : Hash(String, T) = {} of String => T) : Array(T) forall T
          children.reduce(inputs) { |inputs, child| child.call(inputs) }
        end
      end

      # A densely-connected neural network layer.
      #
      # Implements the operation:
      #
      #     output = activation(dot(input, weight) + bias)
      #
      # where "activation" is the element-wise activation function passed
      # as the `activation` argument, "weight" is a weights matrix created
      # by the layer, and "bias" is a bias vector created by the layer
      # (if argument `use_bias` is `true`).
      #
      # Note: the input must be a tensor with rank two. Use
      # `flatten` to convert it to rank two if necessary.
      #
      class Dense < MXNet::Gluon::HybridBlock
        attribute \
          weight : MXNet::Gluon::Parameter,
          bias : MXNet::Gluon::Parameter,
          act : MXNet::Gluon::NN::Activation

        # Creates a new instance.
        #
        # ### Parameters
        # * *units* (`Int32`)
        #   Dimensionality of the output space.
        # * *in_units* (`Int32`, optional)
        #   Size of the input data. If nothing is specified,
        #   initialization is deferred to the first time `#forward` is
        #   called and `in_units` will be inferred from the shape of
        #   input data.
        # * *use_bias* (`Bool`, default = `true`)
        #   Whether the layer uses a bias vector.
        # * *activation* (`String`, optional)
        #   Activation function to use. If nothing is specified, no
        #   activation is applied (it acts like "linear" activation:
        #   `a(x) = x`).
        #
        def initialize(@units : Int32, @in_units : Int32 = 0, use_bias = true, activation = nil, **kwargs)
          super(**kwargs)
          with_name_scope do
            self.weight = params.get(
              "weight",
              shape: [units, in_units],
              init: nil,
              allow_deferred_init: true
            )
            if use_bias
              self.bias = params.get(
                "bias",
                shape: [units],
                init: :zeros,
                allow_deferred_init: true
              )
            else
              self.bias = nil
            end
            if activation
              self.act = MXNet::Gluon::NN::Activation.new(
                activation,
                prefix: "#{activation}_"
              )
            else
              self.act = nil
            end
          end
        end

        def hybrid_forward(inputs : Array(T), params : Hash(String, T)) : Array(T) forall T
          weight = params.delete("weight")
          bias = params.delete("bias")
          output = T.fully_connected(
            inputs.first,
            weight,
            bias,
            no_bias: bias.nil?,
            num_hidden: @units
          )
          outputs = [output]
          outputs = self.act.forward(outputs) if self.act?
          outputs
        end
      end

      module Internal
        # Base class for convolution layers.
        #
        # This layer creates a convolution kernel that is convolved with
        # the input to produce a tensor of outputs.
        #
        class Conv < MXNet::Gluon::HybridBlock
          attribute \
            weight : MXNet::Gluon::Parameter,
            bias : MXNet::Gluon::Parameter,
            act : MXNet::Gluon::NN::Activation

          # Creates a new instance.
          #
          # `N` is the number of dimensions of the convolution.
          #
          # ### Parameters
          # * *channels* (`Int32`)
          #   The dimensionality of the output space (the number of
          #   output channels in the convolution).
          # * *kernel_size* (`Array(Int32)` of N integers)
          #   Specifies the dimensions of the convolution window.
          # * *strides* (`Int32` or `Array(Int32)` of N integers)
          #   Specifies the strides of the convolution.
          # * *padding* (`Int32` or `Array(Int32)` of N integers)
          #   If `padding` is non-zero, then the input is implicitly
          #   zero-padded on both sides for `padding` number of
          #   points.
          # * *dilation* (`Int32` or `Array(Int32)` of N integers)
          #   Specifies the dilation rate to use for dilated
          #   convolution.
          # * *layout* (`String`)
          #   Dimension ordering of data and weight. Can be "NCW",
          #   "NWC", "NCHW", "NHWC", "NCDHW", "NDHWC", etc. "N", "C",
          #   "H", "W", "D" stands for batch, channel, height, width
          #   and depth dimensions respectively. Convolution is
          #   performed over "D", "H", and "W" dimensions.
          # * *in_channels* (`Int32`, default = `0`)
          #   The number of input channels to this layer. If not
          #   specified, initialization will be deferred to the first
          #   time `#forward` is called and `in_channels` will be
          #   inferred from the shape of the input data.
          # * *use_bias* (`Bool`, default = `true`)
          #   Whether the layer uses a bias vector.
          # * *activation* (`String`, optional)
          #   Activation function to use. If nothing is specified, no
          #   activation is applied (it acts like "linear" activation:
          #   `a(x) = x`).
          #
          def initialize(*, @channels : Int32, @kernel_size : Array(Int32),
                         @strides : Array(Int32) | Int32, @padding : Array(Int32) | Int32,
                         @dilation : Array(Int32) | Int32, @layout : String,
                         @in_channels = 0, use_bias = true, activation = nil,
                         **kwargs)
            super(**kwargs)
            with_name_scope do
              size = @kernel_size.size
              if (strides = @strides).is_a?(Int)
                @strides = [strides] * size
              end
              if (padding = @padding).is_a?(Int)
                @padding = [padding] * size
              end
              if (dilation = @dilation).is_a?(Int)
                @dilation = [dilation] * size
              end
              kwargs = {
                kernel: @kernel_size,
                stride: @strides,
                pad: @padding,
                dilate: @dilation,
                no_bias: !use_bias,
                num_filter: @channels,
                layout: @layout
              }
              shape = [0] * (@kernel_size.size + 2)
              shape[@layout.index("C").not_nil!] = @in_channels
              shape[@layout.index("N").not_nil!] = 1
              shapes = infer_weight_shapes(shape, **kwargs)
              self.weight = self.params.get(
                "weight",
                shape: shapes[1],
                init: nil,
                allow_deferred_init: true
              )
              if use_bias
                self.bias = self.params.get(
                  "bias",
                  shape: shapes[2],
                  init: :zeros,
                  allow_deferred_init: true
                )
              else
                self.bias = nil
              end
              if activation
                self.act = MXNet::Gluon::NN::Activation.new(
                  activation,
                  prefix: "#{activation}_"
                )
              else
                self.act = nil
              end
            end
          end

          def hybrid_forward(inputs : Array(T), params : Hash(String, T)) : Array(T) forall T
            weight = params.delete("weight")
            bias = params.delete("bias")
            kwargs = {
              kernel: @kernel_size,
              stride: @strides,
              pad: @padding,
              dilate: @dilation,
              no_bias: bias.nil?,
              num_filter: @channels,
              layout: @layout
            }
            output = T.convolution(
              inputs.first,
              weight,
              bias,
              **kwargs
            )
            outputs = [output]
            outputs = self.act.forward(outputs) if self.act?
            outputs
          end

          private def infer_weight_shapes(shape, **kwargs)
            sym = MXNet::Symbol.var("data", shape: shape)
            sym = MXNet::Symbol.convolution(sym, nil, nil, **kwargs)
            sym.infer_shape_partial([] of Array(Int32)).first.not_nil!
          end

          private def hint
            "conv"
          end
        end

        # Base class for pooling layers.
        #
        class Pooling < MXNet::Gluon::HybridBlock
          # Creates a new instance.
          #
          # `N` is the number of dimensions of the pooling layer.
          #
          # ### Parameters
          # * *pool_size* (`Array(Int32)` of N integers)
          #   Specifies the dimensions of pooling operation.
          # * *strides* (`Int32` or `Array(Int32)` of N integers)
          #   Specifies the strides of the pooling operation.
          # * *padding* (`Int32` or `Array(Int32)` of N integers)
          #   If `padding` is non-zero, then the input is implicitly
          #   zero-padded on both sides for `padding` number of
          #   points.
          #
          def initialize(*, @pool_size : Array(Int32),
                         @strides : Array(Int32) | Int32 | Nil, @padding : Array(Int32) | Int32,
                         **kwargs)
            super(**kwargs)
            if @strides.nil?
              @strides = @pool_size
            end
            size = @pool_size.size
            if (strides = @strides).is_a?(Int)
              @strides = [strides] * size
            end
            if (padding = @padding).is_a?(Int)
              @padding = [padding] * size
            end
          end

          def hybrid_forward(inputs : Array(T), params : Hash(String, T)? = nil) : Array(T) forall T
            kwargs = {
              kernel: @pool_size,
              stride: @strides,
              pad: @padding
            }
            output = T.pooling(inputs.first, **kwargs)
            [output]
          end

          private def hint
            "pool"
          end
        end
      end

      # 1D convolution layer (e.g. temporal convolution).
      #
      # This layer creates a convolution kernel that is convolved with
      # the input over a single spatial (or temporal) dimension to
      # produce a tensor of outputs. If `use_bias` is `true`, a bias
      # vector is created and added to the outputs. If `activation` is
      # not `nil`, the activation is applied to the outputs.  If
      # `in_channels` is not specified, parameter initialization will
      # be deferred to the first time `#forward` is called and
      # `in_channels` will be inferred from the shape of input data.
      #
      class Conv1D < MXNet::Gluon::NN::Internal::Conv
        # Creates a new instance.
        #
        # ### Parameters
        # * *channels* (`Int32`)
        #   The dimensionality of the output space (the number of
        #   output channels in the convolution).
        # * *kernel_size* (`Array(Int32)` of 1 integer)
        #   Specifies the dimensions of the convolution window.
        # * *strides* (`Int32` or `Array(Int32)` of 1 integer, default = `1`)
        #   Specifies the strides of the convolution.
        # * *padding* (`Int32` or `Array(Int32)` of 1 integer, default = `0`)
        #   If `padding` is non-zero, then the input is implicitly
        #   zero-padded on both sides for `padding` number of points.
        # * *dilation* (`Int32` or `Array(Int32)` of 1 integer, default = `1`)
        #   Specifies the dilation rate to use for dilated
        #   convolution.
        # * *layout* (`String`, default = `"NCW"`)
        #    Dimension ordering of data and weight. Only supports
        #    "NCW" layout for now. "N", "C", "W" stands for batch,
        #    channel, and width (time) dimensions respectively.
        #    Convolution is applied on the "W" dimension.
        # * *in_channels* (`Int32`, default = `0`)
        #   The number of input channels to this layer. If not
        #   specified, initialization will be deferred to the first
        #   time `#forward` is called and `in_channels` will be
        #   inferred from the shape of the input data.
        # * *use_bias* (`Bool`, default = `true`)
        #   Whether the layer uses a bias vector.
        # * *activation* (`String`, optional)
        #   Activation function to use. If nothing is specified, no
        #   activation is applied (it acts like "linear" activation:
        #   `a(x) = x`).
        #
        def initialize(*, channels, kernel_size, strides = 1, padding = 0, dilation = 1,
                       layout = "NCW", in_channels = 0, use_bias = true, activation = nil,
                       **kwargs)
          if kernel_size.is_a?(Int)
            kernel_size = [kernel_size]
          end
          unless kernel_size.size == 1
            raise ArgumentError.new("kernel_size must be an integer or an array of 1 integer")
          end
          super(
            **kwargs.merge({
              channels: channels,
              kernel_size: kernel_size,
              strides: strides,
              padding: padding,
              dilation: dilation,
              layout: layout,
              in_channels: in_channels,
              use_bias: use_bias,
              activation: activation
            })
          )
        end
      end

      # 2D convolution layer (e.g. spatial convolution over images).
      #
      # This layer creates a convolution kernel that is convolved with
      # the input to produce a tensor of outputs. If `use_bias` is
      # `true`, a bias vector is created and added to the outputs. If
      # `activation` is not `nil`, the activation is applied to the
      # outputs. If `in_channels` is not specified, parameter
      # initialization will be deferred to the first time `#forward`
      # is called and `in_channels` will be inferred from the shape of
      # input data.
      #
      class Conv2D < MXNet::Gluon::NN::Internal::Conv
        # Creates a new instance.
        #
        # ### Parameters
        # * *channels* (`Int32`)
        #   The dimensionality of the output space (the number of
        #   output channels in the convolution).
        # * *kernel_size* (`Array(Int32)` of 2 integers)
        #   Specifies the dimensions of the convolution window.
        # * *strides* (`Int32` or `Array(Int32)` of 2 integers, default = `1`)
        #   Specifies the strides of the convolution.
        # * *padding* (`Int32` or `Array(Int32)` of 2 integers, default = `0`)
        #   If `padding` is non-zero, then the input is implicitly
        #   zero-padded on both sides for `padding` number of points.
        # * *dilation* (`Int32` or `Array(Int32)` of 2 integers, default = `1`)
        #   Specifies the dilation rate to use for dilated
        #   convolution.
        # * *layout* (`String`, default = `"NCHW"`)
        #   Dimension ordering of data and weight. Only supports
        #   "NCHW" and "NHWC" layout for now. "N", "C", "H", "W"
        #   stands for batch, channel, height, and width dimensions
        #   respectively. Convolution is applied on the "H" and "W"
        #   dimensions.
        # * *in_channels* (`Int32`, default = `0`)
        #   The number of input channels to this layer. If not
        #   specified, initialization will be deferred to the first
        #   time `#forward` is called and `in_channels` will be
        #   inferred from the shape of the input data.
        # * *use_bias* (`Bool`, default = `true`)
        #   Whether the layer uses a bias vector.
        # * *activation* (`String`, optional)
        #   Activation function to use. If nothing is specified, no
        #   activation is applied (it acts like "linear" activation:
        #   `a(x) = x`).
        #
        def initialize(*, channels, kernel_size, strides = 1, padding = 0, dilation = 1,
                       layout = "NCHW", in_channels = 0, use_bias = true, activation = nil,
                       **kwargs)
          if kernel_size.is_a?(Int)
            kernel_size = [kernel_size] * 2
          end
          unless kernel_size.size == 2
            raise ArgumentError.new("kernel_size must be an integer or an array of 2 integers")
          end
          super(
            **kwargs.merge({
              channels: channels,
              kernel_size: kernel_size,
              strides: strides,
              padding: padding,
              dilation: dilation,
              layout: layout,
              in_channels: in_channels,
              use_bias: use_bias,
              activation: activation
            })
          )
        end
      end

      # 3D convolution layer (e.g. spatial convolution over volumes).
      #
      # This layer creates a convolution kernel that is convolved with
      # the input to produce a tensor of outputs. If `use_bias` is
      # `true`, a bias vector is created and added to the outputs. If
      # `activation` is not `nil`, the activation is applied to the
      # outputs. If `in_channels` is not specified, `Parameter`
      # initialization will be deferred to the first time `#forward`
      # is called and `in_channels` will be inferred from the shape of
      # input data.
      #
      class Conv3D < MXNet::Gluon::NN::Internal::Conv
        # Creates a new instance.
        #
        # ### Parameters
        # * *channels* (`Int32`)
        #   The dimensionality of the output space (the number of
        #   output channels in the convolution).
        # * *kernel_size* (`Array(Int32)` of 3 integers)
        #   Specifies the dimensions of the convolution window.
        # * *strides* (`Int32` or `Array(Int32)` of 3 integers, default = `1`)
        #   Specifies the strides of the convolution.
        # * *padding* (`Int32` or `Array(Int32)` of 3 integers, default = `0`)
        #   If `padding` is non-zero, then the input is implicitly
        #   zero-padded on both sides for `padding` number of points.
        # * *dilation* (`Int32` or `Array(Int32)` of 3 integers, default = `1`)
        #   Specifies the dilation rate to use for dilated
        #   convolution.
        # * *layout* (`String`, default = `"NCDHW"`)
        #   Dimension ordering of data and weight. Only supports
        #   "NCDHW" and "NDHWC" layout for now. "N", "C", "H", '"W",
        #   "D" stands for batch, channel, height, width and depth
        #   dimensions respectively. Convolution is applied on the
        #   "D", "H" and "W" dimensions.
        # * *in_channels* (`Int32`, default = `0`)
        #   The number of input channels to this layer. If not
        #   specified, initialization will be deferred to the first
        #   time `#forward` is called and `in_channels` will be
        #   inferred from the shape of the input data.
        # * *use_bias* (`Bool`, default = `true`)
        #   Whether the layer uses a bias vector.
        # * *activation* (`String`, optional)
        #   Activation function to use. If nothing is specified, no
        #   activation is applied (it acts like "linear" activation:
        #   `a(x) = x`).
        #
        def initialize(*, channels, kernel_size, strides = 1, padding = 0, dilation = 1,
                       layout = "NCDHW", in_channels = 0, use_bias = true, activation = nil,
                       **kwargs)
          if kernel_size.is_a?(Int)
            kernel_size = [kernel_size] * 3
          end
          unless kernel_size.size == 3
            raise ArgumentError.new("kernel_size must be an integer or an array of 3 integers")
          end
          super(
            **kwargs.merge({
              channels: channels,
              kernel_size: kernel_size,
              strides: strides,
              padding: padding,
              dilation: dilation,
              layout: layout,
              in_channels: in_channels,
              use_bias: use_bias,
              activation: activation
            })
          )
        end
      end

      # Max pooling operation for one dimensional data.
      #
      class MaxPool1D < MXNet::Gluon::NN::Internal::Pooling
        # Creates a new instance.
        #
        # ### Parameters
        # * *pool_size* (`Array(Int32)` of 1 integer, default = `2`)
        #   Specifies the size of pooling window.
        # * *strides* (`Int32` or `Array(Int32)` of 1 integer, default = `nil`)
        #   Specifies the strides of the pooling operation.
        # * *padding* (`Int32` or `Array(Int32)` of 1 integer, default = `0`)
        #   If `padding` is non-zero, then the input is implicitly
        #   zero-padded on both sides for `padding` number of points.
        #
        def initialize(*, pool_size = 2, strides = nil, padding = 0,
                       **kwargs)
          if pool_size.is_a?(Int)
            pool_size = [pool_size]
          end
          unless pool_size.size == 1
            raise ArgumentError.new("pool_size must be an integer or an array of 1 integer")
          end
          super(
            **kwargs.merge({
              pool_size: pool_size,
              strides: strides,
              padding: padding
            })
          )
        end
      end

      # Max pooling operation for 2D data (e.g. images).
      #
      class MaxPool2D < MXNet::Gluon::NN::Internal::Pooling
        # Creates a new instance.
        #
        # ### Parameters
        # * *pool_size* (`Array(Int32)` of 2 integers, default = `2`)
        #   Specifies the size of pooling window.
        # * *strides* (`Int32` or `Array(Int32)` of 2 integers, default = `nil`)
        #   Specifies the strides of the pooling operation.
        # * *padding* (`Int32` or `Array(Int32)` of 2 integers, default = `0`)
        #   If `padding` is non-zero, then the input is implicitly
        #   zero-padded on both sides for `padding` number of points.
        #
        def initialize(*, pool_size = 2, strides = nil, padding = 0,
                       **kwargs)
          if pool_size.is_a?(Int)
            pool_size = [pool_size] * 2
          end
          unless pool_size.size == 2
            raise ArgumentError.new("pool_size must be an integer or an array of 2 integers")
          end
          super(
            **kwargs.merge({
              pool_size: pool_size,
              strides: strides,
              padding: padding
            })
          )
        end
      end

      # Max pooling operation for 3D data (spatial or spatio-temporal).
      #
      class MaxPool3D < MXNet::Gluon::NN::Internal::Pooling
        # ### Parameters
        # * *pool_size* (`Array(Int32)` of 3 integers, default = `2`)
        #   Specifies the size of pooling window.
        # * *strides* (`Int32` or `Array(Int32)` of 3 integers, default = `nil`)
        #   Specifies the strides of the pooling operation.
        # * *padding* (`Int32` or `Array(Int32)` of 3 integers, default = `0`)
        #   If `padding` is non-zero, then the input is implicitly
        #   zero-padded on both sides for `padding` number of points.
        def initialize(*, pool_size = 2, strides = nil, padding = 0,
                       **kwargs)
          if pool_size.is_a?(Int)
            pool_size = [pool_size] * 3
          end
          unless pool_size.size == 3
            raise ArgumentError.new("pool_size must be an integer or an array of 3 integers")
          end
          super(
            **kwargs.merge({
              pool_size: pool_size,
              strides: strides,
              padding: padding
            })
          )
        end
      end

      # Flattens the input to two dimensions.
      #
      # The input is a tensor with an arbitrary shape:
      # `[N, x1, x2, ..., xn]`. The output is a tensor with shape:
      # `[N, x1 * x2 * ... * xn]`.
      #
      class Flatten < MXNet::Gluon::HybridBlock
        # Creates a new instance.
        #
        def initialize(**kwargs)
          super(**kwargs)
        end

        def hybrid_forward(inputs : Array(T), params : Hash(String, T)? = nil) : Array(T) forall T
          output = T.flatten(
            inputs.first
          )
          [output]
        end
      end
    end
  end
end
