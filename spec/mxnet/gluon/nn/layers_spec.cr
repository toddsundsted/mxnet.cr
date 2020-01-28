require "../../../spec_helper"

create_double(
  MXNet::Gluon::Block,
  def forward(inputs, params)
    [T.zeros([3])]
  end
)

describe MXNet::Gluon::NN::Sequential do
  describe "#add" do
    layer = MXNet::Gluon::NN::Sequential.new
    block = MXNet::Gluon::Block.new

    it "registers block as a child" do
      layer.add(block)
      layer.children.should eq([block])
    end
  end

  describe "#size" do
    layer = MXNet::Gluon::NN::Sequential.new
    block = MXNet::Gluon::Block.new

    it "registers block as a child" do
      layer.add(block)
      layer.size.should eq(1)
    end
  end

  describe "#[]" do
    layer = MXNet::Gluon::NN::Sequential.new
    block = MXNet::Gluon::Block.new

    it "returns block at index" do
      layer.add(block)
      layer[0].should eq(block)
    end
  end

  describe "#[]?" do
    layer = MXNet::Gluon::NN::Sequential.new
    block = MXNet::Gluon::Block.new

    it "returns block at index" do
      layer.add(block)
      layer[0]?.should eq(block)
    end
  end

  describe "#forward" do
    layer = MXNet::Gluon::NN::Sequential.new
    block = instance_double(MXNet::Gluon::Block)

    it "runs a forward pass on its children" do
      layer.add(block)
      layer.forward([MXNet::NDArray.ones([3])])
      verify(block, forward: once)
    end
  end
end

create_double(
  MXNet::Gluon::HybridBlock,
  def hybrid_forward(inputs, params)
    [T.zeros([3])]
  end
)

describe MXNet::Gluon::NN::HybridSequential do
  describe "#add" do
    layer = MXNet::Gluon::NN::HybridSequential.new
    block = MXNet::Gluon::HybridBlock.new

    it "registers block as a child" do
      layer.add(block)
      layer.children.should eq([block])
    end
  end

  describe "#size" do
    layer = MXNet::Gluon::NN::HybridSequential.new
    block = MXNet::Gluon::HybridBlock.new

    it "registers block as a child" do
      layer.add(block)
      layer.size.should eq(1)
    end
  end

  describe "#[]" do
    layer = MXNet::Gluon::NN::HybridSequential.new
    block = MXNet::Gluon::HybridBlock.new

    it "returns block at index" do
      layer.add(block)
      layer[0].should eq(block)
    end
  end

  describe "#[]?" do
    layer = MXNet::Gluon::NN::HybridSequential.new
    block = MXNet::Gluon::HybridBlock.new

    it "returns block at index" do
      layer.add(block)
      layer[0]?.should eq(block)
    end
  end

  describe "#forward" do
    layer = MXNet::Gluon::NN::HybridSequential.new
    block = instance_double(MXNet::Gluon::HybridBlock)

    it "runs a forward pass on its children" do
      layer.add(block)
      layer.forward([MXNet::Symbol.var("input")])
      verify(block, hybrid_forward: once)
    end
  end
end

describe MXNet::Gluon::NN::Dense do
  describe ".new" do
    context "with defaults" do
      layer = MXNet::Gluon::NN::Dense.new(1)

      it "sets the weight and bias" do
        layer.weight.shape.should eq([1, 0])
        layer.bias.shape.should eq([1])
      end
      it "does not add activation" do
        layer.act?.should be_nil
      end
    end

    context "with `in_units: 2`" do
      layer = MXNet::Gluon::NN::Dense.new(1, in_units: 2)

      it "sets the weight and bias" do
        layer.weight.shape.should eq([1, 2])
        layer.bias.shape.should eq([1])
      end
    end

    context "with `use_bias: false`" do
      layer = MXNet::Gluon::NN::Dense.new(1, use_bias: false)

      it "disables bias" do
        layer.bias?.should be_nil
      end
    end

    context "with `activation: :relu`" do
      layer = MXNet::Gluon::NN::Dense.new(1, activation: :relu)

      it "adds activation" do
        layer.act.should be_a(MXNet::Gluon::NN::Activation)
      end
    end
  end

  describe "#collect_params" do
    layer =
      MXNet::Gluon::NN::Dense.new(1, in_units: 2, prefix: "dense_")

    it "returns params for weight and bias" do
      params = MXNet::Gluon::ParameterDict.new(prefix: "dense_").tap do |params|
        params.get("weight", shape: [1, 2])
        params.get("bias", shape: [1])
      end
      layer.collect_params.should eq(params)
    end
  end

  describe "#forward" do
    context "with input units specified" do
      layer = MXNet::Gluon::NN::Dense.new(1, in_units: 2).tap do |layer|
          layer.collect_params.init
        end

      it "runs a forward pass" do
        data = MXNet::NDArray.array([[2_f32, 1_f32]])
        layer.forward([data]).should be_a(Array(MXNet::NDArray))
      end
    end
    context "with input units inferred" do
      layer = MXNet::Gluon::NN::Dense.new(1).tap do |layer|
          layer.collect_params.init
        end

      it "runs a forward pass" do
        data = MXNet::NDArray.array([[2_f32, 1_f32]])
        layer.forward([data]).should be_a(Array(MXNet::NDArray))
      end
    end
  end

  describe "#hybrid_forward" do
    layer = MXNet::Gluon::NN::Dense.new(1)
    data = MXNet::NDArray.array([[2_f32, 1_f32]])
    weight = MXNet::NDArray.array([[0.5_f32, 0.5_f32]])
    bias = MXNet::NDArray.array([-1_f32])

    context "without bias" do
      kwargs = {"weight" => weight}
      output = MXNet::NDArray.array([[1.5_f32]])

      it "runs a forward pass" do
        layer.hybrid_forward([data], kwargs).should eq([output])
      end
    end
    context "with bias" do
      kwargs = {"weight" => weight, "bias" => bias}
      output = MXNet::NDArray.array([[0.5_f32]])

      it "runs a forward pass" do
        layer.hybrid_forward([data], kwargs).should eq([output])
      end
    end
  end
end

describe MXNet::Gluon::NN::Internal::Conv do
  newargs = {
    channels: 1, kernel_size: [2, 2], strides: 1, padding: 0, dilation: 1, layout: "NCHW"
  }

  describe ".new" do
    context "with defaults" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs)

      it "sets the weight and bias" do
        layer.weight.shape.should eq([1, 0, 2, 2])
        layer.bias.shape.should eq([1])
      end
      it "does not add activation" do
        layer.act?.should be_nil
      end
    end

    context "with `in_channels: 3`" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs.merge(in_channels: 3))

      it "sets the weight and bias" do
        layer.weight.shape.should eq([1, 3, 2, 2])
        layer.bias.shape.should eq([1])
      end
    end
    context "with `use_bias: false`" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs.merge(use_bias: false))

      it "disables bias" do
        layer.bias?.should be_nil
      end
    end
    context "with `activation: :relu`" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs.merge(activation: :relu))

      it "adds activation" do
        layer.act.should be_a(MXNet::Gluon::NN::Activation)
      end
    end
  end

  describe "#hybrid_forward" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 1, 4, 4])
    weight = MXNet::NDArray.ones(4).reshape(shape: [1, 1, 2, 2])
    bias =  MXNet::NDArray.array([-1_f32])

    context "without bias" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs)
      params = {"weight" => weight}
      output = MXNet::NDArray.array(
        [[[[10, 14, 18],
           [26, 30, 34],
           [42, 46, 50]
          ]]],
        dtype: :float32
      )

      it "runs a forward pass" do
        layer.hybrid_forward([data], params).to_a.should eq([output])
      end
    end

    context "with bias" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs)
      params = {"weight" => weight, "bias" => bias}
      output = MXNet::NDArray.array(
        [[[[ 9, 13, 17],
           [25, 29, 33],
           [41, 45, 49]
          ]]],
        dtype: :float32
      )

      it "runs a forward pass" do
        layer.hybrid_forward([data], params).to_a.should eq([output])
      end
    end

    context "with `strides: 2`" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs.merge(strides: 2))
      params = {"weight" => weight, "bias" => bias}
      output = MXNet::NDArray.array(
        [[[[ 9, 17],
           [41, 49]
          ]]],
        dtype: :float32
      )

      it "convolves with a stride of 2" do
        layer.hybrid_forward([data], params).to_a.should eq([output])
      end
    end

    context "with `padding: 1`" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs.merge(padding: 1))
      params = {"weight" => weight, "bias" => bias}
      output = MXNet::NDArray.array(
        [[[[-1,  0,  2,  4,  2],
           [ 3,  9, 13, 17,  9],
           [11, 25, 29, 33, 17],
           [19, 41, 45, 49, 25],
           [11, 24, 26, 28, 14]
          ]]],
        dtype: :float32
      )

      it "pads the input by 1" do
        layer.hybrid_forward([data], params).to_a.should eq([output])
      end
    end

    context "with `dilation: 2`" do
      layer = MXNet::Gluon::NN::Internal::Conv.new(**newargs.merge(dilation: 2))
      params = {"weight" => weight, "bias" => bias}
      output = MXNet::NDArray.array(
        [[[[19, 23],
           [35, 39]
          ]]],
        dtype: :float32
      )

      it "dilates at a rate of 2" do
        layer.hybrid_forward([data], params).to_a.should eq([output])
      end
    end
  end
end

describe MXNet::Gluon::NN::Conv1D do
  context "smoke test" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 4, 4])

    it "runs a forward pass" do
      layer = MXNet::Gluon::NN::Conv1D.new(channels: 1, kernel_size: 3).init
      layer.forward([data]).should be_a(Array(MXNet::NDArray))
    end
  end
end

describe MXNet::Gluon::NN::Conv2D do
  context "smoke test" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 1, 4, 4])

    it "runs a forward pass" do
      layer = MXNet::Gluon::NN::Conv2D.new(channels: 1, kernel_size: 3).init
      layer.forward([data]).should be_a(Array(MXNet::NDArray))
    end
  end
end

describe MXNet::Gluon::NN::Conv3D do
  context "smoke test" do
    data =  MXNet::NDArray.array(0..47, dtype: :float32).reshape(shape: [1, 1, 3, 4, 4])

    it "runs a forward pass" do
      layer = MXNet::Gluon::NN::Conv3D.new(channels: 1, kernel_size: 3).init
      layer.forward([data]).should be_a(Array(MXNet::NDArray))
    end
  end
end

describe MXNet::Gluon::NN::Internal::Pooling do
  newargs = {
    pool_size: [2, 2], strides: 2, padding: 0
  }

  describe "#hybrid_forward" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 1, 4, 4])

    context "with defaults" do
      layer = MXNet::Gluon::NN::Internal::Pooling.new(**newargs)
      output = MXNet::NDArray.array(
        [[[[ 5,  7],
           [13, 15]
          ]]],
        dtype: :float32
      )

      it "pools the input" do
        layer.hybrid_forward([data]).to_a.should eq([output])
      end
    end

    context "with `strides: 1`" do
      layer = MXNet::Gluon::NN::Internal::Pooling.new(**newargs.merge(strides: 1))
      output = MXNet::NDArray.array(
        [[[[ 5,  6,  7],
           [ 9, 10, 11],
           [13, 14, 15]
          ]]],
        dtype: :float32
      )

      it "pools with a stride of 1" do
        layer.hybrid_forward([data]).to_a.should eq([output])
      end
    end

    context "with `padding: 1`" do
      layer = MXNet::Gluon::NN::Internal::Pooling.new(**newargs.merge(padding: 1))
      output = MXNet::NDArray.array(
        [[[[ 0,  2,  3],
           [ 8, 10, 11],
           [12, 14, 15]
          ]]],
        dtype: :float32
      )

      it "pools with a padding of 1" do
        layer.hybrid_forward([data]).to_a.should eq([output])
      end
    end
  end
end

describe MXNet::Gluon::NN::MaxPool1D do
  context "smoke test" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 4, 4])

    it "runs a forward pass" do
      layer = MXNet::Gluon::NN::MaxPool1D.new.init
      layer.forward([data]).should be_a(Array(MXNet::NDArray))
    end
  end
end

describe MXNet::Gluon::NN::MaxPool2D do
  context "smoke test" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 1, 4, 4])

    it "runs a forward pass" do
      layer = MXNet::Gluon::NN::MaxPool2D.new.init
      layer.forward([data]).should be_a(Array(MXNet::NDArray))
    end
  end
end

describe MXNet::Gluon::NN::MaxPool3D do
  context "smoke test" do
    data =  MXNet::NDArray.array(0..47, dtype: :float32).reshape(shape: [1, 1, 3, 4, 4])

    it "runs a forward pass" do
      layer = MXNet::Gluon::NN::MaxPool3D.new.init
      layer.forward([data]).should be_a(Array(MXNet::NDArray))
    end
  end
end

describe MXNet::Gluon::NN::Flatten do
  describe "#hybrid_forward" do
    data =  MXNet::NDArray.array(0..15, dtype: :float32).reshape(shape: [1, 1, 4, 4])

    it "flattens the input" do
      layer = MXNet::Gluon::NN::Flatten.new
      output = MXNet::NDArray.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
        dtype: :float32
      )

      layer.hybrid_forward([data]).to_a.should eq([output])
    end
  end
end
