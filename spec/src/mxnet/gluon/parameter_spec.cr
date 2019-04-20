require "../../../spec_helper"
require "../../../../src/mxnet/gluon"

describe MXNet::Gluon::Parameter do
  describe ".new" do
    MXNet::Gluon::Parameter.new("test", shape: 1).shape.should eq([1])
    MXNet::Gluon::Parameter.new("test", shape: [1]).shape.should eq([1])
  end

  describe "#init" do
    context "without deferred initialization" do
      parameter =
        MXNet::Gluon::Parameter.new("foo", shape: [1]).tap do |parameter|
          parameter.init
        end

      it "initializes the data array" do
        parameter.data.should be_a(MXNet::NDArray)
      end
      it "initializes the grad array" do
        parameter.grad.should be_a(MXNet::NDArray)
      end
      it "attaches grads" do
        parameter.list_data.zip(parameter.list_grad).each do |p, g|
          p.grad.should eq(g)
        end
      end
    end

    context "with deferred initialization" do
      parameter =
        MXNet::Gluon::Parameter.new("foo", allow_deferred_init: true).tap do |parameter|
          parameter.init
          parameter.shape = [1]
          parameter._finish_deferred_init
        end

      it "initializes the data array" do
        parameter.data.should be_a(MXNet::NDArray)
      end
      it "initializes the grad array" do
        parameter.grad.should be_a(MXNet::NDArray)
      end
      it "attaches grads" do
        parameter.list_data.zip(parameter.list_grad).each do |p, g|
          p.grad.should eq(g)
        end
      end
    end

    context "with `grad_req: :null`" do
      parameter =
        MXNet::Gluon::Parameter.new("foo", shape: [1], grad_req: :null).tap do |parameter|
          parameter.init
        end

      it "does not attach a grad array" do
        expect_raises(Exception, /Cannot get gradient buffers/) do
          parameter.list_grad
        end
        expect_raises(Exception, /Cannot get gradient buffer/) do
          parameter.grad
        end
      end
    end

    context "for 'init'" do
      parameter = MXNet::Gluon::Parameter.new("foo", shape: 1)

      it "accepts a class" do
        parameter.init(init: MXNet::Initializer::Zero)
        parameter.data.to_a.should eq([0])
      end
      it "accepts an instance" do
        parameter.init(init: MXNet::Initializer::Zero.new)
        parameter.data.to_a.should eq([0])
      end
      it "accepts a string" do
        parameter.init(init: "zeros")
        parameter.data.to_a.should eq([0])
      end
      it "accepts a symbol" do
        parameter.init(init: :zeros)
        parameter.data.to_a.should eq([0])
      end
    end

    context "for 'default_init'" do
      parameter = MXNet::Gluon::Parameter.new("foo", shape: 1)

      it "accepts a class" do
        parameter.init(default_init: MXNet::Initializer::Zero)
        parameter.data.to_a.should eq([0])
      end
      it "accepts an instance" do
        parameter.init(default_init: MXNet::Initializer::Zero.new)
        parameter.data.to_a.should eq([0])
      end
      it "accepts a string" do
        parameter.init(default_init: "zeros")
        parameter.data.to_a.should eq([0])
      end
      it "accepts a symbol" do
        parameter.init(default_init: :zeros)
        parameter.data.to_a.should eq([0])
      end
    end
  end

  describe "#list_ctx" do
    context "without deferred initialization" do
      parameter =
        MXNet::Gluon::Parameter.new("foo", shape: [1, 2]).tap do |parameter|
          parameter.init
        end

      it "returns the contexts" do
        parameter.list_ctx.should eq([MXNet.cpu])
      end
    end
    context "with deferred initialization" do
      parameter =
        MXNet::Gluon::Parameter.new("foo", allow_deferred_init: true).tap do |parameter|
          parameter.init
        end

      it "returns the contexts" do
        parameter.list_ctx.should eq([MXNet.cpu])
      end
    end
  end

  describe "#var" do
    context "without shape and dtype" do
      parameter = MXNet::Gluon::Parameter.new("foo")

      it "returns a symbol" do
        parameter.var.should be_a(MXNet::Symbol)
      end
      it "has the name of the parameter" do
        parameter.var.name.should eq("foo")
      end
      it "has no attributes by default" do
        parameter.var.list_attr.should be_empty
      end
    end

    context "with shape and dtype" do
      parameter = MXNet::Gluon::Parameter.new("bar", shape: [1], dtype: :int32)

      it "has attributes for shape and dtype" do
        parameter.var.list_attr.should eq({"__shape__" => "[1]", "__dtype__" => "4"})
      end
    end
  end

  describe "#data" do
    parameter = MXNet::Gluon::Parameter.new("foo", shape: [1])

    it "fails if the parameter has not been initialized" do
      expect_raises(Exception) do
        parameter.data
      end
    end
    it "fails if the parameter has not been initialized on the specified context" do
      parameter.init(ctx: MXNet.cpu)
      expect_raises(Exception) do
        parameter.data(ctx: MXNet.gpu)
      end
    end
    it "succeeds if the parameter has been initialized on the specified context" do
      parameter.init(ctx: MXNet.cpu)
      parameter.data(ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
    it "returns the initialized data" do
      parameter.init
      parameter.data.should be_a(MXNet::NDArray)
    end
  end

  describe "#set_data" do
    parameter = MXNet::Gluon::Parameter.new("foo", shape: [1])
    data = MXNet::NDArray.ones([1])

    it "fails if the parameter has not been initialized" do
      expect_raises(Exception) do
        parameter.set_data(data)
      end
    end
    it "sets the data" do
      parameter.init
      parameter.set_data(data)
      parameter.data.to_a.should eq([1])
    end
  end

  describe "#grad" do
    parameter = MXNet::Gluon::Parameter.new("foo", shape: [1])

    it "fails if the parameter has not been initialized" do
      expect_raises(Exception) do
        parameter.grad
      end
    end
    it "fails if the parameter has not been initialized on the specified context" do
      parameter.init(ctx: MXNet.cpu)
      expect_raises(Exception) do
        parameter.grad(ctx: MXNet.gpu)
      end
    end
    it "succeeds if the parameter has been initialized on the specified context" do
      parameter.init(ctx: MXNet.cpu)
      parameter.grad(ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
    it "returns the initialized grad" do
      parameter.init
      parameter.grad.should be_a(MXNet::NDArray)
    end
  end

  describe "#zero_grad" do
    parameter = MXNet::Gluon::Parameter.new("foo", shape: [1])

    it "fails if the parameter has not been initialized" do
      expect_raises(Exception) do
        parameter.zero_grad
      end
    end
    it "sets the grad" do
      parameter.init
      parameter.zero_grad
      parameter.grad.to_a.should eq([0])
    end
  end

  describe "#shape=" do
    context "with no shape" do
      parameter = MXNet::Gluon::Parameter.new("foo")

      it "assigns the shape" do
        parameter.shape = [1, 2]
        parameter.shape.should eq([1, 2])
      end
    end
    context "with incomplete shape" do
      parameter = MXNet::Gluon::Parameter.new("foo", shape: [1, 0, 3])

      it "completes the shape" do
        parameter.shape = [1, 2, 3]
        parameter.shape.should eq([1, 2, 3])
      end
    end
    context "with shape" do
      parameter = MXNet::Gluon::Parameter.new("foo", shape: [1, 2])

      it "raises an error" do
        expect_raises(Exception) do
          parameter.shape = [1, 3]
        end
      end
    end
  end

  describe "#==" do
    parameter = MXNet::Gluon::Parameter.new("foo", shape: [1])

    it "is true if name and shape are equal" do
      (parameter == MXNet::Gluon::Parameter.new("foo", shape: [1])).should be_true
    end
    it "is false if name is not equal" do
      (parameter == MXNet::Gluon::Parameter.new("boo", shape: [1])).should be_false
    end
    it "is false if shape is not equal" do
      (parameter == MXNet::Gluon::Parameter.new("foo", shape: [2])).should be_false
    end
  end

  describe "#_load_init" do
    it "loads the parameter with data" do
      parameter = MXNet::Gluon::Parameter.new("foo", shape: [0], dtype: :int32)
      parameter._load_init(MXNet.cpu, MXNet::NDArray.array([1, 2]))
      parameter.data.should eq(MXNet::NDArray.array([1, 2]))
    end

    it "reloads the parameter with data" do
      parameter = MXNet::Gluon::Parameter.new("foo", shape: [3], dtype: :int32).init(:ones)
      parameter._load_init(MXNet.cpu, MXNet::NDArray.array([1, 2, 3], dtype: :int32))
      parameter.data.should eq(MXNet::NDArray.array([1, 2, 3], dtype: :int32))
    end

    it "raises an error if shapes are incompatible" do
      expect_raises(Exception) do
        parameter = MXNet::Gluon::Parameter.new("foo", shape: [1])
        parameter._load_init(MXNet.cpu, MXNet::NDArray.array([1, 2]))
      end
    end

    it "raises an error if dtypes are incompatible" do
      expect_raises(Exception) do
        parameter = MXNet::Gluon::Parameter.new("foo", dtype: :float32)
        parameter._load_init(MXNet.cpu, MXNet::NDArray.array([1, 2]))
      end
    end
  end
end

describe MXNet::Gluon::Constant do
  it "creates a constant parameter" do
    const = MXNet::Gluon::Constant.new("const", [[1, 2], [3, 4]]).init
    const.data.should eq(MXNet::NDArray.array([[1, 2], [3, 4]]))
  end
end

describe MXNet::Gluon::ParameterDict do
  describe "#get" do
    context "without a shared dict" do
      parameter_dict = MXNet::Gluon::ParameterDict.new

      it "creates a new parameter if not in dict" do
        parameter_dict.get("foo").should be_a(MXNet::Gluon::Parameter)
      end
      it "retrieves a previously created parameter" do
        parameter_dict.get("bar").should eq(parameter_dict.get("bar"))
      end
      it "uses keyword arguments to create a parameter" do
        parameter_dict.get("baz", shape: [1, 1]).shape.should eq([1, 1])
      end
    end

    context "with a shared dict" do
      shared_dict =
        MXNet::Gluon::ParameterDict.new.tap do |shared_dict|
          shared_dict.get("foo")
        end
      parameter_dict =
        MXNet::Gluon::ParameterDict.new(shared: shared_dict).tap do |parameter_dict|
          parameter_dict.get("bar")
        end

      it "retrieves a parameter from the shared dict" do
        parameter_dict.get("foo").should eq(shared_dict.get("foo"))
      end
    end
  end

  describe "#update" do
    it "copies parameters into dict" do
      other_dict =
        MXNet::Gluon::ParameterDict.new.tap do |other_dict|
          other_dict.get("foo", shape: 1)
        end
      parameter_dict =
        MXNet::Gluon::ParameterDict.new.tap do |parameter_dict|
          parameter_dict.update(other_dict)
        end

      parameter_dict.get("foo").should eq(other_dict.get("foo"))
    end

    it "fails if parameters already exist" do
      other_dict =
        MXNet::Gluon::ParameterDict.new.tap do |other_dict|
          other_dict.get("foo", shape: 1)
        end
      parameter_dict =
        MXNet::Gluon::ParameterDict.new.tap do |parameter_dict|
          parameter_dict.get("foo")
        end

      expect_raises(ArgumentError) do
        parameter_dict.update(other_dict)
      end
    end
  end

  describe "#init" do
    parameter_dict =
      MXNet::Gluon::ParameterDict.new(prefix: "name").tap do |parameter_dict|
        parameter_dict.get("foo", shape: 1)
      end

    it "initializes all parameters" do
      parameter_dict.init

      parameter_dict.get("foo").data.should be_a(MXNet::NDArray)
    end
  end

  describe "#keys" do
    parameter_dict =
      MXNet::Gluon::ParameterDict.new.tap do |parameter_dict|
        parameter_dict.get("test")
      end

    it "is returns the keys" do
      parameter_dict.keys.should eq(["test"])
    end
  end

  describe "#has_key?" do
    parameter_dict =
      MXNet::Gluon::ParameterDict.new.tap do |parameter_dict|
        parameter_dict.get("test")
      end

    it "is true if key exists" do
      parameter_dict.has_key?("test").should be_true
    end
  end

  describe "#values" do
    parameter = nil
    parameter_dict =
      MXNet::Gluon::ParameterDict.new.tap do |parameter_dict|
        parameter = parameter_dict.get("test")
      end

    it "is returns the values" do
      parameter_dict.values.should eq([parameter])
    end
  end

  describe "#has_value?" do
    parameter = nil
    parameter_dict =
      MXNet::Gluon::ParameterDict.new.tap do |parameter_dict|
        parameter = parameter_dict.get("test")
      end

    it "is true if value exists" do
      parameter_dict.has_value?(parameter).should be_true
    end
  end

  describe "#==" do
    parameter_dict =
      MXNet::Gluon::ParameterDict.new(prefix: "name").tap do |parameter_dict|
        parameter_dict.get("test")
      end

    it "is true if prefix and items are equal" do
      other_dict = MXNet::Gluon::ParameterDict.new(prefix: "name").tap do |other_dict|
        other_dict.get("test")
      end

      (parameter_dict == other_dict).should be_true
    end
    it "is false if prefix is not equal" do
      other_dict = MXNet::Gluon::ParameterDict.new(prefix: "other").tap do |other_dict|
        other_dict.get("test")
      end

      (parameter_dict == other_dict).should be_false
    end
    it "is false items is not equal" do
      other_dict = MXNet::Gluon::ParameterDict.new(prefix: "name").tap do |other_dict|
        other_dict.get("other")
      end

      (parameter_dict == other_dict).should be_false
    end
  end

  describe "#to_a" do
    parameter_dict =
      MXNet::Gluon::ParameterDict.new(prefix: "name").tap do |parameter_dict|
        parameter_dict.get("test")
      end

    it "should convert the dict into an array" do
      parameter_dict.to_a.should eq([{"nametest", parameter_dict.get("test")}])
    end
  end
end
