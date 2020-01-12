require "../../../spec_helper"
require "../../../../src/mxnet/gluon"

describe MXNet::Gluon::BlockScope do
  describe ".create" do
    it "creates prefix with a counter" do
      prefix, _ = MXNet::Gluon::BlockScope.create(nil, nil, "foo")
      prefix.should match(/^foo[0-9]+_$/)
    end

    it "creates a ParameterDict with a prefix" do
      _, params = MXNet::Gluon::BlockScope.create(nil, nil, "foo")
      params.prefix.should match(/^foo[0-9]+_$/)
    end
  end
end

create_double(MXNet::Gluon::Block)

class FooBar < MXNet::Gluon::Block
  attribute \
    foo : MXNet::Gluon::Parameter | MXNet::Gluon::Block,
    bar : MXNet::Gluon::Parameter | MXNet::Gluon::Block
end

describe MXNet::Gluon::Block do
  context "attribute access" do
    block = MXNet::Gluon::Block.new

    it "sets a block attribute" do
      a = block.set_attr("a", MXNet::Gluon::Block.new)
      block.get_attr("a").should be(a)
    end

    it "sets a parameter attribute" do
      b = block.set_attr("b", MXNet::Gluon::Parameter.new("b"))
      block.get_attr("b").should be(b)
    end

    it "raises exception if attribute is set to another type" do
      expect_raises(Exception) do
        block.set_attr("c", MXNet::Gluon::Block.new)
        block.set_attr("c", MXNet::Gluon::Parameter.new("c"))
      end
    end

    it "raises exception if attribute is not set" do
      expect_raises(Exception) do
        block.get_attr("d")
      end
    end
  end

  context "attribute assignment" do
    block = FooBar.new

    it "assigns a block attribute" do
      foo = block.foo = MXNet::Gluon::Block.new
      block.foo.should be(foo)
    end

    it "assigns a parameter attribute" do
      bar = block.bar = MXNet::Gluon::Parameter.new("bar")
      block.bar.should be(bar)
    end
  end

  describe "#new" do
    context "au naturale" do
      block = MXNet::Gluon::Block.new

      it "assigns a unique prefix" do
        block.prefix.should_not eq(MXNet::Gluon::Block.new.prefix)
        block.prefix.should match(/^block[0-9]+_$/)
      end
    end

    context "with prefix" do
      block = MXNet::Gluon::Block.new(prefix: "foo")

      it "uses the assigned prefix" do
        block.prefix.should eq("foo")
      end
    end

    context "with params" do
      params = MXNet::Gluon::ParameterDict.new.tap do |params|
        params.get("foo")
      end

      block = MXNet::Gluon::Block.new(params: params)

      it "shares the assigned params" do
        block.params.get("foo").should be(params.get("foo"))
      end
    end
  end

  describe "#with_name_scope" do
    block = FooBar.new

    it "prepends prefixes to scoped blocks" do
      block.with_name_scope do
        block.foo = FooBar.new
        block.foo.as(FooBar).with_name_scope do
          block.foo.as(FooBar).bar = FooBar.new
        end
      end
      block.foo.as(FooBar).prefix.should match(/^foobar[0-9]+_foobar0_$/)
      block.foo.as(FooBar).bar.as(FooBar).prefix.should match(/^foobar[0-9]+_foobar0_foobar0_$/)
    end
  end

  describe "#collect_params" do
    block =
      MXNet::Gluon::Block.new(prefix: "block_").tap do |block|
        block.params.get("foo")
        block.params.get("bar")
        block.params.get("baz")
      end
    child =
      MXNet::Gluon::Block.new(prefix: "block_").tap do |block|
        block.params.get("qoz")
      end

    it "returns all its parameters" do
      params = MXNet::Gluon::ParameterDict.new(prefix: "block_").tap do |params|
        params.get("foo")
        params.get("bar")
        params.get("baz")
      end
      block.collect_params.should eq(params)
    end

    it "returns the matching parameters" do
      params = MXNet::Gluon::ParameterDict.new(prefix: "block_").tap do |params|
        params.get("bar")
        params.get("baz")
      end
      block.collect_params(/_ba/).should eq(params)
    end

    it "returns matching parameters from children" do
      block.set_attr("qoz", child)
      params = MXNet::Gluon::ParameterDict.new(prefix: "block_").tap do |params|
        params.get("qoz")
      end
      block.collect_params(/_q/).should eq(params)
    end
  end

  describe "#init" do
    block =
      MXNet::Gluon::Block.new.tap do |block|
        block.params.get("foo", shape: 1)
      end

    it "initializes all parameters" do
      block.init
      block.params.get("foo").data.should be_a(MXNet::NDArray)
    end
  end

  describe "#register_child" do
    block = MXNet::Gluon::Block.new
    child = MXNet::Gluon::Block.new

    it "registers block" do
      block.register_child(child)
      block.get_attr("0").should eq(child)
    end
  end

  describe "#register_parameter" do
    block = MXNet::Gluon::Block.new
    param = MXNet::Gluon::Parameter.new("param")

    it "registers parameter" do
      block.register_parameter(param)
      block.get_attr("param").should eq(param)
    end
  end

  describe "#hybridize" do
    block = MXNet::Gluon::Block.new
    child = instance_double(MXNet::Gluon::Block)

    it "calls hybridize on its children" do
      block.register_child(child)
      block.hybridize
      verify(child, hybridize: once)
    end
  end

  describe "#forward" do
    context "given a Symbol" do
      it "must be implemented in a subclass" do
        expect_raises(NotImplementedError) do
          MXNet::Gluon::Block.new.forward([] of MXNet::Symbol)
        end
      end
    end

    context "given an NDArray" do
      it "must be implemented in a subclass" do
        expect_raises(NotImplementedError) do
          MXNet::Gluon::Block.new.forward([] of MXNet::NDArray)
        end
      end
    end
  end

  describe "#save_parameters" do
    temp = File.tempname("save")
    block =
      FooBar.new.tap do |block|
        block.foo = block.params.get("foo", shape: [2], init: :zeros)
        block.init
      end
    data = Bytes[
      0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xc9, 0xfa, 0x93, 0xf9, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x66, 0x6f, 0x6f
    ]

    it "saves the parameter data" do
      block.save_parameters(temp)
      buffer = Bytes.new(83)
      File.open(temp, "r") do |file|
        file.read(buffer)
        buffer.should eq(data)
      end
    end
  end

  describe "#load_parameters" do
    temp = File.tempname("load")
    block =
      FooBar.new.tap do |block|
        block.foo = block.params.get("foo", shape: [2], init: :zeros)
        block.init
      end
    data = Bytes[
      0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xc9, 0xfa, 0x93, 0xf9, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x98, 0xf6, 0x54, 0x3c, 0xdc, 0xcb, 0xf6, 0x3c,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x66, 0x6f, 0x6f
    ]

    it "loads the parameter data" do
      File.open(temp, "w") do |file|
        file.write(data)
        file.close
        block.load_parameters(temp)
        a, b = block.as(FooBar).foo.as(MXNet::Gluon::Parameter).data.to_a
        a.should be_close(0.0129982, 0.0001)
        b.should be_close(0.0301265, 0.0001)
      end
    end

    it "raises an error if parameter not present in file" do
      block =
        FooBar.new.tap do |block|
          block.bar = block.params.get("bar", shape: [2], init: :zeros)
          block.init
        end

      expect_raises(Exception, /`allow_missing` to `true`/) do
        File.open(temp, "w") do |file|
          file.write(data)
          file.close
          block.load_parameters(temp)
        end
      end
    end

    it "raises an error if value not present in block" do
      block =
        MXNet::Gluon::Block.new.tap do |block|
          block.init
        end

      expect_raises(Exception, /`ignore_extra` to `true`/) do
        File.open(temp, "w") do |file|
          file.write(data)
          file.close
          block.load_parameters(temp)
        end
      end
    end
  end
end

class Quux < MXNet::Gluon::HybridBlock
  attribute c : MXNet::Gluon::Parameter

  def initialize(**kwargs)
    super
    with_name_scope do
      self.c = params.get("c", init: :ones, allow_deferred_init: true)
    end
  end

  def hybrid_forward(inputs, params)
    [inputs.reduce(&.+) + params["c"]]
  end
end

create_double(Quux)

describe MXNet::Gluon::HybridBlock do
  describe "#register_child" do
    block = MXNet::Gluon::HybridBlock.new
    child = MXNet::Gluon::Block.new

    it "doesn't accept a block" do
      expect_raises(Exception) do
        block.register_child(child)
      end
    end
  end

  describe "#hybrid_forward" do
    context "given a Symbol" do
      it "must be implemented in a subclass" do
        expect_raises(NotImplementedError) do
          MXNet::Gluon::HybridBlock.new.hybrid_forward([] of MXNet::Symbol)
        end
      end
    end

    context "given an NDArray" do
      it "must be implemented in a subclass" do
        expect_raises(NotImplementedError) do
          MXNet::Gluon::HybridBlock.new.hybrid_forward([] of MXNet::NDArray)
        end
      end
    end
  end

  context "given a subclass" do
    describe "#forward" do
      it "should infer the shape from the input" do
        block = Quux.new.init
        block.forward([MXNet::NDArray.array([1, 2, 3])])
        block.c.shape.should eq([3])
      end

      it "should infer the dtype from the input" do
        block = Quux.new.init
        block.forward([MXNet::NDArray.array([1, 2, 3])])
        block.c.dtype.should eq(:int32)
      end

      it "should cache the operation" do
        block = instance_double(Quux).init.tap do |block|
          block.hybridize
        end
        block.forward([MXNet::NDArray.array([1, 2, 3])]).map(&.to_a).should eq([[2, 3, 4]])
        block.forward([MXNet::NDArray.array([2, 3, 1])]).map(&.to_a).should eq([[3, 4, 2]])
        block.forward([MXNet::NDArray.array([3, 1, 2])]).map(&.to_a).should eq([[4, 2, 3]])
        verify(block, hybrid_forward: once)
      end
    end
  end
end

class CachedBlock < MXNet::Gluon::Block
  include MXNet::Gluon::CachedGraph

  attribute c : MXNet::Gluon::Parameter

  def initialize
    super
    with_name_scope do
      self.c = params.get("c", init: :ones, allow_deferred_init: true)
    end
  end

  def hybrid_forward(inputs, params)
    [inputs.reduce(&.+) + params["c"]]
  end

  def call_cached(args)
    super
  end
end

describe MXNet::Gluon::CachedGraph do
  describe "#infer_shape" do
    it "should infer the shape from the input" do
      block = CachedBlock.new
      block.infer_shape([MXNet::NDArray.array([1, 2, 3])])
      block.c.shape.should eq([3])
    end
  end

  describe "#infer_dtype" do
    it "should infer the dtype from the input" do
      block = CachedBlock.new
      block.infer_dtype([MXNet::NDArray.array([1, 2, 3])])
      block.c.dtype.should eq(:int32)
    end
  end

  describe "#call_cached" do
    it "calls the cached operator" do
      block = CachedBlock.new.init
      res = block.call_cached([MXNet::NDArray.array([1.0, 1.5, 2.0])])
      res.should eq([MXNet::NDArray.array([2.0, 2.5, 3.0])])
    end
  end
end
