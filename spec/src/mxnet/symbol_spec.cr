require "../../spec_helper"

class MXNet::NDArray
  # Redefined this method for testing. The original method
  # performs an element-wise comparison for equality.
  def ==(other : self)
    return false unless other.dtype == self.dtype
    return false unless other.shape == self.shape
    return false unless other.raw == self.raw
    true
  end
end

describe "MXNet::Symbol" do
  describe ".var" do
    it "creates a symbolic variable" do
      MXNet::Symbol.var("s").should be_a(MXNet::Symbol)
    end
  end

  describe "#name" do
    it "returns the name of the symbol" do
      MXNet::Symbol.var("foo").name.should eq("foo")
    end
  end

  describe "#list_arguments" do
    it "returns the arguments of the symbol" do
      MXNet::Symbol.var("foo").list_arguments.should eq(["foo"])
    end
  end

  describe "#list_outputs" do
    it "returns the outputs of the symbol" do
      MXNet::Symbol.var("foo").list_outputs.should eq(["foo"])
    end
  end

  describe "#bind" do
    it "binds an array of arguments" do
      MXNet::Symbol.var("a").bind(args: [MXNet::NDArray.array([1])]).should be_a(MXNet::Executor)
    end
    it "binds a hash of arguments" do
      MXNet::Symbol.var("a").bind(args: {"a" => MXNet::NDArray.array([1])}).should be_a(MXNet::Executor)
    end
  end

  describe "#eval" do
    it "evaluates arguments" do
      MXNet::Symbol.var("a").eval(MXNet::NDArray.array([1])).to_a.should be_a(Array(MXNet::NDArray))
    end
    it "evaluates named arguments" do
      MXNet::Symbol.var("a").eval(a: MXNet::NDArray.array([1])).to_a.should be_a(Array(MXNet::NDArray))
    end
  end

  describe "#to_s" do
    it "pretty-prints the symbol" do
      MXNet::Symbol.var("foo").to_s.should eq("<Symbol foo>")
    end
  end

  args = {
    a: MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]]),
    b: MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]]),
    c: MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
  }

  describe "#+" do
    it "adds a scalar to an array" do
      a = MXNet::Symbol.var("a")
      (a + 2).eval(**args).first.should eq(MXNet::NDArray.array([[3.0, 4.0], [5.0, 6.0]]))
    end
    it "adds two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a + b).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe "#-" do
    it "subtracts a scalar from an array" do
      a = MXNet::Symbol.var("a")
      (a - 2).eval(**args).first.should eq(MXNet::NDArray.array([[-1.0, 0.0], [1.0, 2.0]]))
    end
    it "subtracts two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a - b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
  end

  describe "#*" do
    it "multiplies an array by a scalar" do
      a = MXNet::Symbol.var("a")
      (a * 2).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 4.0], [6.0, 8.0]]))
    end
    it "multiplies two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a * b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 8.0], [3.0, 4.0]]))
    end
  end

  describe "#/" do
    it "divides an array by a scalar" do
      a = MXNet::Symbol.var("a")
      (a / 2).eval(**args).first.should eq(MXNet::NDArray.array([[0.5, 1.0], [1.5, 2.0]]))
    end
    it "divides two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a / b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.5], [3.0, 4.0]]))
    end
  end

  describe "#**" do
    it "exponentiates an array by a scalar" do
      a = MXNet::Symbol.var("a")
      (a ** 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 4.0], [9.0, 16.0]]))
    end
    it "exponentiates two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a ** b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 16.0], [3.0, 4.0]]))
    end
  end

  describe "#reshape" do
    it "reshapes the input array" do
      a = MXNet::Symbol.var("a")
      a.reshape(shape: [4]).eval(**args).first.should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
      a.reshape([4]).eval(**args).first.should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
      a.reshape(4).eval(**args).first.should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
    end

    it "supports special values for dimensions" do
      c = MXNet::Symbol.var("c")
      c.reshape([-1, 0], reverse: false).eval(**args).first.shape.should eq([2_u32, 4_u32])
      c.reshape([-1, 0], reverse: true).eval(**args).first.shape.should eq([4_u32, 2_u32])
    end
  end

  describe "#flatten" do
    it "flattens the input array" do
      c = MXNet::Symbol.var("c")
      c.flatten.eval(**args).first.shape.should eq([1_u32, 8_u32])
    end
  end
end
