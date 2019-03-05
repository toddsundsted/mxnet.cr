require "../../spec_helper"

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
end
