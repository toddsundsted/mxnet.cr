require "../../spec_helper"

class BaseTest < MXNet::Base
  def self.output(arg)
    super
  end
end

describe "MXNet::Base" do
  describe ".output" do
    it "recursively pretty-prints it's argument" do
      BaseTest.output([1_u8, 2_i64, -3_f32]).should eq("[1,2,-3.0]")
      BaseTest.output([nil, :symbol, "string"]).should eq("[None,symbol,string]")
      BaseTest.output([[1]]).should eq("[[1]]")
      BaseTest.output({0}).should eq("[0]")
    end
  end
end

describe "MXNet::NDArray" do
  it "pretty-prints keyword arguments" do
    MXNet::NDArray::Internal._zeros(shape: [3_u8]).first.to_a.should eq([0.0, 0.0, 0.0])
  end
end

describe "MXNet::Symbol" do
  it "pretty-prints keyword arguments" do
    MXNet::Symbol::Internal._zeros(shape: [3_u8]).eval.first.to_a.should eq([0.0, 0.0, 0.0])
  end
end
