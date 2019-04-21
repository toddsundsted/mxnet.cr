require "../../spec_helper"

class UtilTest
  extend MXNet::Util
end

describe MXNet::Util do
  describe ".output" do
    it "recursively pretty-prints it's argument" do
      UtilTest.output([1_u8, 2_i64, -3_f32]).should eq("[1,2,-3.0]")
      UtilTest.output([nil, :symbol, "string"]).should eq("[None,symbol,string]")
      UtilTest.output([[1]]).should eq("[[1]]")
      UtilTest.output({0}).should eq("[0]")
    end
  end
end

describe MXNet::NDArray do
  it "pretty-prints keyword arguments" do
    MXNet::NDArray::Internal._zeros(shape: [3_u8]).to_a.should eq([0.0, 0.0, 0.0])
  end
end

describe MXNet::Symbol do
  it "pretty-prints keyword arguments" do
    MXNet::Symbol::Internal._zeros(shape: [3_u8]).eval.first.to_a.should eq([0.0, 0.0, 0.0])
  end
end
