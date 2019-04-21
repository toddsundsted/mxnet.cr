require "../../spec_helper"

describe MXNet::CachedOp do
  describe "#call" do
    a = MXNet::Symbol.var("a")
    b = MXNet::Symbol.var("b")
    cached_op = MXNet::CachedOp.new(2.25 * a + b, {} of String => String)

    it "calls the cached op" do
      x = cached_op.call([MXNet::NDArray.array([0.0, 1.0]), MXNet::NDArray.array([0.5, 0.5])])
      x.should eq([MXNet::NDArray.array([0.5, 2.75])])
    end

    it "writes the results to the output array" do
      y = MXNet::NDArray.array([0.0, 0.0])
      x = cached_op.call([MXNet::NDArray.array([0.0, 1.0]), MXNet::NDArray.array([0.5, 0.5])], out: y)
      x.should eq([MXNet::NDArray.array([0.5, 2.75])])
      x.first.should be(y)
    end
  end
end
