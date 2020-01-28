require "../spec_helper"

describe MXNet::Random do
  describe ".seed" do
    it "seeds random number generators for all devices" do
      MXNet::Random.seed(123).should be_nil
    end

    it "seeds the random number generator for the CPU device" do
      MXNet::Random.seed(123, MXNet.cpu).should be_nil
    end
  end
end
