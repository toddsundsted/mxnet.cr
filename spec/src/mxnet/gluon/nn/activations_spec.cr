require "../../../../spec_helper"

describe MXNet::Gluon::NN::Activation do
  describe ".new" do
    it "accepts a string" do
      act = MXNet::Gluon::NN::Activation.new("relu")
      act.forward([MXNet::NDArray.array([-1.0, 0.0, 1.0])])
        .should eq([MXNet::NDArray.array([0.0, 0.0, 1.0])])
    end

    it "accepts a symbol" do
      act = MXNet::Gluon::NN::Activation.new(:relu)
      act.forward([MXNet::NDArray.array([-1.0, 0.0, 1.0])])
        .should eq([MXNet::NDArray.array([0.0, 0.0, 1.0])])
    end
  end
end
