require "../../spec_helper"

class MXNet::Optimizer
  def get_lr(index)
    previous_def
  end

  def get_wd(index)
    previous_def
  end
end

describe MXNet::Optimizer do
  describe ".create" do
    it "accepts a class" do
      opt = MXNet::Optimizer.create(MXNet::Optimizer::SGD)
      opt.should be_a(MXNet::Optimizer::SGD)
    end
    it "accepts an instance" do
      opt = MXNet::Optimizer.create(MXNet::Optimizer::SGD.new)
      opt.should be_a(MXNet::Optimizer::SGD)
    end
    it "accepts a string" do
      opt = MXNet::Optimizer.create("sgd")
      opt.should be_a(MXNet::Optimizer::SGD)
    end
    it "accepts a symbol" do
      opt = MXNet::Optimizer.create(:sgd)
      opt.should be_a(MXNet::Optimizer::SGD)
    end
  end
end

describe MXNet::Optimizer::SGD do
  describe "#update" do
    optimizer = MXNet::Optimizer::SGD.new(lr: 0.1)

    it "updates the weight" do
      weight = MXNet::NDArray.array([1.0])
      gradient = MXNet::NDArray.array([0.5])
      optimizer.update(0, weight, gradient, nil)
      weight.should be_close(0.95, 0.01)
    end
  end

  describe "#get_lr" do
    optimizer = MXNet::Optimizer::SGD.new(lr: 0.1)

    it "returns the product of the learning rates" do
      optimizer.set_lr_mult({1 => 0.1})
      optimizer.get_lr(1).should be_close(0.01, 0.001)
    end
  end

  describe "#get_wd" do
    optimizer = MXNet::Optimizer::SGD.new(wd: 0.2)

    it "returns the product of the learning rates" do
      optimizer.set_wd_mult({1 => 0.2})
      optimizer.get_wd(1).should be_close(0.04, 0.001)
    end
  end
end
