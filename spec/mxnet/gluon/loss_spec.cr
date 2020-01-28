require "../../spec_helper"

describe MXNet::Gluon::Loss::L1Loss do
  describe "#call" do
    loss = MXNet::Gluon::Loss::L1Loss.new

    it "calculates L1 loss" do
      prediction = MXNet::NDArray.array([1.0])
      label = MXNet::NDArray.array([2.0])
      loss.call([prediction, label]).first.as_scalar.should eq(1.0)
    end

    context "when called with Symbol" do
      it "calculates L1 loss" do
        prediction = MXNet::Symbol.var("prediction")
        label = MXNet::Symbol.var("label")
        input = {MXNet::NDArray.array([1.0]), MXNet::NDArray.array([2.0])}
        loss.call([prediction, label]).first.eval(*input).first.as_scalar.should eq(1.0)
      end
    end
  end
end

describe MXNet::Gluon::Loss::L2Loss do
  describe "#call" do
    loss = MXNet::Gluon::Loss::L2Loss.new

    it "calculates L2 loss" do
      prediction = MXNet::NDArray.array([1.0])
      label = MXNet::NDArray.array([2.0])
      loss.call([prediction, label]).first.as_scalar.should eq(0.5)
    end

    context "when called with Symbol" do
      it "calculates L2 loss" do
        prediction = MXNet::Symbol.var("prediction")
        label = MXNet::Symbol.var("label")
        input = {MXNet::NDArray.array([1.0]), MXNet::NDArray.array([2.0])}
        loss.call([prediction, label]).first.eval(*input).first.as_scalar.should eq(0.5)
      end
    end
  end
end

describe MXNet::Gluon::Loss::SoftmaxCrossEntropyLoss do
  describe "#call" do
    context "au naturale" do
      loss = MXNet::Gluon::Loss::SoftmaxCrossEntropyLoss.new

      it "calculates softmax cross-entropy loss" do
        prediction = MXNet::NDArray.array([0.0, 1.0, 0.0])
        label = MXNet::NDArray.array([1.0])
        loss.call([prediction, label]).first.as_scalar.should be_close(0.551, 0.001)
      end
    end

    context "when created with `sparse_label: false`" do
      loss = MXNet::Gluon::Loss::SoftmaxCrossEntropyLoss.new(sparse_label: false)

      it "calculates softmax cross-entropy loss" do
        prediction = MXNet::NDArray.array([0.0, 1.0, 0.0])
        label = MXNet::NDArray.array([0.0, 1.0, 0.0])
        loss.call([prediction, label]).first.as_scalar.should be_close(0.551, 0.001)
      end
    end

    context "when created with `from_logits: true`" do
      loss = MXNet::Gluon::Loss::SoftmaxCrossEntropyLoss.new(from_logits: true)

      it "calculates softmax cross-entropy loss" do
        prediction = MXNet::NDArray.array([0.0, -1.0, 0.0])
        label = MXNet::NDArray.array([1.0])
        loss.call([prediction, label]).first.as_scalar.should be_close(1.000, 0.001)
      end
    end
  end
end
