require "../../../spec_helper"
require "../../../../src/mxnet/gluon"

describe MXNet::Gluon::Block do
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
end

describe MXNet::Gluon::HybridBlock do
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
end
