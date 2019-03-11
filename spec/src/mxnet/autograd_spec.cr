require "../../spec_helper"

describe "MXNet::Autograd" do
  describe ".is_recording" do
    it "is false, by default" do
      MXNet::Autograd.is_recording.should be_false
    end
  end

  describe ".is_training" do
    it "is false, by default" do
      MXNet::Autograd.is_training.should be_false
    end
  end

  describe ".record" do
    it "enables and restores record mode" do
      MXNet::Autograd.record do
        MXNet::Autograd.is_recording.should be_true
      end
      MXNet::Autograd.is_recording.should be_false
    end
  end

  describe ".pause" do
    it "disables and restores record mode" do
      MXNet::Autograd.record do
        MXNet::Autograd.pause do
          MXNet::Autograd.is_recording.should be_false
        end
        MXNet::Autograd.is_recording.should be_true
      end
    end
  end

  describe ".train_mode" do
    it "enables and restores train mode" do
      MXNet::Autograd.train_mode do
        MXNet::Autograd.is_training.should be_true
      end
      MXNet::Autograd.is_training.should be_false
    end
  end

  describe ".predict_mode" do
    it "disables and restores train mode" do
      MXNet::Autograd.train_mode do
        MXNet::Autograd.predict_mode do
          MXNet::Autograd.is_training.should be_false
        end
        MXNet::Autograd.is_training.should be_true
      end
    end
  end

  it "computes gradient" do
    x = MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]).attach_grad
    y = MXNet::Autograd.record do
      x * x + 1
    end
    y.backward
    x.grad.to_a.should eq([2.0, 4.0, 6.0, 8.0])
  end
end
