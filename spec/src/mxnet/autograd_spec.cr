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

  it "records computation history for automatic differentiation" do
    x = MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0])
    g = MXNet::NDArray.array([0.0, 0.0, 0.0, 0.0])
    MXNet::Autograd.mark_variables(x, g)
    y = MXNet::Autograd.record do
      x * x + 1
    end
    MXNet::Autograd.backward(y)
  end
end
