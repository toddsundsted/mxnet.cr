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

  describe ".mark_variables" do
    context "first and second arguments" do
      it "must be the same length" do
        expect_raises(ArgumentError) do
          MXNet::Autograd.mark_variables(
            [MXNet::NDArray.array([1]), MXNet::NDArray.array([2, 3])],
            [MXNet::NDArray.array([1, 2, 3])]
          )
        end
        MXNet::Autograd.mark_variables(
          [] of MXNet::NDArray,
          [] of MXNet::NDArray
        )
      end
    end

    it "attaches grads to vars" do
      vars = [MXNet::NDArray.random_uniform(shape: 1)]
      grads = [MXNet::NDArray.random_uniform(shape: 1)]
      MXNet::Autograd.mark_variables(vars, grads)
      vars.first.grad.should eq(grads.first)
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
