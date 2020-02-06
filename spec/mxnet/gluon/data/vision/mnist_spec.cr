require "../../../../spec_helper"

describe MXNet::Gluon::Data::Vision::MNIST do
  describe ".new" do
    train_data = File.join(Dir.tempdir, "train-images-idx3-ubyte.gz")
    train_label = File.join(Dir.tempdir, "train-labels-idx1-ubyte.gz")
    test_data = File.join(Dir.tempdir, "t10k-images-idx3-ubyte.gz")
    test_label = File.join(Dir.tempdir, "t10k-labels-idx1-ubyte.gz")

    it "downloads and caches training files" do
      MXNet::Gluon::Data::Vision::MNIST.new(root: Dir.tempdir, train: true)
      File.exists?(train_data).should be_true
      File.exists?(train_label).should be_true
    ensure
      File.exists?(train_data) && File.delete(train_data)
      File.exists?(train_label) && File.delete(train_label)
    end

    it "downloads and caches testing files" do
      MXNet::Gluon::Data::Vision::MNIST.new(root: Dir.tempdir, train: false)
      File.exists?(test_data).should be_true
      File.exists?(test_label).should be_true
    ensure
      File.exists?(test_data) && File.delete(test_data)
      File.exists?(test_label) && File.delete(test_label)
    end
  end
end
