require "../../spec_helper"

describe "MXNet::Initializer" do
  describe ".create" do
    it "accepts a class" do
      init = MXNet::Initializer.create(MXNet::Initializer::Zero)
      init.should be_a(MXNet::Initializer::Zero)
    end
    it "accepts an instance" do
      init = MXNet::Initializer.create(MXNet::Initializer::Zero.new)
      init.should be_a(MXNet::Initializer::Zero)
    end
    it "accepts a string" do
      init = MXNet::Initializer.create("zeros")
      init.should be_a(MXNet::Initializer::Zero)
    end
    it "accepts a symbol" do
      init = MXNet::Initializer.create(:zeros)
      init.should be_a(MXNet::Initializer::Zero)
    end
  end
end
