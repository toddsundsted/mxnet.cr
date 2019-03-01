require "../../spec_helper"

describe "MXNet::Context" do
  describe ".cpu" do
    it "is a shortcut for .new(:cpu, ...)" do
      MXNet::Context.cpu.should eq(MXNet::Context.new(:cpu, 0))
    end
  end

  describe ".gpu" do
    it "is a shortcut for .new(:gpu, ...)" do
      MXNet::Context.gpu(1).should eq(MXNet::Context.new(:gpu, 1))
    end
  end

  describe ".current" do
    it "returns the default context" do
      MXNet::Context.current.should eq(MXNet::Context.new(:cpu, 0))
    end
  end
end
