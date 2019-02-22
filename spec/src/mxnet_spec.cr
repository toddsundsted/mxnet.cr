require "../spec_helper"

describe "MXNet" do
  describe "#version" do
    it "returns the version of the MXNet library" do
      MXNet.version.should be_a(Int32)
    end
  end
end
