require "../../../spec_helper"

describe MXNet::Name::Manager do
  describe "#get" do
    it "returns the name if specified" do
      MXNet::Name::Manager.current.get("foobar", "test").should eq("foobar")
    end

    it "returns a unique name generated from the hint" do
      name1 = MXNet::Name::Manager.current.get(nil, "test")
      name2 = MXNet::Name::Manager.current.get(nil, "test")
      name1.should match(/test/)
      name2.should match(/test/)
      name1.should_not eq(name2)
    end
  end
end
