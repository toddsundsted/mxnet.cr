require "../../../spec_helper"

describe MXNet::Gluon::Data::SimpleDataset do
  context "basic operation" do
    dataset = MXNet::Gluon::Data::SimpleDataset.new([1, 2, 3, 4])

    describe "#size" do
      it "returns the size" do
        dataset.size.should eq(4)
      end
    end

    describe "#[]" do
      it "returns the value at the specified index" do
        dataset[0].should eq(1)
        dataset[1].should eq(2)
        dataset[-1].should eq(4)
      end
    end
  end

  describe "#transform" do
    dataset = MXNet::Gluon::Data::SimpleDataset.new([1, 2, 3, 4])

    context "by default" do
      it "transforms lazily" do
        res = dataset.transform { |x| x * 10 }
        res.should be_a(MXNet::Gluon::Data::LazyTransformDataset(Int32, Int32))
        res.size.should eq(4)
        res[0].should eq(10)
        res[1].should eq(20)
        res[-1].should eq(40)
      end
    end

    context "with lazy: false" do
      it "transforms immediately" do
        res = dataset.transform(lazy: false) { |x| x * 10 }
        res.should be_a(MXNet::Gluon::Data::SimpleDataset(Int32))
        res.size.should eq(4)
        res[0].should eq(10)
        res[1].should eq(20)
        res[-1].should eq(40)
      end
    end
  end
end

describe MXNet::Gluon::Data::LazyTransformDataset do
  dataset = MXNet::Gluon::Data::LazyTransformDataset.new([1, 2, 3, 4]) { |x| x * 10 }

  describe "#size" do
    it "returns the size" do
      dataset.size.should eq(4)
    end
  end

  describe "#[]" do
    it "returns the transformed value at the specified index" do
      dataset[0].should eq(10)
      dataset[1].should eq(20)
      dataset[-1].should eq(40)
    end
  end
end

class TestDataset1 < MXNet::Gluon::Data::DownloadedDataset(Int32, String, Tuple(Int32, String))
  def initialize(*, root = File.join(Dir.tempdir, "test"), transform = nil)
    super(root: root, transform: transform)
  end

  private def get_data
    {[1, 2, 3, 4], ["one", "two", "three", "four"]}
  end
end

class TestDataset2 < MXNet::Gluon::Data::DownloadedDataset(Int32, Int32, Tuple(Int32, String))
  def initialize(*, root = File.join(Dir.tempdir, "test"), transform = nil)
    super(root: root, transform: transform)
  end

  private def get_data
    {[1, 2, 3, 4], [1, 2, 3, 4]}
  end
end

describe MXNet::Gluon::Data::DownloadedDataset do
  context "without transformation" do
    dataset = TestDataset1.new

    describe "#size" do
      it "returns the size" do
        dataset.size.should eq(4)
      end
    end

    describe "#[]" do
      it "returns the value at the specified index" do
        dataset[0].should eq({1, "one"})
        dataset[1].should eq({2, "two"})
        dataset[-1].should eq({4, "four"})
      end
    end
  end

  context "with transformation" do
    dataset = TestDataset2.new(transform: ->(i : Int32, j : Int32) { {i * 10, j.humanize} })

    describe "#size" do
      it "returns the size" do
        dataset.size.should eq(4)
      end
    end

    describe "#[]" do
      it "returns the value at the specified index" do
        dataset[0].should eq({10, "1.0"})
        dataset[1].should eq({20, "2.0"})
        dataset[-1].should eq({40, "4.0"})
      end
    end
  end
end

describe "Chaining Test" do
  a = MXNet::Gluon::Data::SimpleDataset.new([1, 2, 3, 4])
  b = a.transform { |i| {i, i} }
  c = b.transform { |i| i.map(&.to_s) }
  d = c.transform { |i| i.first.to_f }
  it "returns transformed values" do
    a.to_a.should eq([1, 2, 3, 4])
    b.to_a.should eq([{1, 1}, {2, 2}, {3, 3}, {4, 4}])
    c.to_a.should eq([{"1", "1"}, {"2", "2"}, {"3", "3"}, {"4", "4"}])
    d.to_a.should eq([1.0, 2.0, 3.0, 4.0])
  end
end
