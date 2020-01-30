require "../../../spec_helper"

describe MXNet::Gluon::Data::SequentialSampler do
  sampler = MXNet::Gluon::Data::SequentialSampler.new(100)

  describe "#size" do
    it "is 100" do
      sampler.size.should eq(100)
    end
  end

  describe "#each" do
    it "results in an array from 0 to 99 in sequence" do
      sampler.each.to_a.should eq((0..99).to_a)
    end
  end
end

describe MXNet::Gluon::Data::RandomSampler do
  sampler = MXNet::Gluon::Data::RandomSampler.new(100)

  describe "#size" do
    it "is 100" do
      sampler.size.should eq(100)
    end
  end

  describe "#each" do
    it "results in an array from 0 to 99 out of sequence" do
      sampler.each.to_a.should_not eq((0..99).to_a)
    end
  end
end

describe MXNet::Gluon::Data::BatchSampler do
  context "with `last_batch = :keep`" do
    sampler = MXNet::Gluon::Data::BatchSampler.new(
      MXNet::Gluon::Data::SequentialSampler.new(10), 3, :keep
    )

    describe "#size" do
      it "is 4 (mini-batches)" do
        sampler.size.should eq(4)
      end
    end

    describe "#each" do
      it "results in a final partial batch" do
        sampler.each.to_a.should eq([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
      end
    end
  end

  context "with `last_batch = :discard`" do
    sampler = MXNet::Gluon::Data::BatchSampler.new(
      MXNet::Gluon::Data::SequentialSampler.new(10), 3, :discard
    )

    describe "#size" do
      it "is 3 (mini-batches)" do
        sampler.size.should eq(3)
      end
    end

    describe "#each" do
      it "results in an omitted final batch" do
        sampler.each.to_a.should eq([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
      end
    end
  end

  context "with `last_batch = :rollover`" do
    sampler = MXNet::Gluon::Data::BatchSampler.new(
      MXNet::Gluon::Data::SequentialSampler.new(10), 3, :rollover
    )
    2.times { sampler.to_a ; sampler.rewind }

    describe "#size" do
      it "is 3 (mini-batches)" do
        sampler.size.should eq(4)
      end
    end

    describe "#each" do
      it "results in a rolled-over final batch" do
        sampler.each.to_a.should eq([[8, 9, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
      end
    end
  end
end

describe "Chaining Test" do
  sampler = MXNet::Gluon::Data::BatchSampler.new(
    MXNet::Gluon::Data::BatchSampler.new(
      MXNet::Gluon::Data::SequentialSampler.new(10), 3, :discard
    ), 2, :keep
  )

  describe "#size" do
    it "is 3" do
      sampler.size.should eq(2)
    end
  end

  describe "#each" do
    it "results in an array of arrays or arrays" do
      sampler.each.to_a.should eq([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8]]])
    end
  end
end
