require "../../../spec_helper"

describe MXNet::Gluon::Data::DataLoader do
  shuffle = false
  batch_size = 3
  last_batch = :keep

  context "basic operation" do
    dataset = (0..99).to_a
    loader = MXNet::Gluon::Data::DataLoader(Int32, Array(Int32)).new(dataset, shuffle: shuffle, batch_size: batch_size, last_batch: last_batch)

    describe "#size" do
      it "is 34 (mini-batches)" do
        loader.size.should eq(34)
      end
    end

    describe "#each" do
      it "results in an array of slices" do
        loader.each.to_a.should eq((0..99).each_slice(3).to_a)
      end
    end
  end

  context "with `shuffle: true`" do
    dataset = (0..99).to_a
    loader = MXNet::Gluon::Data::DataLoader(Int32, Array(Int32)).new(dataset, shuffle: true, batch_size: batch_size, last_batch: last_batch)

    describe "#size" do
      it "is 34 (mini-batches)" do
        loader.size.should eq(34)
      end
    end

    describe "#each" do
      it "results in an array out of sequence" do
        loader.each.to_a.should_not eq((0..99).each_slice(3).to_a)
      end
    end
  end

  context "with `batch_size: 10`" do
    dataset = (0..99).to_a
    loader = MXNet::Gluon::Data::DataLoader(Int32, Array(Int32)).new(dataset, shuffle: shuffle, batch_size: 10, last_batch: last_batch)

    describe "#size" do
      it "is 10 (mini-batches)" do
        loader.size.should eq(10)
      end
    end

    describe "#each" do
      it "results in an array of slices" do
        loader.each.to_a.should eq((0..99).each_slice(10).to_a)
      end
    end
  end

  context "with `last_batch: :discard`" do
    dataset = (0..99).to_a
    loader = MXNet::Gluon::Data::DataLoader(Int32, Array(Int32)).new(dataset, shuffle: shuffle, batch_size: batch_size, last_batch: :discard)

    describe "#size" do
      it "is 33 (mini-batches)" do
        loader.size.should eq(33)
      end
    end

    describe "#each" do
      it "results in an array of slices" do
        loader.each.to_a.should eq((0..98).each_slice(3).to_a)
      end
    end
  end

  context "with `last_batch: :rollover`" do
    dataset = (0..99).to_a
    loader = MXNet::Gluon::Data::DataLoader(Int32, Array(Int32)).new(dataset, shuffle: shuffle, batch_size: batch_size, last_batch: :rollover)
    loader.to_a ; loader.rewind

    describe "#size" do
      it "is 33 (mini-batches)" do
        loader.size.should eq(33)
      end
    end

    describe "#each" do
      it "results in an array of slices" do
        loader.each.to_a.should eq((0..99).to_a.rotate(-1).each_slice(3).to_a[0..-2])
      end
    end
  end

  describe "Batchification" do
    context "using built-in batchify function" do
      it "converts batches into NDArrays" do
        dataset = (0..99).to_a
        loader = MXNet::Gluon::Data::DataLoader(Int32, MXNet::NDArray).new(dataset, shuffle: false, batch_size: 10)
        loader.next.should eq(MXNet::NDArray.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
      end

      it "converts the elements of tuples" do
        dataset = [{1, 2}, {3, 4}, {5, 6}, {7, 8}]
        loader = MXNet::Gluon::Data::DataLoader(Tuple(Int32, Int32), Tuple(Array(Int32), Array(Int32))).new(dataset, shuffle: false, batch_size: 2)
        loader.next.should eq({[1, 3], [2, 4]})
      end

      it "converts the elements of tuples" do
        dataset = [{[1, 1], 2}, {[3, 3], 4}, {[5, 5], 6}, {[7, 7], 8}]
        loader = MXNet::Gluon::Data::DataLoader(Tuple(Array(Int32), Int32), Tuple(MXNet::NDArray, MXNet::NDArray)).new(dataset, shuffle: false, batch_size: 2)
        loader.next.should eq({MXNet::NDArray.array([[1, 1], [3, 3]]), MXNet::NDArray.array([2, 4])})
      end

      it "takes NDArrays as source" do
        dataset = [MXNet::NDArray.array([1, 1]), MXNet::NDArray.array([2, 2]), MXNet::NDArray.array([3, 3]), MXNet::NDArray.array([4, 4])]
        loader = MXNet::Gluon::Data::DataLoader(MXNet::NDArray, MXNet::NDArray).new(dataset, shuffle: false, batch_size: 3)
        loader.next.should eq(MXNet::NDArray.array([[1, 1], [2, 2], [3, 3]]))
      end
    end

    context "with explicit batchify function" do
      it "converts batches while batchifying" do
        dataset = (0..99).to_a
        loader = MXNet::Gluon::Data::DataLoader(Int32, Array(String)).new(
          dataset,
          batchify_fn: ->(i : Array(Int32)) { i.map(&.to_s) },
          shuffle: false,
          batch_size: 10
        )
        loader.next.should eq(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
      end

      it "converts the elements of tuples" do
        dataset = [{1, 2}, {3, 4}, {5, 6}, {7, 8}]
        loader = MXNet::Gluon::Data::DataLoader(Tuple(Int32, Int32), Tuple(Array(Int32), Array(Int32), Array(Int32))).new(
          dataset,
          batchify_fn: ->(i : Array(Tuple(Int32, Int32))) { {i.map(&.[0]), i.map(&.[1]), i.map { |j, k| j + k } } },
          shuffle: false,
          batch_size: 2)
        loader.next.should eq({[1, 3], [2, 4], [3, 7]})
      end
    end
  end
end
