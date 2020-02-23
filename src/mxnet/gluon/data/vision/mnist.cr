require "../../data"
require "gzip"

module MXNet
  module Gluon
    module Data
      module Vision
        class MNIST(T) < MXNet::Gluon::Data::DownloadedDataset(MXNet::NDArray, Int32, T)
          TRAIN_DATA = {"train-images-idx3-ubyte.gz", "6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d"}
          TRAIN_LABEL = {"train-labels-idx1-ubyte.gz", "2a80914081dc54586dbdf242f9805a6b8d2a15fc"}
          TEST_DATA = {"t10k-images-idx3-ubyte.gz", "c3a25af1f52dad7f726cce8cacb138654b760d48"}
          TEST_LABEL = {"t10k-labels-idx1-ubyte.gz", "763e7fa3757d93b0cdec073cef058b2004252c17"}

          # Creates a new instance.
          #
          # ### Parameters
          # * *root* (`String`, optional)
          #   Directory in which to cache downloaded files.
          #   Automatically created if it does not already exist.
          # * *train* (`Bool`, optional)
          #   Whether to load the training or testing data.
          #
          def self.new(root = File.join("~/", ".mxnet", "datasets", "mnist"), train = true)
            MNIST(Tuple(MXNet::NDArray, Int32)).new(transform: nil, root: root, train: train)
          end

          # Creates a new instance.
          #
          # Transforms each sample with the supplied transformer.
          #
          # ### Parameters
          # * *transform* (`Proc`, required)
          #   Transformation to apply to each sample.
          # * *root* (`String`, optional)
          #   Directory in which to cache downloaded files.
          #   Automatically created if it does not already exist.
          # * *train* (`Bool`, optional)
          #   Whether to load the training or testing data.
          #
          def initialize(transform : Proc(MXNet::NDArray, Int32, T)?, root = File.join("~/", ".mxnet", "datasets", "mnist"), @train = true)
            super(root: root, transform: transform)
          end

          def get_data
            if @train
              data, label = TRAIN_DATA, TRAIN_LABEL
            else
              data, label = TEST_DATA, TEST_LABEL
            end

            namespace = "gluon/dataset/mnist"
            data_file = MXNet::Gluon::Utils.download(
              MXNet::Gluon::Utils.get_repo_file_url(namespace, data[0]),
              path: self.root,
              sha1_hash: data[1]
            )
            label_file = MXNet::Gluon::Utils.download(
              MXNet::Gluon::Utils.get_repo_file_url(namespace, label[0]),
              path: self.root,
              sha1_hash: label[1]
            )

            data_out = [] of MXNet::NDArray
            File.open(data_file) do |io|
              io = Gzip::Reader.new(io)
              magic = UInt32.from_io(io, IO::ByteFormat::BigEndian)
              count = UInt32.from_io(io, IO::ByteFormat::BigEndian)
              nrows = UInt32.from_io(io, IO::ByteFormat::BigEndian)
              ncols = UInt32.from_io(io, IO::ByteFormat::BigEndian)

              data = IO::Memory.new
              IO.copy(io, data)
              slice = data.to_slice

              data_out = Array(MXNet::NDArray).new(count) do |i|
                MXNet::NDArray.array(slice[i * nrows * ncols, nrows * ncols], dtype: :float32).reshape(shape: [28, 28])
              end
            end

            label_out = [] of Int32
            File.open(label_file) do |io|
              io = Gzip::Reader.new(io)
              magic = UInt32.from_io(io, IO::ByteFormat::BigEndian)
              count = UInt32.from_io(io, IO::ByteFormat::BigEndian)

              label = IO::Memory.new
              IO.copy(io, label)
              slice = label.to_slice

              label_out = Array(Int32).new(count) do |i|
                slice[i].to_i32
              end
            end

            {data_out, label_out}
          end
        end
      end
    end
  end
end
