require "../data"

module MXNet
  module Gluon
    module Data

      # Abstract dataset.
      #
      # All datasets should implement this interface.
      #
      # `Dataset` includes `Indexable`, so subclasses must define
      # `#size`, which returns the total number elements, and
      # `#unsafe_fetch`, which returns the indexed element.
      #
      abstract class Dataset(T)
        include Indexable(T)

        abstract def size

        abstract def unsafe_fetch(idx)

        # Returns a new dataset with each sample transformed by the
        # supplied transformer block.
        #
        # ### Parameters
        # * *lazy* (`Bool`, default = `true`)
        #   If `false`, transforms all samples at once. Otherwise,
        #   transforms each sample on demand. Note that if the
        #   transformer block is stochastic, you must set `lazy` to
        #   `true` or you will get the same result on all epochs.
        #
        def transform(lazy = true, &proc : T -> U) forall U
          trans = LazyTransformDataset(T, U).new(self, &proc)
          lazy ? trans : SimpleDataset(U).new(trans.to_a)
        end
      end

      # Simple `Dataset` wrapper for arrays and other classes that
      # implement `Indexable`.
      #
      class SimpleDataset(T) < Dataset(T)
        # Creates a new instance.
        #
        # ### Parameters
        # * *dataset* (`Indexable`)
        #
        def initialize(@dataset : Indexable(T))
        end

        def size
          @dataset.size
        end

        def unsafe_fetch(idx)
          @dataset[idx]
        end
      end

      # Lazily transformed `Dataset` wrapper for arrays and other
      # classes that implement `Indexable`.
      #
      class LazyTransformDataset(T, U) < Dataset(U)
        # Creates a new instance.
        #
        # ### Parameters
        # * *dataset* (`Indexable`)
        #
        def initialize(@dataset : Indexable(T), &@proc : T -> U)
        end

        def size
          @dataset.size
        end

        def unsafe_fetch(idx)
          @proc.call(@dataset[idx])
        end
      end

      # Abstract base class for MNIST, CIFAR10, etc.
      #
      # Subclasses must define `#get_data`, which returns arrays of
      # data and labels for the dataset.
      #
      abstract class DownloadedDataset(T, U, V) < Dataset(V)
        @data : Array(T)?
        @label : Array(U)?

        # Creates a new instance.
        #
        # ### Parameters
        # * *root* (`String`)
        #   Directory in which to cache downloaded files.
        #   Automatically created if it does not already exist.
        # * *transform* (`Proc`, optional)
        #   Optional transformation to apply to each returned sample.
        #
        def initialize(root, transform : Proc(T, U, V)? = nil)
          @root = File.expand_path(root)
          Dir.mkdir_p(@root) unless Dir.exists?(@root)
          @transform = transform
          @data, @label = get_data
        end

        getter :root

        def size
          @label.try(&.size) || 0
        end

        def unsafe_fetch(idx)
          data, label = @data.not_nil!, @label.not_nil!
          if transform = @transform
            transform.call(data[idx], label[idx])
          else
            {data[idx], label[idx]}
          end
        end

        private abstract def get_data
      end
    end
  end
end
