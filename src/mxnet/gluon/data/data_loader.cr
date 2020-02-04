require "../data"

module MXNet
  module Gluon
    module Data
      # Loads data from a `Dataset` and returns batches of data.
      #
      # `DataLoader` is parameterized by two types: `Element`, which
      # is the type of the elements in the supplied dataset, and `Batch`,
      # which is the type of the returned batches.
      #
      # Batched samples of the source dataset are turned into a batch
      # with the batchify function `batchify_fn`. The default batchify
      # function operates on a dataset with elements that are either
      # non-aggregate elements or tuples of elements. If the type of
      # `Batch` is `MXNet::NDArray` or is a tuple containing this
      # type, the default batchify function will attempt to transform
      # the batched samples into instances of `MXNet::NDArray`.
      # The default batchify function is:
      #
      # ```
      # private class Batchify(E, B)
      #   def self.batchify(data : Array(E)) : B
      #     {% if B < Tuple && E < Tuple %}
      #       {% raise "the default batchify function requires types have the same size: #{B}.size != #{E}.size" unless B.size == E.size %}
      #       {
      #         {% for i in (0...B.size) %}
      #           Batchify({{E.type_vars[i]}}, {{B.type_vars[i]}}).batchify(data.map(&.[{{i}}])),
      #         {% end %}
      #       }
      #     {% elsif E == MXNet::NDArray %}
      #       MXNet::NDArray::Ops._stack(data, num_args: data.size)
      #     {% elsif B == MXNet::NDArray && (E < Number || E < Array) %}
      #       MXNet::NDArray.array(data)
      #     {% elsif B == Array(E) %}
      #       data
      #     {% else %}
      #       {% raise "the default batchify function can't transform a batched sample of #{E} into #{B}" %}
      #     {% end %}
      #   end
      # end
      # ```
      #
      class DataLoader(Element, Batch)
        include Iterator(Batch)

        @sampler : MXNet::Gluon::Data::BatchSampler(Int32)

        # Creates a new instance.
        #
        # ### Parameters
        # * *dataset* (`Indexable`)
        #   Source dataset. Note that any `Indexable` can be directly
        #   used as a `Dataset`.
        # * *shuffle* (`Bool`)
        #   Whether or not to shuffle the samples.
        # * *batch_size* (`Int32`)
        #   Size of batch.
        # * *last_batch* (`:keep`, `:discard`, `:rollover`)
        #   Specifies how the last batch is handled if `batch_size`
        #   does not evenly divide sampler sequence size. If `:keep`,
        #   the last batch will be returned directly, but will contain
        #   fewer elements than `batch_size` requires. If `:discard`,
        #   the last batch will be discarded.  If `:rollover`, the
        #   remaining elements will be rolled over to the next
        #   iteration.
        # * *batchify_fn* (`Proc`, default = `default_batchify_fn`)
        #   Function that specifies how to merge samples into a batch.
        #
        def initialize(@dataset : Indexable(Element),
                       *, shuffle, batch_size, last_batch = :keep,
                       @batchify_fn : Array(Element) -> Batch = ->default_batchify_fn(Array(Element)))
          sampler = shuffle ?
                       MXNet::Gluon::Data::RandomSampler.new(@dataset.size) :
                       MXNet::Gluon::Data::SequentialSampler.new(@dataset.size)
          @sampler = MXNet::Gluon::Data::BatchSampler.new(sampler, batch_size, last_batch)
        end

        def size
          @sampler.size
        end

        def next
          if (value = @sampler.next).responds_to?(:map)
            @batchify_fn.call(value.map { |i| @dataset[i] })
          else
            stop
          end
        end

        def rewind
          if (sampler = @sampler).responds_to?(:rewind)
            sampler.rewind
          end
          self
        end

        # The default batchify function.
        #
        # Implementing the default batchify function inside a class,
        # like this, allows us to specialize the function on the
        # return value, which is not otherwise possible.
        #
        private class Batchify(E, B)
          def self.batchify(data : Array(E)) : B
            {% if B < Tuple && E < Tuple %}
              {% raise "the default batchify function requires types have the same size: #{B}.size != #{E}.size" unless B.size == E.size %}
              {
                {% for i in (0...B.size) %}
                  Batchify({{E.type_vars[i]}}, {{B.type_vars[i]}}).batchify(data.map(&.[{{i}}])),
                {% end %}
              }
            {% elsif E == MXNet::NDArray %}
              MXNet::NDArray::Ops._stack(data, num_args: data.size)
            {% elsif B == MXNet::NDArray && (E < Number || E < Array) %}
              MXNet::NDArray.array(data)
            {% elsif B == Array(E) %}
              data
            {% else %}
              {% raise "the default batchify function can't transform a batched sample of #{E} into #{B}" %}
            {% end %}
          end
        end

        private def default_batchify_fn(data : Array(Element)) : Batch
          Batchify(Element, Batch).batchify(data)
        end
      end
    end
  end
end
