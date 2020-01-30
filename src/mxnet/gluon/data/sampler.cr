require "../data"

module MXNet
  module Gluon
    module Data
      # Base class for samplers.
      #
      # All samplers should subclass `Sampler` and define `#size` and
      # `#each` methods.
      #
      abstract class Sampler(T)
        include Enumerable(T)

        abstract def size

        abstract def each(&block : T -> _)

        abstract def each
      end

      # Samples elements from [0, size) sequentially.
      #
      class SequentialSampler < Sampler(Int32)
        # Creates a new instance.
        #
        # ### Parameters
        # * *size* (`Int32`)
        #   Size of the sequence.
        #
        def initialize(@size : Int32)
        end

        def size
          @size
        end

        def each
          @size.times do |i|
            yield i
          end
        end

        def each
          @size.times
        end
      end

      # Samples elements from [0, size) randomly without replacement.
      #
      class RandomSampler < Sampler(Int32)
        # Creates a new instance.
        #
        # ### Parameters
        # * *size* (`Int32`)
        #   Size of the sequence.
        #
        def initialize(@size : Int32)
        end

        def size
          @size
        end

        def each
          @size.times.to_a.shuffle.each do |i|
            yield i
          end
        end

        def each
          @size.times.to_a.shuffle.each
        end
      end

      # Wraps another `Sampler` and returns mini-batches of samples.
      #
      # ```
      # sampler = MXNet::Gluon::Data::BatchSampler.new(
      #   MXNet::Gluon::Data::SequentialSampler.new(10), 3, :keep
      # )
      # sampler.to_a # => [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
      # ```
      #
      class BatchSampler(T) < Sampler(Array(T))
        include Iterator(Array(T))

        @enum : Enumerable(T)

        # Creates a new instance.
        #
        # ### Parameters
        # * *sampler* (`Sampler`)
        #   The source sampler.
        # * *batch_size* (`Int32`)
        #   Size of mini-batches.
        # * *last_batch* (`:keep`, `:discard`, `:rollover`)
        #   Specifies how the last batch is handled if `batch_size`
        #   does not evenly divide sampler sequence size. If `:keep`,
        #   the last batch will be returned directly, but will contain
        #   fewer elements than `batch_size` requires. If `:discard`,
        #   the last batch will be discarded.  If `:rollover`, the
        #   remaining elements will be rolled over to the next
        #   iteration.
        #
        def initialize(@sampler : Sampler(T), @batch_size : Int32, @last_batch = :keep)
          unless [:keep, :discard, :rollover].includes?(last_batch)
            raise ArgumentError.new("last_batch must be either :keep, :discard, or :rollover")
          end
          @enum = @sampler.each
          @batch = [] of T
        end

        # Returns the number of elements in the sequence.
        #
        def size
          case @last_batch
          when :discard
            (@sampler.size / @batch_size).to_i
          when :keep
            ((@sampler.size + @batch_size - 1) / @batch_size).to_i
          when :rollover
            ((@sampler.size + @batch.size) / @batch_size).to_i
          else
            raise NotImplementedError.new("unsupported: #{@last_batch}")
          end
        end

        # Rewinds the sequence.
        #
        # If the source sampler is a `BatchSampler`, this rewinds it
        # as well.
        #
        def rewind
          if (sampler = @sampler).responds_to?(:rewind)
            sampler.rewind
          end
          @enum = @sampler.each
          self
        end

        # Returns the next element in the sequence, or
        # `Iterator::Stop::INSTANCE` if there are no more elements.
        #
        def next
          batch = [] of T
          unless @batch.empty?
            batch, @batch = @batch, batch
          end
          @enum.each do |i|
            batch << i
            if batch.size == @batch_size
              return batch
            end
          end
          unless batch.empty?
            case @last_batch
            when :discard
            when :keep
              return batch
            when :rollover
              @batch = batch
            end
          end
          stop
        end
      end
    end
  end
end
