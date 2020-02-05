require "../mxnet"

module MXNet
  # The `Gluon` library is a high-level interface for MXNet designed
  # to be easy to use, while keeping most of the flexibility of a low
  # level API. `Gluon` supports both imperative and symbolic
  # programming, making it easy to train complex models imperatively
  # and then to deploy as a symbolic graph.
  #
  # ```
  # net = MXNet::Gluon::NN::HybridSequential.new.tap do |net|
  #   # When instantiated, `HybridSequential` stores a chain of
  #   # neural network layers. Once presented with data, it executes
  #   # each layer in turn, using the output of one layer as the input
  #   # for the next. Calling `#hybridize` caches the neural network
  #   # for high performance.
  #   net.with_name_scope do
  #     net.add(
  #       MXNet::Gluon::NN::Dense.new(64, activation: :relu), # 1st layer (64 nodes)
  #       MXNet::Gluon::NN::Dense.new(64, activation: :relu), # 2nd hidden layer
  #       MXNet::Gluon::NN::Dense.new(10)
  #     )
  #   end
  #   net.init
  #   net.hybridize
  # end
  # ```
  #
  module Gluon
  end
end

require "./gluon/utils"
require "./gluon/data"
require "./gluon/parameter"
require "./gluon/block"
require "./gluon/trainer"
require "./gluon/loss"
require "./gluon/nn"
