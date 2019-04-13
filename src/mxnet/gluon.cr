require "../mxnet"

module MXNet
  # The `Gluon` library is a high-level interface for MXNet designed
  # to be easy to use, while keeping most of the flexibility of a low
  # level API. `Gluon` supports both imperative and symbolic
  # programming, making it easy to train complex models imperatively
  # and then to deploy as a symbolic graph.
  #
  module Gluon
  end
end

require "./gluon/parameter"
require "./gluon/trainer"
