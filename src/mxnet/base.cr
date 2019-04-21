require "./operations"

module MXNet
  # Base class for `NDArray` and `Symbol`.
  #
  class Base
    private macro inherited
      include MXNet::Operations
      extend MXNet::Operations
      extend MXNet::Util
    end
  end
end
