require "./operations"

module MXNet
  # Util class.
  #
  module Util
    # Recursively pretty-prints *arg* to *io*.
    #
    # Suitable for formatting MXNet keyword arguments.
    #
    # ### Parameters
    # * *arg* (any)
    #   Value to pretty-print.
    # * *io* (`IO`)
    #   `IO` instance.
    #
    private def output(arg, io)
      if arg.responds_to?(:join)
        io << '['
        arg.join(io, ",") { |a| output(a, io) }
        io << ']'
      else
        io << arg
      end
    end

    # Recursively pretty-prints *arg*.
    #
    # Suitable for formatting MXNet keyword arguments.
    #
    # ### Parameters
    # * *arg* (any)
    #   Value to pretty-print.
    #
    def output(arg)
      String.build do |io|
        output(arg, io)
      end
    end
  end
end
