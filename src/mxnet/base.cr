require "./operations"

module MXNet
  # Base class for `NDArray` and `Symbol`.
  #
  class Base
    # Recursively pretty-print *arg* to *io*.
    #
    # Suitable for formatting MXNet keyword arguments.
    #
    # ### Parameters
    # * *arg* (any)
    #   Value to pretty-print.
    # * *io* (`IO`)
    #   `IO` instance.
    #
    private def self.output(arg, io)
      case arg
      when .responds_to?(:join)
        io << '['
        arg.join(",", io) { |a| output(a, io) }
        io << ']'
      when Nil
        io << "None"
      else
        io << arg
      end
    end

    # Recursively pretty-print *arg*.
    #
    # Suitable for formatting MXNet keyword arguments.
    #
    # ### Parameters
    # * *arg* (any)
    #   Value to pretty-print.
    #
    protected def self.output(arg)
      String.build do |io|
        output(arg, io)
      end
    end
  end
end
