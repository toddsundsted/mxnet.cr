module MXNet
  # The base class of an initializer.
  #
  # Custom initializers can be created by subclassing `Initializer`
  # and implementing the required function `#init_array`. By default,
  # the created initializer will be registered under its simplified
  # class name (`class.name.split("::").last.downcase`) but it may
  # also be registered under another name by calling `#register`.
  #
  #     class CustomInit < MXNet::Initializer
  #       register :myinit
  #       def init_array(array)
  #         array[0..-1] = 1.0
  #       end
  #     end
  #
  abstract class Initializer
    @@registry = Hash(String, Initializer.class).new

    protected def self.register_initializer(name, initializer)
      @@registry[name.to_s] = initializer
    end

    private macro inherited
      MXNet::Initializer.register_initializer({{@type.name.split("::").last.downcase}}, {{@type}})
    end

    protected def self.register(name)
      MXNet::Initializer.register_initializer(name, self)
    end

    def self.create(initializer)
      case initializer
      when ::String, ::Symbol
        @@registry[initializer.to_s].new
      when .responds_to?(:new)
        initializer.new
      else
        initializer
      end
    end

    # Override to initialize array.
    #
    # ### Parameters
    # * *array* (`NDArray`)
    #   Array to initialize.
    #
    abstract def init_array(array : NDArray)

    # Initializes array to zero.
    #
    class Zero < Initializer
      register :zeros
      def init_array(array)
        array[0..-1] = 0.0
      end
    end

    # Initializes array to one.
    #
    class One < Initializer
      register :ones
      def init_array(array)
        array[0..-1] = 1.0
      end
    end

    # Initializes the weights to a given constant value.
    #
    class Constant < Initializer
      # Creates a new instance.
      #
      # The value passed in can be a scalar or a `NDarray` that matches
      # the shape of the parameter to be set.
      #
      # ### Parameters
      # * *value* (`Float` | `NDArray`, required)
      #   The value to set.
      #
      def initialize(@value : Float32 | Float64 | NDArray = 0.0)
      end

      def init_array(array)
        array[0..-1] = @value
      end
    end

    # Initializes array with random values uniformly sampled from a
    # given range.
    #
    class Uniform < Initializer
      # Creates a new instance.
      #
      # ### Parameters
      # * *scale* (`Float`, optional)
      #   The bound on the range of the generated random values.
      #   Values are generated from the range `[-scale, scale]`.
      #   Default *scale* is 0.07.
      #
      def initialize(@scale = 0.07)
      end

      def init_array(array)
        MXNet::NDArray.random_uniform(-@scale, @scale, out: array)
      end
    end

    # Initializes array with random values sampled from a normal
    # distribution with a mean of zero and standard deviation of
    # *sigma*.
    #
    class Normal < Initializer
      # Creates a new instance.
      #
      # ### Parameters
      # * *sigma* (`Float`, optional)
      #   Standard deviation of the normal distribution. The default
      #   standard deviation is 0.01.
      #
      def initialize(@sigma = 0.01)
      end

      def init_array(array)
        MXNet::NDArray.random_normal(0, @sigma, out: array)
      end
    end
  end
end
