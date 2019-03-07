module MXNet
  # Random number interface of MXNet.
  class Random
    # Seeds the random number generators in MXNet.
    #
    # This affects the behavior of modules in MXNet that use random
    # number generators, like the dropout operator and the `MXNet::NDArray`
    # random sampling operators.
    #
    # ### Parameters
    # * *seed_state* (`Int32`)
    #   The random number seed.
    # * *ctx* (`Context` or `:all` for all devices, default `:all`)
    #   The device context of the generator. The default is `:all`
    #   which means seeding random number generators of all devices.
    #
    # ### Notes
    # Random number generators in MXNet are device specific.
    # `MXNet::Random.seed(seed_state)` sets the state of each
    # generator using `seed_state` and the device id. Therefore,
    # random numbers generated from different devices can be different
    # even if they are seeded using the same seed.
    #
    def self.seed(seed_state : Int32, ctx : Context | ::Symbol = :all)
      if ctx == :all
        MXNet::Internal.libcall(MXRandomSeed, seed_state)
      elsif ctx.is_a?(Context)
        MXNet::Internal.libcall(MXRandomSeedContext, seed_state, ctx.device.first, ctx.device.last)
      else
        raise ArgumentError.new("invalid context: #{ctx}")
      end
    end
  end
end
