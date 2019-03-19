module MXNet
  # MXNet context.
  #
  class Context
    private DEVICE_TYPE_SYM_TO_INT = {
      :cpu => 1,
      :gpu => 2,
      :cpu_pinned => 3,
      :cpu_shared => 5
    }
    private DEVICE_TYPE_INT_TO_SYM = {
      1 => :cpu,
      2 => :gpu,
      3 => :cpu_pinned,
      5 => :cpu_shared
    }

    @@default = Context.cpu

    @device_type : Int32
    @device_id : Int32

    # Constructs a context.
    #
    # MXNet can run operations on a CPU and different GPUs. A context
    # describes the device on which computation should be carried out.
    #
    # Use `.cpu` and `.gpu` as shortcuts.
    #
    # ### Parameters
    # * *device_type* (`:cpu`, `:gpu` or `Int32`)
    #   Symbol representing the device type, or the device type.
    # * *device_id*   (`Int32`, default = 0)
    #   Device id of the device (for GPUs).
    #
    # ###  See also
    #
    # [How to run MXNet on multiple CPU/GPUs](http://mxnet.io/faq/multi_devices.html)
    #
    def initialize(device_type : ::Symbol | Int32, device_id : Int32 = 0)
      case device_type
      when ::Symbol
        @device_type = DEVICE_TYPE_SYM_TO_INT[device_type]
        @device_id = device_id
      else
        @device_type = device_type
        @device_id = device_id
      end
    end

    # Returns a CPU context.
    #
    # This function is a shortcut for `MXNet::Context.new(:cpu, device_id)`.
    # For most operations, when no context is specified, the default
    # context is `MXNet::Context.cpu`.
    #
    # ### Parameters
    # * *device_id* (`Int32`, default = 0)
    #   Device id of the device. Not required for the CPU
    #   context. Included to make the interface compatible with GPU
    #   contexts.
    #
    def self.cpu(device_id : Int32 = 0)
      new(:cpu, device_id)
    end

    # Returns a GPU context.
    #
    # This function is a shortcut for `MXNet::Context.new(:gpu, device_id)`.
    # The K GPUs on a node are typically numbered 0, ..., K-1.
    #
    # ### Parameters
    # * *device_id* (`Int32`, default = 0)
    #   Device id of the device. Required for the GPU contexts.
    #
    def self.gpu(device_id : Int32 = 0)
      new(:gpu, device_id)
    end

    # Queries CUDA for the number of GPUs present.
    #
    # Returns the number of GPUs.
    #
    # Note: not supported on MXNet versions < 1.3.0.
    #
    def self.num_gpus
      {% if compare_versions(MXNet::Internal::MXNET_VERSION, "1.3.0") >= 0 %}
        Internal.libcall(MXGetGPUCount, out count)
        count
      {% else %}
        raise MXNetException.new("not supported on MXNet version #{MXNet::Internal::MXNET_VERSION}")
      {% end %}
    end

    # Queries CUDA for the free and total bytes of GPU global memory.
    #
    # Returns the free and total memory as a two-element tuple.
    #
    # ### Parameters
    # * *device_id* (`Int32`, default = 0)
    #   Device id of the device.
    #
    # Note: not supported on MXNet versions < 1.3.0.
    #
    def self.gpu_memory_info(device_id : Int32 = 0)
      {% if compare_versions(MXNet::Internal::MXNET_VERSION, "1.4.0") >= 0 %}
        Internal.libcall(MXGetGPUMemoryInformation64, device_id, out free_mem, out total_mem)
        {free_mem, total_mem}
      {% elsif compare_versions(MXNet::Internal::MXNET_VERSION, "1.3.0") >= 0 %}
        Internal.libcall(MXGetGPUMemoryInformation, device_id, out free_mem, out total_mem)
        {free_mem.to_u64, total_mem.to_u64}
      {% else %}
        raise MXNetException.new("not supported on MXNet version #{MXNet::Internal::MXNET_VERSION}")
      {% end %}
    end

    # Returns the current context.
    #
    def self.current
      @@default
    end

    def device_type
      DEVICE_TYPE_INT_TO_SYM[@device_type]
    end

    def device_id
      @device_id
    end

    def device : {Int32, Int32}
      {@device_type, @device_id}
    end

    # Compares contexts.
    #
    # Two contexts are equal if they have the same device type and device id.
    #
    def ==(other : self)
      other.device_type == self.device_type && other.device_id == self.device_id
    end

    def to_s(io)
      io << device_type << "(" << device_id << ")"
    end
  end
end
