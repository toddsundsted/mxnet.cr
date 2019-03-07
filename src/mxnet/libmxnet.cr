module MXNet
  class Internal
    class LibraryException < Exception
    end

    macro libcall(expr, *args)
      unless MXNet::Internal::LibMXNet.{{ expr }}({{ *args }}) == 0
        raise MXNet::Internal::LibraryException.new(String.new(MXNet::Internal::LibMXNet.MXGetLastError))
      end
    end

    @[Link("mxnet")]

    lib LibMXNet
      type NDArrayHandle = Void*
      type SymbolHandle = Void*
      type ExecutorHandle = Void*
      type OpHandle = Void*

      alias MXUInt = UInt32
      alias NNUInt = UInt32

      fun MXGetLastError() : UInt8*
      fun MXGetVersion(i : Int32*) : Int32
      fun MXNDArrayCreateEx(
        shape : UInt32*,
        ndim : UInt32,
        dev_type : Int32, dev_id : Int32,
        delay_alloc : Int32,
        dtype : Int32,
        handle : NDArrayHandle*
      ) : Int32
      fun MXNDArraySyncCopyFromCPU(handle : NDArrayHandle, data : Void*, size : LibC::SizeT) : Int32
      fun MXNDArraySyncCopyToCPU(handle : NDArrayHandle, data : Void*, size : LibC::SizeT) : Int32
      fun MXNDArrayGetShape(handle : NDArrayHandle, dim : UInt32*, pdata : UInt32**) : Int32
      fun MXNDArrayGetContext(handle : NDArrayHandle, dev_type : Int32*, dev_id : Int32*) : Int32
      fun MXNDArrayGetDType(handle : NDArrayHandle, dtype : UInt32*) : Int32
      fun MXNDArrayFree(handle : NDArrayHandle) : Int32
      fun NNGetOpHandle(name : UInt8*, op : OpHandle*) : Int32
      fun MXImperativeInvoke(
        creator : OpHandle,
        num_inputs : Int32, inputs : NDArrayHandle*,
        num_outputs : Int32*, outputs : NDArrayHandle**,
        num_params : Int32, param_keys : UInt8**, param_vals : UInt8**
      ) : Int32
      fun MXSymbolCreateVariable(
        name : UInt8*,
        handle : SymbolHandle*
      ) : Int32
      fun MXSymbolGetName(handle : SymbolHandle, name : UInt8**, success : Int32*) : Int32
      fun MXSymbolListArguments(handle : SymbolHandle, size : MXUInt*, str_array : UInt8***) : Int32
      fun MXSymbolListOutputs(handle : SymbolHandle, size : MXUInt*, str_array : UInt8***) : Int32
      fun MXSymbolFree(handle : SymbolHandle) : Int32
      fun MXExecutorBindEX(
        handle : SymbolHandle,
        dev_type : Int32, dev_id : Int32,
        num_map_keys : MXUInt,
        map_keys : UInt8**,
        map_dev_types : Int32*,
        map_dev_ids : Int32*,
        len : MXUInt,
        in_args : NDArrayHandle*,
        arg_grad_store : NDArrayHandle*,
        grad_req_type : MXUInt*,
        aux_states_len : MXUInt,
        aux_states : NDArrayHandle*,
        shared_exec : ExecutorHandle,
        exec_handle : ExecutorHandle*
      ) : Int32
      fun MXExecutorForward(
        handle : ExecutorHandle,
        is_train : Int32
      ) : Int32
      fun MXExecutorBackwardEx(
        handle : ExecutorHandle,
        len : MXUInt,
        head_grads : NDArrayHandle*,
        is_train : Int32
      ) : Int32
      fun MXExecutorOutputs(
        handle : ExecutorHandle,
        num_outputs : UInt32*,
        outputs : NDArrayHandle**
      ) : Int32
      fun MXExecutorFree(handle : ExecutorHandle) : Int32
      fun MXSymbolCreateAtomicSymbol(
        creator : OpHandle,
        num_param : MXUInt,
        keys : UInt8**,
        vals : UInt8**,
        sym_handle : SymbolHandle*
      ) : Int32
      fun NNSymbolCompose(
        handle : SymbolHandle,
        name : UInt8*,
        num_args : NNUInt,
        keys : UInt8**,
        args : SymbolHandle*
      ) : Int32
    end
  end
end
