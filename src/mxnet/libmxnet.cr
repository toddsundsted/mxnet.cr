module MXNet
  module Internal
    class LibraryException < Exception
    end

    macro libcall(expr, *args)
      unless MXNet::Internal::LibMXNet.{{ expr }}({{ *args }}) == 0
        raise MXNet::Internal::LibraryException.new(String.new(MXNet::Internal::LibMXNet.MXGetLastError))
      end
    end

    @[Link(ldflags: "`python '#{__DIR__}/libmxnet.py' library`")]

    lib LibMXNet
      type NDArrayHandle = Void*
      type SymbolHandle = Void*
      type ExecutorHandle = Void*
      type CachedOpHandle = Void*
      type OpHandle = Void*

      alias MXUInt = UInt32
      alias NNUInt = UInt32

      fun MXGetLastError() : UInt8*
      fun MXGetVersion(
        version : Int32*
      ) : Int32
      fun MXGetGPUCount(
        count : Int32*
      ) : Int32
      fun MXGetGPUMemoryInformation(
        dev_id : Int32,
        free_mem : Int32*,
        total_mem : Int32*
      ) : Int32
      fun MXGetGPUMemoryInformation64(
        dev_id : Int32,
        free_mem : UInt64*,
        total_mem : UInt64*
      ) : Int32

      # NDArray
      fun MXNDArrayCreateEx(
        shape : UInt32*,
        ndim : UInt32,
        dev_type : Int32, dev_id : Int32,
        delay_alloc : Int32,
        dtype : Int32,
        handle : NDArrayHandle*
      ) : Int32
      fun MXNDArrayFree(
        handle : NDArrayHandle
      ) : Int32
      fun MXNDArraySyncCopyFromCPU(
        handle : NDArrayHandle,
        data : Void*,
        size : LibC::SizeT
      ) : Int32
      fun MXNDArraySyncCopyToCPU(
        handle : NDArrayHandle,
        data : Void*,
        size : LibC::SizeT
      ) : Int32
      fun MXNDArrayGetGrad(
        handle : NDArrayHandle,
        grad_handle : NDArrayHandle*
      ) : Int32
      fun MXNDArrayGetShape(
        handle : NDArrayHandle,
        dim : UInt32*,
        pdata : UInt32**
      ) : Int32
      fun MXNDArrayGetContext(
        handle : NDArrayHandle,
        dev_type : Int32*,
        dev_id : Int32*
      ) : Int32
      fun MXNDArrayGetDType(
        handle : NDArrayHandle,
        dtype : UInt32*
      ) : Int32
      fun MXNDArraySave(
        fname : UInt8*,
        num_args : MXUInt,
        args : NDArrayHandle*,
        keys : UInt8**
      ) : Int32
      fun MXNDArrayLoad(
        fname : UInt8*,
        size : MXUInt*,
        arr : NDArrayHandle**,
        name_size : MXUInt*,
        names : UInt8***
      ) : Int32

      # Symbol
      fun MXSymbolCreateGroup(
        num_symbols : MXUInt,
        symbols : SymbolHandle*,
        out : SymbolHandle*
      ) : Int32
      fun MXSymbolCreateVariable(
        name : UInt8*,
        handle : SymbolHandle*
      ) : Int32
      fun MXSymbolFree(
        handle : SymbolHandle
      ) : Int32
      fun MXSymbolGetName(
        handle : SymbolHandle,
        name : UInt8**,
        success : Int32*
      ) : Int32
      fun MXSymbolSetAttr(
        handle : SymbolHandle,
        key : UInt8*,
        value : UInt8*
      ) : Int32
      fun MXSymbolGetAttr(
        handle : SymbolHandle,
        key : UInt8*,
        value : UInt8**,
        success : Int32*
      ) : Int32
      fun MXSymbolListAttrShallow(
        handle : SymbolHandle,
        size : MXUInt*,
        out : UInt8***
      ) : Int32
      fun MXSymbolListAttr(
        handle : SymbolHandle,
        size : MXUInt*,
        out : UInt8***
      ) : Int32
      fun MXSymbolListArguments(
        handle : SymbolHandle,
        size : MXUInt*,
        str_array : UInt8***
      ) : Int32
      fun MXSymbolListOutputs(
        handle : SymbolHandle,
        size : MXUInt*,
        str_array : UInt8***
      ) : Int32
      fun MXSymbolListAuxiliaryStates(
        handle : SymbolHandle,
        size : MXUInt*,
        str_array : UInt8***
      ) : Int32
      fun NNSymbolCompose(
        handle : SymbolHandle,
        name : UInt8*,
        num_args : NNUInt,
        keys : UInt8**,
        args : SymbolHandle*
      ) : Int32
      fun MXSymbolInferShape(
        handle : SymbolHandle,
        num_args : MXUInt,
        keys : UInt8**,
        arg_ind_ptr : MXUInt*,
        arg_shape_data : MXUInt*,
        in_shape_size : MXUInt*,
        in_shape_ndim : MXUInt**,
        in_shape_data : MXUInt***,
        out_shape_size : MXUInt*,
        out_shape_ndim : MXUInt**,
        out_shape_data : MXUInt***,
        aux_shape_size : MXUInt*,
        aux_shape_ndim : MXUInt**,
        aux_shape_data : MXUInt***,
        complete : Int32*
      ) : Int32
      fun MXSymbolInferShapePartial(
        handle : SymbolHandle,
        num_args : MXUInt,
        keys : UInt8**,
        arg_ind_ptr : MXUInt*,
        arg_shape_data : MXUInt*,
        in_shape_size : MXUInt*,
        in_shape_ndim : MXUInt**,
        in_shape_data : MXUInt***,
        out_shape_size : MXUInt*,
        out_shape_ndim : MXUInt**,
        out_shape_data : MXUInt***,
        aux_shape_size : MXUInt*,
        aux_shape_ndim : MXUInt**,
        aux_shape_data : MXUInt***,
        complete : Int32*
      ) : Int32
      fun MXSymbolInferType(
        handle : SymbolHandle,
        num_args : MXUInt,
        keys : UInt8**,
        arg_type_data : Int32*,
        in_type_size : MXUInt*,
        in_type_data : Int32**,
        out_type_size : MXUInt*,
        out_type_data : Int32**,
        aux_type_size : MXUInt*,
        aux_type_data : Int32**,
        complete : Int32*
      ) : Int32
      fun MXSymbolInferTypePartial(
        handle : SymbolHandle,
        num_args : MXUInt,
        keys : UInt8**,
        arg_type_data : Int32*,
        in_type_size : MXUInt*,
        in_type_data : Int32**,
        out_type_size : MXUInt*,
        out_type_data : Int32**,
        aux_type_size : MXUInt*,
        aux_type_data : Int32**,
        complete : Int32*
      ) : Int32

      # Autograd
      fun MXAutogradMarkVariables(
        num_var : MXUInt,
        var_handles : NDArrayHandle*,
        reqs_array : MXUInt*,
        grad_handles : NDArrayHandle*
      ) : Int32
      fun MXAutogradBackwardEx(
        num_outputs : MXUInt,
        output_handles : NDArrayHandle*,
        ograd_handles : NDArrayHandle*,
        num_variables : MXUInt,
        var_handles : NDArrayHandle*,
        retain_graph : Int32,
        create_graph : Int32,
        is_train : Int32,
        grad_handles : NDArrayHandle **,
        grad_stypes : Int32**
      ) : Int32
      fun MXAutogradSetIsRecording(
        is_recording : Int32,
        previous : Int32*
      ) : Int32
      fun MXAutogradSetIsTraining(
        is_training : Int32,
        previous : Int32*
      ) : Int32
      fun MXAutogradIsRecording(
        current : Bool*
      ) : Int32
      fun MXAutogradIsTraining(
        current : Bool*
      ) : Int32

      # Executor
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
      fun MXExecutorFree(
        handle : ExecutorHandle
      ) : Int32

      # Random
      fun MXRandomSeed(
        seed : Int32
      ) : Int32
      fun MXRandomSeedContext(
        seed : Int32,
        dev_type : Int32, dev_id : Int32
      ) : Int32

      # Cached Op
      fun MXCreateCachedOpEx(
        handle : SymbolHandle,
        num_flags : Int32,
        keys : UInt8**,
        vals : UInt8**,
        out : CachedOpHandle*
      ) : Int32
      fun MXFreeCachedOp(
        handle : CachedOpHandle
      ) : Int32
      fun MXInvokeCachedOpEx(
        handle : CachedOpHandle,
        num_inputs : Int32,
        inputs : NDArrayHandle*,
        num_outputs : Int32*,
        outputs : NDArrayHandle**,
        stypes : Int32**
      ) : Int32

      fun NNGetOpHandle(
        name : UInt8*,
        op : OpHandle*
      ) : Int32
      fun MXImperativeInvoke(
        creator : OpHandle,
        num_inputs : Int32,
        inputs : NDArrayHandle*,
        num_outputs : Int32*,
        outputs : NDArrayHandle**,
        num_params : Int32,
        param_keys : UInt8**,
        param_vals : UInt8**
      ) : Int32
      fun MXSymbolCreateAtomicSymbol(
        creator : OpHandle,
        num_param : MXUInt,
        keys : UInt8**,
        vals : UInt8**,
        sym_handle : SymbolHandle*
      ) : Int32
    end
  end
end
