module MXNet
  # Extends `MXNet::NDArray` and `MXNet::Symbol` classes with
  # wrappers for native MXNet operations.
  #
  module Operations
    # :nodoc:
    OP_INFO = {
      # the "manual" tag indicates definitions that differ from those
      # exported from the MXNet library (via `MXSymbolGetAtomicSymbolInfo`)
      "Activation": {"Activation",["data"],["act_type"],nil},
      "BatchNorm": {"BatchNorm",["data","gamma","beta","moving_mean","moving_var"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var","axis","cudnn_off"]},
      "BatchNorm_v1": {"BatchNorm_v1",["data","gamma","beta"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var"]},
      "BilinearSampler": {"BilinearSampler",["data","grid"],nil,nil},
      "BlockGrad": {"BlockGrad",["data"],nil,nil},
      "Cast": {"Cast",["data"],["dtype"],nil},
      "Concat": {"Concat",["*data"],["num_args"],["dim"]},
      "Convolution": {"Convolution",["data","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      "Convolution_v1": {"Convolution_v1",["data","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      "Correlation": {"Correlation",["data1","data2"],nil,["kernel_size","max_displacement","stride1","stride2","pad_size","is_multiply"]},
      "Crop": {"Crop",nil,["num_args"],["offset","h_w","center_crop"]},
      "CuDNNBatchNorm": {"CuDNNBatchNorm",["data","gamma","beta","moving_mean","moving_var"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var","axis","cudnn_off"]},
      "Custom": {"Custom",["*data"],["op_type"],nil},
      "Deconvolution": {"Deconvolution",["data","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","adj","target_shape","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      "Dropout": {"Dropout",["data"],nil,["p","mode","axes"]},
      "ElementWiseSum": {"add_n",["*args"],nil,nil},
      "Embedding": {"Embedding",["data","weight"],["input_dim","output_dim"],["dtype","sparse_grad"]},
      "Flatten": {"Flatten",["data"],nil,nil},
      "FullyConnected": {"FullyConnected",["data","weight","bias"],["num_hidden"],["no_bias","flatten"]},
      "GridGenerator": {"GridGenerator",["data"],["transform_type"],["target_shape"]},
      "IdentityAttachKLSparseReg": {"IdentityAttachKLSparseReg",["data"],nil,["sparseness_target","penalty","momentum"]},
      "InstanceNorm": {"InstanceNorm",["data","gamma","beta"],nil,["eps"]},
      "L2Normalization": {"L2Normalization",["data"],nil,["eps","mode"]},
      "LRN": {"LRN",["data"],["nsize"],["alpha","beta","knorm"]},
      "LayerNorm": {"LayerNorm",["data","gamma","beta"],nil,["axis","eps","output_mean_var"]},
      "LeakyReLU": {"LeakyReLU",["data","gamma"],nil,["act_type","slope","lower_bound","upper_bound"]},
      "LinearRegressionOutput": {"LinearRegressionOutput",["data","label"],nil,["grad_scale"]},
      "LogisticRegressionOutput": {"LogisticRegressionOutput",["data","label"],nil,["grad_scale"]},
      "MAERegressionOutput": {"MAERegressionOutput",["data","label"],nil,["grad_scale"]},
      "MakeLoss": {"MakeLoss",["data"],nil,["grad_scale","valid_thresh","normalization"]},
      "Pad": {"Pad",["data"],["mode","pad_width"],["constant_value"]},
      "Pooling": {"Pooling",["data"],nil,["kernel","pool_type","global_pool","cudnn_off","pooling_convention","stride","pad","p_value","count_include_pad"]},
      "Pooling_v1": {"Pooling_v1",["data"],nil,["kernel","pool_type","global_pool","pooling_convention","stride","pad"]},
      "RNN": {"RNN",["data","parameters","state","state_cell"],["state_size","num_layers","mode"],["bidirectional","p","state_outputs"]},
      "ROIPooling": {"ROIPooling",["data","rois"],["pooled_size","spatial_scale"],nil},
      "Reshape": {"Reshape",["data"],nil,["shape","reverse","target_shape","keep_highest"]},
      "SVMOutput": {"SVMOutput",["data","label"],nil,["margin","regularization_coefficient","use_linear"]},
      "SequenceLast": {"SequenceLast",["data","sequence_length"],nil,["use_sequence_length","axis"]},
      "SequenceMask": {"SequenceMask",["data","sequence_length"],nil,["use_sequence_length","value","axis"]},
      "SequenceReverse": {"SequenceReverse",["data","sequence_length"],nil,["use_sequence_length","axis"]},
      "SliceChannel": {"SliceChannel",["data"],["num_outputs"],["axis","squeeze_axis"]},
      "Softmax": {"Softmax",["data"],nil,["grad_scale","ignore_label","multi_output","use_ignore","preserve_shape","normalization","out_grad","smooth_alpha"]},
      "SoftmaxActivation": {"SoftmaxActivation",["data"],nil,["mode"]},
      "SoftmaxOutput": {"SoftmaxOutput",["data","label"],nil,["grad_scale","ignore_label","multi_output","use_ignore","preserve_shape","normalization","out_grad","smooth_alpha"]},
      "SpatialTransformer": {"SpatialTransformer",["data","loc"],["transform_type","sampler_type"],["target_shape"]},
      "SwapAxis": {"SwapAxis",["data"],nil,["dim1","dim2"]},
      "UpSampling": {"UpSampling",["*data"],["scale","sample_type","num_args"],["num_filter","multi_input_mode","workspace"]},
      "_CachedOp": {"_CachedOp",nil,nil,nil},
      "_CrossDeviceCopy": {"_CrossDeviceCopy",nil,nil,nil},
      "_CustomFunction": {"_CustomFunction",nil,nil,nil},
      "_Div": {"elemwise_div",["lhs","rhs"],nil,nil},
      "_DivScalar": {"_div_scalar",["data"],["scalar"],nil},
      "_Equal": {"_equal",["lhs","rhs"],nil,nil},
      "_EqualScalar": {"_equal_scalar",["data"],["scalar"],nil},
      "_Greater": {"_greater",["lhs","rhs"],nil,nil},
      "_GreaterEqualScalar": {"_greater_equal_scalar",["data"],["scalar"],nil},
      "_GreaterScalar": {"_greater_scalar",["data"],["scalar"],nil},
      "_Greater_Equal": {"_greater_equal",["lhs","rhs"],nil,nil},
      "_Hypot": {"_hypot",["lhs","rhs"],nil,nil},
      "_HypotScalar": {"_hypot_scalar",["data"],["scalar"],nil},
      "_Lesser": {"_lesser",["lhs","rhs"],nil,nil},
      "_LesserEqualScalar": {"_lesser_equal_scalar",["data"],["scalar"],nil},
      "_LesserScalar": {"_lesser_scalar",["data"],["scalar"],nil},
      "_Lesser_Equal": {"_lesser_equal",["lhs","rhs"],nil,nil},
      "_LogicalAndScalar": {"_logical_and_scalar",["data"],["scalar"],nil},
      "_LogicalOrScalar": {"_logical_or_scalar",["data"],["scalar"],nil},
      "_LogicalXorScalar": {"_logical_xor_scalar",["data"],["scalar"],nil},
      "_Logical_And": {"_logical_and",["lhs","rhs"],nil,nil},
      "_Logical_Or": {"_logical_or",["lhs","rhs"],nil,nil},
      "_Logical_Xor": {"_logical_xor",["lhs","rhs"],nil,nil},
      "_Maximum": {"_maximum",["lhs","rhs"],nil,nil},
      "_MaximumScalar": {"_maximum_scalar",["data"],["scalar"],nil},
      "_Minimum": {"_minimum",["lhs","rhs"],nil,nil},
      "_MinimumScalar": {"_minimum_scalar",["data"],["scalar"],nil},
      "_Minus": {"elemwise_sub",["lhs","rhs"],nil,nil},
      "_MinusScalar": {"_minus_scalar",["data"],["scalar"],nil},
      "_Mod": {"_mod",["lhs","rhs"],nil,nil},
      "_ModScalar": {"_mod_scalar",["data"],["scalar"],nil},
      "_Mul": {"elemwise_mul",["lhs","rhs"],nil,nil},
      "_MulScalar": {"_mul_scalar",["data"],["scalar"],nil},
      "_NDArray": {"_NDArray",["*data"],["info"],nil},
      "_Native": {"_Native",["*data"],["info"],["need_top_grad"]},
      "_NoGradient": {"_NoGradient",nil,nil,nil},
      "_NotEqualScalar": {"_not_equal_scalar",["data"],["scalar"],nil},
      "_Not_Equal": {"_not_equal",["lhs","rhs"],nil,nil},
      "_Plus": {"elemwise_add",["lhs","rhs"],nil,nil},
      "_PlusScalar": {"_plus_scalar",["data"],["scalar"],nil},
      "_Power": {"_power",["lhs","rhs"],nil,nil},
      "_PowerScalar": {"_power_scalar",["data"],["scalar"],nil},
      "_RDivScalar": {"_rdiv_scalar",["data"],["scalar"],nil},
      "_RMinusScalar": {"_rminus_scalar",["data"],["scalar"],nil},
      "_RModScalar": {"_rmod_scalar",["data"],["scalar"],nil},
      "_RPowerScalar": {"_rpower_scalar",["data"],["scalar"],nil},
      "_add": {"elemwise_add",["lhs","rhs"],nil,nil},
      "_arange": {"_arange",nil,["start"],["stop","step","repeat","ctx","dtype"]},
      "_broadcast_backward": {"_broadcast_backward",nil,nil,nil},
      "_cond": {"_cond",["*data"],["num_args","num_outputs","cond_input_locs","then_input_locs","else_input_locs"],nil},
      "_contrib_AdaptiveAvgPooling2D": {"_contrib_AdaptiveAvgPooling2D",["data"],nil,["output_size"]},
      "_contrib_BilinearResize2D": {"_contrib_BilinearResize2D",["data"],["height","width"],nil},
      "_contrib_CTCLoss": {"_contrib_CTCLoss",["data","label","data_lengths","label_lengths"],nil,["use_data_lengths","use_label_lengths","blank_label"]},
      "_contrib_DeformableConvolution": {"_contrib_DeformableConvolution",["data","offset","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","num_deformable_group","workspace","no_bias","layout"]},
      "_contrib_DeformablePSROIPooling": {"_contrib_DeformablePSROIPooling",nil,["spatial_scale","output_dim","group_size","pooled_size"],["part_size","sample_per_part","trans_std","no_trans"]},
      "_contrib_MultiBoxDetection": {"_contrib_MultiBoxDetection",["cls_prob","loc_pred","anchor"],nil,["clip","threshold","background_id","nms_threshold","force_suppress","variances","nms_topk"]},
      "_contrib_MultiBoxPrior": {"_contrib_MultiBoxPrior",["data"],nil,["sizes","ratios","clip","steps","offsets"]},
      "_contrib_MultiBoxTarget": {"_contrib_MultiBoxTarget",["anchor","label","cls_pred"],nil,["overlap_threshold","ignore_label","negative_mining_ratio","negative_mining_thresh","minimum_negative_samples","variances"]},
      "_contrib_MultiProposal": {"_contrib_MultiProposal",["cls_prob","bbox_pred","im_info"],nil,["rpn_pre_nms_top_n","rpn_post_nms_top_n","threshold","rpn_min_size","scales","ratios","feature_stride","output_score","iou_loss"]},
      "_contrib_PSROIPooling": {"_contrib_PSROIPooling",nil,["spatial_scale","output_dim","pooled_size"],["group_size"]},
      "_contrib_Proposal": {"_contrib_Proposal",["cls_prob","bbox_pred","im_info"],nil,["rpn_pre_nms_top_n","rpn_post_nms_top_n","threshold","rpn_min_size","scales","ratios","feature_stride","output_score","iou_loss"]},
      "_contrib_ROIAlign": {"_contrib_ROIAlign",["data","rois"],["pooled_size","spatial_scale"],["sample_ratio"]},
      "_contrib_SparseEmbedding": {"_contrib_SparseEmbedding",["data","weight"],["input_dim","output_dim"],["dtype","sparse_grad"]},
      "_contrib_SyncBatchNorm": {"_contrib_SyncBatchNorm",["data","gamma","beta","moving_mean","moving_var"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var","ndev","key"]},
      "_contrib_backward_quadratic": {"_contrib_backward_quadratic",nil,nil,nil},
      "_contrib_bipartite_matching": {"_contrib_bipartite_matching",["data"],["threshold"],["is_ascend","topk"]},
      "_contrib_box_iou": {"_contrib_box_iou",["lhs","rhs"],nil,["format"]},
      "_contrib_box_nms": {"_contrib_box_nms",["data"],nil,["overlap_thresh","valid_thresh","topk","coord_start","score_index","id_index","force_suppress","in_format","out_format"]},
      "_contrib_box_non_maximum_suppression": {"_contrib_box_nms",["data"],nil,["overlap_thresh","valid_thresh","topk","coord_start","score_index","id_index","force_suppress","in_format","out_format"]},
      "_contrib_count_sketch": {"_contrib_count_sketch",["data","h","s"],["out_dim"],["processing_batch_size"]},
      "_contrib_ctc_loss": {"_contrib_CTCLoss",["data","label","data_lengths","label_lengths"],nil,["use_data_lengths","use_label_lengths","blank_label"]},
      "_contrib_dequantize": {"_contrib_dequantize",["data","min_range","max_range"],nil,["out_type"]},
      "_contrib_div_sqrt_dim": {"_contrib_div_sqrt_dim",["data"],nil,nil},
      "_contrib_fft": {"_contrib_fft",["data"],nil,["compute_size"]},
      "_contrib_ifft": {"_contrib_ifft",["data"],nil,["compute_size"]},
      "_contrib_quadratic": {"_contrib_quadratic",["data"],nil,["a","b","c"]},
      "_contrib_quantize": {"_contrib_quantize",["data","min_range","max_range"],nil,["out_type"]},
      "_contrib_quantized_conv": {"_contrib_quantized_conv",["data","weight","bias","min_data","max_data","min_weight","max_weight","min_bias","max_bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      "_contrib_quantized_flatten": {"_contrib_quantized_flatten",["data","min_data","max_data"],nil,nil},
      "_contrib_quantized_fully_connected": {"_contrib_quantized_fully_connected",["data","weight","bias","min_data","max_data","min_weight","max_weight","min_bias","max_bias"],["num_hidden"],["no_bias","flatten"]},
      "_contrib_quantized_pooling": {"_contrib_quantized_pooling",["data","min_data","max_data"],nil,["kernel","pool_type","global_pool","cudnn_off","pooling_convention","stride","pad","p_value","count_include_pad"]},
      "_contrib_requantize": {"_contrib_requantize",["data","min_range","max_range"],nil,["min_calib_range","max_calib_range"]},
      "_copy": {"_copy",["data"],nil,nil},
      "_copyto": {"_copyto",["data"],nil,nil},
      "_crop_assign": {"_slice_assign",["lhs","rhs"],["begin","end"],["step"]},
      "_crop_assign_scalar": {"_slice_assign_scalar",["data"],["begin","end"],["scalar","step"]},
      "_cvcopyMakeBorder": {"_cvcopyMakeBorder",nil,["top","bot","left","right"],["type","value","values"]},
      "_cvimdecode": {"_cvimdecode",nil,nil,["flag","to_rgb"]},
      "_cvimread": {"_cvimread",nil,["filename"],["flag","to_rgb"]},
      "_cvimresize": {"_cvimresize",nil,["w","h"],["interp"]},
      "_div": {"elemwise_div",["lhs","rhs"],nil,nil},
      "_div_scalar": {"_div_scalar",["data"],["scalar"],nil},
      "_equal": {"_equal",["lhs","rhs"],nil,nil},
      "_equal_scalar": {"_equal_scalar",["data"],["scalar"],nil},
      "_eye": {"_eye",nil,["N"],["M","k","ctx","dtype"]},
      "_foreach": {"_foreach",["*data"],["num_args","num_outputs","num_out_data","in_state_locs","in_data_locs","remain_locs"],nil},
      "_full": {"_full",nil,["value"],["shape","ctx","dtype"]},
      "_grad_add": {"_grad_add",["lhs","rhs"],nil,nil},
      "_greater": {"_greater",["lhs","rhs"],nil,nil},
      "_greater_equal": {"_greater_equal",["lhs","rhs"],nil,nil},
      "_greater_equal_scalar": {"_greater_equal_scalar",["data"],["scalar"],nil},
      "_greater_scalar": {"_greater_scalar",["data"],["scalar"],nil},
      "_histogram": {"_histogram",["data","bins"],nil,["bin_cnt","range"]},
      "_hypot": {"_hypot",["lhs","rhs"],nil,nil},
      "_hypot_scalar": {"_hypot_scalar",["data"],["scalar"],nil},
      "_identity_with_attr_like_rhs": {"_identity_with_attr_like_rhs",["lhs","rhs"],nil,nil},
      "_image_adjust_lighting": {"_image_adjust_lighting",["data"],["alpha"],nil},
      "_image_flip_left_right": {"_image_flip_left_right",["data"],nil,nil},
      "_image_flip_top_bottom": {"_image_flip_top_bottom",["data"],nil,nil},
      "_image_normalize": {"_image_normalize",["data"],["mean","std"],nil},
      "_image_random_brightness": {"_image_random_brightness",["data"],["min_factor","max_factor"],nil},
      "_image_random_color_jitter": {"_image_random_color_jitter",["data"],["brightness","contrast","saturation","hue"],nil},
      "_image_random_contrast": {"_image_random_contrast",["data"],["min_factor","max_factor"],nil},
      "_image_random_flip_left_right": {"_image_random_flip_left_right",["data"],nil,nil},
      "_image_random_flip_top_bottom": {"_image_random_flip_top_bottom",["data"],nil,nil},
      "_image_random_hue": {"_image_random_hue",["data"],["min_factor","max_factor"],nil},
      "_image_random_lighting": {"_image_random_lighting",["data"],nil,["alpha_std"]},
      "_image_random_saturation": {"_image_random_saturation",["data"],["min_factor","max_factor"],nil},
      "_image_to_tensor": {"_image_to_tensor",["data"],nil,nil},
      "_imdecode": {"_imdecode",["mean"],["index","x0","y0","x1","y1","c","size"],nil},
      "_lesser": {"_lesser",["lhs","rhs"],nil,nil},
      "_lesser_equal": {"_lesser_equal",["lhs","rhs"],nil,nil},
      "_lesser_equal_scalar": {"_lesser_equal_scalar",["data"],["scalar"],nil},
      "_lesser_scalar": {"_lesser_scalar",["data"],["scalar"],nil},
      "_linalg_gelqf": {"_linalg_gelqf",["A"],nil,nil},
      "_linalg_gemm": {"_linalg_gemm",["A","B","C"],nil,["transpose_a","transpose_b","alpha","beta","axis"]},
      "_linalg_gemm2": {"_linalg_gemm2",["A","B"],nil,["transpose_a","transpose_b","alpha","axis"]},
      "_linalg_potrf": {"_linalg_potrf",["A"],nil,nil},
      "_linalg_potri": {"_linalg_potri",["A"],nil,nil},
      "_linalg_sumlogdiag": {"_linalg_sumlogdiag",["A"],nil,nil},
      "_linalg_syevd": {"_linalg_syevd",["A"],nil,nil},
      "_linalg_syrk": {"_linalg_syrk",["A"],nil,["transpose","alpha"]},
      "_linalg_trmm": {"_linalg_trmm",["A","B"],nil,["transpose","rightside","alpha"]},
      "_linalg_trsm": {"_linalg_trsm",["A","B"],nil,["transpose","rightside","alpha"]},
      "_logical_and": {"_logical_and",["lhs","rhs"],nil,nil},
      "_logical_and_scalar": {"_logical_and_scalar",["data"],["scalar"],nil},
      "_logical_or": {"_logical_or",["lhs","rhs"],nil,nil},
      "_logical_or_scalar": {"_logical_or_scalar",["data"],["scalar"],nil},
      "_logical_xor": {"_logical_xor",["lhs","rhs"],nil,nil},
      "_logical_xor_scalar": {"_logical_xor_scalar",["data"],["scalar"],nil},
      "_maximum": {"_maximum",["lhs","rhs"],nil,nil},
      "_maximum_scalar": {"_maximum_scalar",["data"],["scalar"],nil},
      "_minimum": {"_minimum",["lhs","rhs"],nil,nil},
      "_minimum_scalar": {"_minimum_scalar",["data"],["scalar"],nil},
      "_minus": {"elemwise_sub",["lhs","rhs"],nil,nil},
      "_minus_scalar": {"_minus_scalar",["data"],["scalar"],nil},
      "_mod": {"_mod",["lhs","rhs"],nil,nil},
      "_mod_scalar": {"_mod_scalar",["data"],["scalar"],nil},
      "_mul": {"elemwise_mul",["lhs","rhs"],nil,nil},
      "_mul_scalar": {"_mul_scalar",["data"],["scalar"],nil},
      "_not_equal": {"_not_equal",["lhs","rhs"],nil,nil},
      "_not_equal_scalar": {"_not_equal_scalar",["data"],["scalar"],nil},
      "_onehot_encode": {"_onehot_encode",nil,nil,nil},
      "_ones": {"_ones",nil,nil,["shape","ctx","dtype"]},
      "_plus": {"elemwise_add",["lhs","rhs"],nil,nil},
      "_plus_scalar": {"_plus_scalar",["data"],["scalar"],nil},
      "_power": {"_power",["lhs","rhs"],nil,nil},
      "_power_scalar": {"_power_scalar",["data"],["scalar"],nil},
      "_random_exponential": {"_random_exponential",nil,nil,["lam","shape","ctx","dtype"]},
      "_random_gamma": {"_random_gamma",nil,nil,["alpha","beta","shape","ctx","dtype"]},
      "_random_generalized_negative_binomial": {"_random_generalized_negative_binomial",nil,nil,["mu","alpha","shape","ctx","dtype"]},
      "_random_negative_binomial": {"_random_negative_binomial",nil,nil,["k","p","shape","ctx","dtype"]},
      "_random_normal": {"_random_normal",nil,nil,["loc","scale","shape","ctx","dtype"]},
      "_random_poisson": {"_random_poisson",nil,nil,["lam","shape","ctx","dtype"]},
      "_random_uniform": {"_random_uniform",nil,nil,["low","high","shape","ctx","dtype"]},
      "_random_randint": {"_random_randint",nil,["low","high"],["shape","ctx","dtype"]}, # manual
      "_ravel_multi_index": {"_ravel_multi_index",["data"],nil,["shape"]},
      "_rdiv_scalar": {"_rdiv_scalar",["data"],["scalar"],nil},
      "_rminus_scalar": {"_rminus_scalar",["data"],["scalar"],nil},
      "_rmod_scalar": {"_rmod_scalar",["data"],["scalar"],nil},
      "_rpower_scalar": {"_rpower_scalar",["data"],["scalar"],nil},
      "_sample_exponential": {"_sample_exponential",["lam"],nil,["shape","dtype"]},
      "_sample_gamma": {"_sample_gamma",["alpha","beta"],nil,["shape","dtype"]},
      "_sample_generalized_negative_binomial": {"_sample_generalized_negative_binomial",["mu","alpha"],nil,["shape","dtype"]},
      "_sample_multinomial": {"_sample_multinomial",["data"],nil,["shape","get_prob","dtype"]},
      "_sample_negative_binomial": {"_sample_negative_binomial",["k","p"],nil,["shape","dtype"]},
      "_sample_normal": {"_sample_normal",["mu","sigma"],nil,["shape","dtype"]},
      "_sample_poisson": {"_sample_poisson",["lam"],nil,["shape","dtype"]},
      "_sample_uniform": {"_sample_uniform",["low","high"],nil,["shape","dtype"]},
      "_scatter_elemwise_div": {"_scatter_elemwise_div",["lhs","rhs"],nil,nil},
      "_scatter_minus_scalar": {"_scatter_minus_scalar",["data"],["scalar"],nil},
      "_scatter_plus_scalar": {"_scatter_plus_scalar",["data"],["scalar"],nil},
      "_scatter_set_nd": {"_scatter_set_nd",["lhs","rhs","indices"],["shape"],nil},
      "_set_value": {"_set_value",nil,nil,nil},
      "_shuffle": {"_shuffle",["data"],nil,nil},
      "_slice_assign": {"_slice_assign",["lhs","rhs"],["begin","end"],["step"]},
      "_slice_assign_scalar": {"_slice_assign_scalar",["data"],["begin","end"],["scalar","step"]},
      "_sparse_ElementWiseSum": {"add_n",["*args"],nil,nil},
      "_sparse_Embedding": {"Embedding",["data","weight"],["input_dim","output_dim"],["dtype","sparse_grad"]},
      "_sparse_FullyConnected": {"FullyConnected",["data","weight","bias"],["num_hidden"],["no_bias","flatten"]},
      "_sparse_LinearRegressionOutput": {"LinearRegressionOutput",["data","label"],nil,["grad_scale"]},
      "_sparse_LogisticRegressionOutput": {"LogisticRegressionOutput",["data","label"],nil,["grad_scale"]},
      "_sparse_MAERegressionOutput": {"MAERegressionOutput",["data","label"],nil,["grad_scale"]},
      "_sparse_abs": {"abs",["data"],nil,nil},
      "_sparse_adagrad_update": {"_sparse_adagrad_update",["weight","grad","history"],["lr"],["epsilon","wd","rescale_grad","clip_gradient"]},
      "_sparse_adam_update": {"adam_update",["weight","grad","mean","var"],["lr"],["beta1","beta2","epsilon","wd","rescale_grad","clip_gradient","lazy_update"]},
      "_sparse_add_n": {"add_n",["*args"],nil,nil},
      "_sparse_arccos": {"arccos",["data"],nil,nil},
      "_sparse_arccosh": {"arccosh",["data"],nil,nil},
      "_sparse_arcsin": {"arcsin",["data"],nil,nil},
      "_sparse_arcsinh": {"arcsinh",["data"],nil,nil},
      "_sparse_arctan": {"arctan",["data"],nil,nil},
      "_sparse_arctanh": {"arctanh",["data"],nil,nil},
      "_sparse_broadcast_add": {"broadcast_add",["lhs","rhs"],nil,nil},
      "_sparse_broadcast_div": {"broadcast_div",["lhs","rhs"],nil,nil},
      "_sparse_broadcast_minus": {"broadcast_sub",["lhs","rhs"],nil,nil},
      "_sparse_broadcast_mul": {"broadcast_mul",["lhs","rhs"],nil,nil},
      "_sparse_broadcast_plus": {"broadcast_add",["lhs","rhs"],nil,nil},
      "_sparse_broadcast_sub": {"broadcast_sub",["lhs","rhs"],nil,nil},
      "_sparse_cast_storage": {"cast_storage",["data"],["stype"],nil},
      "_sparse_cbrt": {"cbrt",["data"],nil,nil},
      "_sparse_ceil": {"ceil",["data"],nil,nil},
      "_sparse_clip": {"clip",["data"],["a_min","a_max"],nil},
      "_sparse_concat": {"Concat",["*data"],["num_args"],["dim"]},
      "_sparse_cos": {"cos",["data"],nil,nil},
      "_sparse_cosh": {"cosh",["data"],nil,nil},
      "_sparse_degrees": {"degrees",["data"],nil,nil},
      "_sparse_dot": {"dot",["lhs","rhs"],nil,["transpose_a","transpose_b","forward_stype"]},
      "_sparse_elemwise_add": {"elemwise_add",["lhs","rhs"],nil,nil},
      "_sparse_elemwise_div": {"elemwise_div",["lhs","rhs"],nil,nil},
      "_sparse_elemwise_mul": {"elemwise_mul",["lhs","rhs"],nil,nil},
      "_sparse_elemwise_sub": {"elemwise_sub",["lhs","rhs"],nil,nil},
      "_sparse_exp": {"exp",["data"],nil,nil},
      "_sparse_expm1": {"expm1",["data"],nil,nil},
      "_sparse_fix": {"fix",["data"],nil,nil},
      "_sparse_floor": {"floor",["data"],nil,nil},
      "_sparse_ftrl_update": {"ftrl_update",["weight","grad","z","n"],["lr"],["lamda1","beta","wd","rescale_grad","clip_gradient"]},
      "_sparse_gamma": {"gamma",["data"],nil,nil},
      "_sparse_gammaln": {"gammaln",["data"],nil,nil},
      "_sparse_log": {"log",["data"],nil,nil},
      "_sparse_log10": {"log10",["data"],nil,nil},
      "_sparse_log1p": {"log1p",["data"],nil,nil},
      "_sparse_log2": {"log2",["data"],nil,nil},
      "_sparse_make_loss": {"make_loss",["data"],nil,nil},
      "_sparse_mean": {"mean",["data"],nil,["axis","keepdims","exclude"]},
      "_sparse_negative": {"negative",["data"],nil,nil},
      "_sparse_norm": {"norm",["data"],nil,["ord","axis","keepdims"]},
      "_sparse_radians": {"radians",["data"],nil,nil},
      "_sparse_relu": {"relu",["data"],nil,nil},
      "_sparse_retain": {"_sparse_retain",["data","indices"],nil,nil},
      "_sparse_rint": {"rint",["data"],nil,nil},
      "_sparse_round": {"round",["data"],nil,nil},
      "_sparse_rsqrt": {"rsqrt",["data"],nil,nil},
      "_sparse_sgd_mom_update": {"sgd_mom_update",["weight","grad","mom"],["lr"],["momentum","wd","rescale_grad","clip_gradient","lazy_update"]},
      "_sparse_sgd_update": {"sgd_update",["weight","grad"],["lr"],["wd","rescale_grad","clip_gradient","lazy_update"]},
      "_sparse_sigmoid": {"sigmoid",["data"],nil,nil},
      "_sparse_sign": {"sign",["data"],nil,nil},
      "_sparse_sin": {"sin",["data"],nil,nil},
      "_sparse_sinh": {"sinh",["data"],nil,nil},
      "_sparse_slice": {"slice",["data"],["begin","end"],["step"]},
      "_sparse_sqrt": {"sqrt",["data"],nil,nil},
      "_sparse_square": {"square",["data"],nil,nil},
      "_sparse_stop_gradient": {"BlockGrad",["data"],nil,nil},
      "_sparse_sum": {"sum",["data"],nil,["axis","keepdims","exclude"]},
      "_sparse_tan": {"tan",["data"],nil,nil},
      "_sparse_tanh": {"tanh",["data"],nil,nil},
      "_sparse_trunc": {"trunc",["data"],nil,nil},
      "_sparse_where": {"where",["condition","x","y"],nil,nil},
      "_sparse_zeros_like": {"zeros_like",["data"],nil,nil},
      "_square_sum": {"_square_sum",["data"],nil,["axis","keepdims","exclude"]},
      "_sub": {"elemwise_sub",["lhs","rhs"],nil,nil},
      "_unravel_index": {"_unravel_index",["data"],nil,["shape"]},
      "_while_loop": {"_while_loop",["*data"],["num_args","num_outputs","num_out_data","max_iterations","cond_input_locs","func_input_locs","func_var_locs"],nil},
      "_zeros": {"_zeros",nil,nil,["shape","ctx","dtype"]},
      "abs": {"abs",["data"],nil,nil},
      "adam_update": {"adam_update",["weight","grad","mean","var"],["lr"],["beta1","beta2","epsilon","wd","rescale_grad","clip_gradient","lazy_update"]},
      "add_n": {"add_n",["*args"],nil,nil},
      "arccos": {"arccos",["data"],nil,nil},
      "arccosh": {"arccosh",["data"],nil,nil},
      "arcsin": {"arcsin",["data"],nil,nil},
      "arcsinh": {"arcsinh",["data"],nil,nil},
      "arctan": {"arctan",["data"],nil,nil},
      "arctanh": {"arctanh",["data"],nil,nil},
      "argmax": {"argmax",["data"],nil,["axis","keepdims"]},
      "argmax_channel": {"argmax_channel",["data"],nil,nil},
      "argmin": {"argmin",["data"],nil,["axis","keepdims"]},
      "argsort": {"argsort",["data"],nil,["axis","is_ascend"]},
      "batch_dot": {"batch_dot",["lhs","rhs"],nil,["transpose_a","transpose_b","forward_stype"]},
      "batch_take": {"batch_take",["a","indices"],nil,nil},
      "broadcast_add": {"broadcast_add",["lhs","rhs"],nil,nil},
      "broadcast_axes": {"broadcast_axis",["data"],nil,["axis","size"]},
      "broadcast_axis": {"broadcast_axis",["data"],nil,["axis","size"]},
      "broadcast_div": {"broadcast_div",["lhs","rhs"],nil,nil},
      "broadcast_equal": {"broadcast_equal",["lhs","rhs"],nil,nil},
      "broadcast_greater": {"broadcast_greater",["lhs","rhs"],nil,nil},
      "broadcast_greater_equal": {"broadcast_greater_equal",["lhs","rhs"],nil,nil},
      "broadcast_hypot": {"broadcast_hypot",["lhs","rhs"],nil,nil},
      "broadcast_lesser": {"broadcast_lesser",["lhs","rhs"],nil,nil},
      "broadcast_lesser_equal": {"broadcast_lesser_equal",["lhs","rhs"],nil,nil},
      "broadcast_like": {"broadcast_like",["lhs","rhs"],nil,nil},
      "broadcast_logical_and": {"broadcast_logical_and",["lhs","rhs"],nil,nil},
      "broadcast_logical_or": {"broadcast_logical_or",["lhs","rhs"],nil,nil},
      "broadcast_logical_xor": {"broadcast_logical_xor",["lhs","rhs"],nil,nil},
      "broadcast_maximum": {"broadcast_maximum",["lhs","rhs"],nil,nil},
      "broadcast_minimum": {"broadcast_minimum",["lhs","rhs"],nil,nil},
      "broadcast_minus": {"broadcast_sub",["lhs","rhs"],nil,nil},
      "broadcast_mod": {"broadcast_mod",["lhs","rhs"],nil,nil},
      "broadcast_mul": {"broadcast_mul",["lhs","rhs"],nil,nil},
      "broadcast_not_equal": {"broadcast_not_equal",["lhs","rhs"],nil,nil},
      "broadcast_plus": {"broadcast_add",["lhs","rhs"],nil,nil},
      "broadcast_power": {"broadcast_power",["lhs","rhs"],nil,nil},
      "broadcast_sub": {"broadcast_sub",["lhs","rhs"],nil,nil},
      "broadcast_to": {"broadcast_to",["data"],nil,["shape"]},
      "cast": {"Cast",["data"],["dtype"],nil},
      "cast_storage": {"cast_storage",["data"],["stype"],nil},
      "cbrt": {"cbrt",["data"],nil,nil},
      "ceil": {"ceil",["data"],nil,nil},
      "choose_element_0index": {"choose_element_0index",nil,nil,nil},
      "clip": {"clip",["data"],["a_min","a_max"],nil},
      "concat": {"Concat",["*data"],["num_args"],["dim"]},
      "cos": {"cos",["data"],nil,nil},
      "cosh": {"cosh",["data"],nil,nil},
      "crop": {"slice",["data"],["begin","end"],["step"]},
      "degrees": {"degrees",["data"],nil,nil},
      "depth_to_space": {"depth_to_space",["data"],["block_size"],nil},
      "diag": {"diag",["data"],nil,["k"]},
      "dot": {"dot",["lhs","rhs"],nil,["transpose_a","transpose_b","forward_stype"]},
      "elemwise_add": {"elemwise_add",["lhs","rhs"],nil,nil},
      "elemwise_div": {"elemwise_div",["lhs","rhs"],nil,nil},
      "elemwise_mul": {"elemwise_mul",["lhs","rhs"],nil,nil},
      "elemwise_sub": {"elemwise_sub",["lhs","rhs"],nil,nil},
      "exp": {"exp",["data"],nil,nil},
      "expand_dims": {"expand_dims",["data"],["axis"],nil},
      "expm1": {"expm1",["data"],nil,nil},
      "fill_element_0index": {"fill_element_0index",nil,nil,nil},
      "fix": {"fix",["data"],nil,nil},
      "flatten": {"Flatten",["data"],nil,nil},
      "flip": {"reverse",["data"],["axis"],nil},
      "floor": {"floor",["data"],nil,nil},
      "ftml_update": {"ftml_update",["weight","grad","d","v","z"],["lr","t"],["beta1","beta2","epsilon","wd","rescale_grad","clip_grad"]},
      "ftrl_update": {"ftrl_update",["weight","grad","z","n"],["lr"],["lamda1","beta","wd","rescale_grad","clip_gradient"]},
      "gamma": {"gamma",["data"],nil,nil},
      "gammaln": {"gammaln",["data"],nil,nil},
      "gather_nd": {"gather_nd",["data","indices"],nil,nil},
      "hard_sigmoid": {"hard_sigmoid",["data"],nil,["alpha","beta"]},
      "identity": {"_copy",["data"],nil,nil},
      "khatri_rao": {"khatri_rao",["*args"],nil,nil},
      "linalg_gelqf": {"_linalg_gelqf",["A"],nil,nil},
      "linalg_gemm": {"_linalg_gemm",["A","B","C"],nil,["transpose_a","transpose_b","alpha","beta","axis"]},
      "linalg_gemm2": {"_linalg_gemm2",["A","B"],nil,["transpose_a","transpose_b","alpha","axis"]},
      "linalg_potrf": {"_linalg_potrf",["A"],nil,nil},
      "linalg_potri": {"_linalg_potri",["A"],nil,nil},
      "linalg_sumlogdiag": {"_linalg_sumlogdiag",["A"],nil,nil},
      "linalg_syrk": {"_linalg_syrk",["A"],nil,["transpose","alpha"]},
      "linalg_trmm": {"_linalg_trmm",["A","B"],nil,["transpose","rightside","alpha"]},
      "linalg_trsm": {"_linalg_trsm",["A","B"],nil,["transpose","rightside","alpha"]},
      "log": {"log",["data"],nil,nil},
      "log10": {"log10",["data"],nil,nil},
      "log1p": {"log1p",["data"],nil,nil},
      "log2": {"log2",["data"],nil,nil},
      "log_softmax": {"log_softmax",["data"],nil,["axis","temperature"]},
      "logical_not": {"logical_not",["data"],nil,nil},
      "make_loss": {"make_loss",["data"],nil,nil},
      "max": {"max",["data"],nil,["axis","keepdims","exclude"]},
      "max_axis": {"max",["data"],nil,["axis","keepdims","exclude"]},
      "mean": {"mean",["data"],nil,["axis","keepdims","exclude"]},
      "min": {"min",["data"],nil,["axis","keepdims","exclude"]},
      "min_axis": {"min",["data"],nil,["axis","keepdims","exclude"]},
      "mp_sgd_mom_update": {"mp_sgd_mom_update",["weight","grad","mom","weight32"],["lr"],["momentum","wd","rescale_grad","clip_gradient","lazy_update"]},
      "mp_sgd_update": {"mp_sgd_update",["weight","grad","weight32"],["lr"],["wd","rescale_grad","clip_gradient","lazy_update"]},
      "nanprod": {"nanprod",["data"],nil,["axis","keepdims","exclude"]},
      "nansum": {"nansum",["data"],nil,["axis","keepdims","exclude"]},
      "negative": {"negative",["data"],nil,nil},
      "norm": {"norm",["data"],nil,["ord","axis","keepdims"]},
      "normal": {"_random_normal",nil,nil,["loc","scale","shape","ctx","dtype"]},
      "one_hot": {"one_hot",["indices"],["depth"],["on_value","off_value","dtype"]},
      "ones_like": {"ones_like",["data"],nil,nil},
      "pad": {"Pad",["data"],["mode","pad_width"],["constant_value"]},
      "pick": {"pick",["data","index"],nil,["axis","keepdims"]},
      "prod": {"prod",["data"],nil,["axis","keepdims","exclude"]},
      "radians": {"radians",["data"],nil,nil},
      "random_exponential": {"_random_exponential",nil,nil,["lam","shape","ctx","dtype"]},
      "random_gamma": {"_random_gamma",nil,nil,["alpha","beta","shape","ctx","dtype"]},
      "random_generalized_negative_binomial": {"_random_generalized_negative_binomial",nil,nil,["mu","alpha","shape","ctx","dtype"]},
      "random_negative_binomial": {"_random_negative_binomial",nil,nil,["k","p","shape","ctx","dtype"]},
      "random_normal": {"_random_normal",nil,nil,["loc","scale","shape","ctx","dtype"]},
      "random_poisson": {"_random_poisson",nil,nil,["lam","shape","ctx","dtype"]},
      "random_uniform": {"_random_uniform",nil,nil,["low","high","shape","ctx","dtype"]},
      "random_randint": {"_random_randint",nil,["low","high"],["shape","ctx","dtype"]}, # manual
      "ravel_multi_index": {"_ravel_multi_index",["data"],nil,["shape"]},
      "rcbrt": {"rcbrt",["data"],nil,nil},
      "reciprocal": {"reciprocal",["data"],nil,nil},
      "relu": {"relu",["data"],nil,nil},
      "repeat": {"repeat",["data"],["repeats"],["axis"]},
      "reshape": {"Reshape",["data"],["shape"],["reverse","target_shape","keep_highest"]}, # manual
      "reshape_like": {"reshape_like",["lhs","rhs"],nil,nil},
      "reverse": {"reverse",["data"],["axis"],nil},
      "rint": {"rint",["data"],nil,nil},
      "rmsprop_update": {"rmsprop_update",["weight","grad","n"],["lr"],["gamma1","epsilon","wd","rescale_grad","clip_gradient","clip_weights"]},
      "rmspropalex_update": {"rmspropalex_update",["weight","grad","n","g","delta"],["lr"],["gamma1","gamma2","epsilon","wd","rescale_grad","clip_gradient","clip_weights"]},
      "round": {"round",["data"],nil,nil},
      "rsqrt": {"rsqrt",["data"],nil,nil},
      "sample_exponential": {"_sample_exponential",["lam"],nil,["shape","dtype"]},
      "sample_gamma": {"_sample_gamma",["alpha","beta"],nil,["shape","dtype"]},
      "sample_generalized_negative_binomial": {"_sample_generalized_negative_binomial",["mu","alpha"],nil,["shape","dtype"]},
      "sample_multinomial": {"_sample_multinomial",["data"],nil,["shape","get_prob","dtype"]},
      "sample_negative_binomial": {"_sample_negative_binomial",["k","p"],nil,["shape","dtype"]},
      "sample_normal": {"_sample_normal",["mu","sigma"],nil,["shape","dtype"]},
      "sample_poisson": {"_sample_poisson",["lam"],nil,["shape","dtype"]},
      "sample_uniform": {"_sample_uniform",["low","high"],nil,["shape","dtype"]},
      "scatter_nd": {"scatter_nd",["data","indices"],["shape"],nil},
      "sgd_mom_update": {"sgd_mom_update",["weight","grad","mom"],["lr"],["momentum","wd","rescale_grad","clip_gradient","lazy_update"]},
      "sgd_update": {"sgd_update",["weight","grad"],["lr"],["wd","rescale_grad","clip_gradient","lazy_update"]},
      "shape_array": {"shape_array",["data"],nil,nil},
      "shuffle": {"_shuffle",["data"],nil,nil},
      "sigmoid": {"sigmoid",["data"],nil,nil},
      "sign": {"sign",["data"],nil,nil},
      "signsgd_update": {"signsgd_update",["weight","grad"],["lr"],["wd","rescale_grad","clip_gradient"]},
      "signum_update": {"signum_update",["weight","grad","mom"],["lr"],["momentum","wd","rescale_grad","clip_gradient","wd_lh"]},
      "sin": {"sin",["data"],nil,nil},
      "sinh": {"sinh",["data"],nil,nil},
      "size_array": {"size_array",["data"],nil,nil},
      "slice": {"slice",["data"],["begin","end"],["step"]},
      "slice_axis": {"slice_axis",["data"],["axis","begin","end"],nil},
      "slice_like": {"slice_like",["data","shape_like"],nil,["axes"]},
      "smooth_l1": {"smooth_l1",["data"],["scalar"],nil},
      "softmax": {"softmax",["data"],nil,["axis","temperature"]},
      "softmax_cross_entropy": {"softmax_cross_entropy",["data","label"],nil,nil},
      "softsign": {"softsign",["data"],nil,nil},
      "sort": {"sort",["data"],nil,["axis","is_ascend"]},
      "space_to_depth": {"space_to_depth",["data"],["block_size"],nil},
      "split": {"SliceChannel",["data"],["num_outputs"],["axis","squeeze_axis"]},
      "sqrt": {"sqrt",["data"],nil,nil},
      "square": {"square",["data"],nil,nil},
      "squeeze": {"squeeze",["*data"],nil,["axis"]},
      "stack": {"stack",["*data"],["num_args"],["axis"]},
      "stop_gradient": {"BlockGrad",["data"],nil,nil},
      "sum": {"sum",["data"],nil,["axis","keepdims","exclude"]},
      "sum_axis": {"sum",["data"],nil,["axis","keepdims","exclude"]},
      "swapaxes": {"SwapAxis",["data"],nil,["dim1","dim2"]},
      "take": {"take",["a","indices"],nil,["axis","mode"]},
      "tan": {"tan",["data"],nil,nil},
      "tanh": {"tanh",["data"],nil,nil},
      "tile": {"tile",["data"],["reps"],nil},
      "topk": {"topk",["data"],nil,["axis","k","ret_typ","is_ascend"]},
      "transpose": {"transpose",["data"],nil,["axes"]},
      "trunc": {"trunc",["data"],nil,nil},
      "uniform": {"_random_uniform",nil,nil,["low","high","shape","ctx","dtype"]},
      "unravel_index": {"_unravel_index",["data"],nil,["shape"]},
      "where": {"where",["condition","x","y"],nil,nil},
      "zeros_like": {"zeros_like",["data"],nil,nil},
    }

    private macro extended
      class {{@type}}::Ops end
      class {{@type}}::Internal end
      class {{@type}}::Contrib end
      class {{@type}}::Linalg end
      class {{@type}}::Sparse end

      {% for op, ps in MXNet::Operations::OP_INFO %}
        {% op = op.stringify %}
        {% keywords = {"begin", "end"} %}
        {% name = op.gsub(/^(_contrib_|_linalg_|_sparse_|_)/, "") %}
        {% pre = {"_contrib_", "_linalg_", "_sparse_", "_"}.find { |pre| op.starts_with?(pre) } || "" %}
        {% mod = {"_contrib_": "Contrib", "_linalg_": "Linalg", "_sparse_": "Sparse", "_": "Internal"}[pre] || "Ops" %}
        {%
          args = ps[1] && ps[1].map do |a|
            if (a.starts_with?("*"))
              a = a[1..-1].downcase
              t = "Array(#{@type})"
            else
              a = a.downcase
              t = "#{@type}?"
            end
            keywords.includes?(a) ?
              {"#{a.id} _#{a.id} : #{t.id}".id, "_#{a.id}".id} :
              {"#{a.id} : #{t.id}".id, a.id}
          end
          kwargs = ps[2] && ps[2].map do |a|
            a = a.downcase
            keywords.includes?(a) ?
              {"#{a.id} _#{a.id}".id, "#{a.id}: _#{a.id}".id} :
              {a.id, "#{a.id}: #{a.id}".id}
          end
        %}
        {% if args && kwargs %}
          def {{@type}}::{{mod.id}}.{{"_#{name.id}".id}}({{*args.map(&.first)}}, {{*kwargs.map(&.first)}}, **kwargs)
            {% if @type == MXNet::NDArray %}
              {{@type}}.imperative_invoke({{op}}, {{*args.map(&.last)}}, **kwargs.merge({{*kwargs.map(&.last)}}))
            {% elsif @type == MXNet::Symbol %}
              {{@type}}.create_symbol({{op}}, {{*args.map(&.last)}}, **kwargs.merge({{*kwargs.map(&.last)}}))
            {% end %}
          end
        {% elsif args %}
          def {{@type}}::{{mod.id}}.{{"_#{name.id}".id}}({{*args.map(&.first)}}, **kwargs)
            {% if @type == MXNet::NDArray %}
              {{@type}}.imperative_invoke({{op}}, {{*args.map(&.last)}}, **kwargs)
            {% elsif @type == MXNet::Symbol %}
              {{@type}}.create_symbol({{op}}, {{*args.map(&.last)}}, **kwargs)
            {% end %}
          end
        {% elsif kwargs %}
          def {{@type}}::{{mod.id}}.{{"_#{name.id}".id}}({{*kwargs.map(&.first)}}, **kwargs)
            {% if @type == MXNet::NDArray %}
              {{@type}}.imperative_invoke({{op}}, **kwargs.merge({{*kwargs.map(&.last)}}))
            {% elsif @type == MXNet::Symbol %}
              {{@type}}.create_symbol({{op}}, **kwargs.merge({{*kwargs.map(&.last)}}))
            {% end %}
          end
        {% else %}
          def {{@type}}::{{mod.id}}.{{"_#{name.id}".id}}(**kwargs)
            {% if @type == MXNet::NDArray %}
              {{@type}}.imperative_invoke({{op}}, **kwargs)
            {% elsif @type == MXNet::Symbol %}
              {{@type}}.create_symbol({{op}}, **kwargs)
            {% end %}
          end
        {% end %}
      {% end %}
    end

    private macro def_class_and_fluent_method(op, name)
      {% args1 = MXNet::Operations::OP_INFO[name.stringify][1] %}
      {% args2 = MXNet::Operations::OP_INFO[name.stringify][2] %}
      {% args = [args1, args2].reject(&.is_a?(NilLiteral)).reduce { |a, b| a + b } %}
      {% if args.size > 1 %}
        def self.{{name}}({{*args.map { |a| ["begin", "end"].includes?(a) ? "#{a.id} _#{a.id}".id : a.id }}}, **kwargs)
          {{op}}._{{name}}({{*args.map { |a| ["begin", "end"].includes?(a) ? "_#{a.id}".id : a.id }}}, **kwargs)
        end
        # Convenience fluent method for `.{{name}}`.
        def {{name}}({{*args[1..-1].map { |a| ["begin", "end"].includes?(a) ? "#{a.id} _#{a.id}".id : a.id }}}, **kwargs)
          {{@type}}.{{name}}(self, {{*args[1..-1].map { |a| ["begin", "end"].includes?(a) ? "_#{a.id}".id : a.id }}}, **kwargs)
        end
      {% elsif args.size > 0 %}
        def self.{{name}}({{*args.map { |a| ["begin", "end"].includes?(a) ? "#{a.id} _#{a.id}".id : a.id }}}, **kwargs)
          {{op}}._{{name}}({{*args.map { |a| ["begin", "end"].includes?(a) ? "_#{a.id}".id : a.id }}}, **kwargs)
        end
        # Convenience fluent method for `.{{name}}`.
        def {{name}}(**kwargs)
          {{@type}}.{{name}}(self, **kwargs)
        end
      {% else %}
        def self.{{name}}(**kwargs)
          {{op}}._{{name}}(**kwargs)
        end
      {% end %}
    end

    private macro included
      {%
        if @type == MXNet::NDArray
          type = "NDArray".id
          prefix =
            "# * *data* (`NDArray`, required)
             #   Input data.".id
          suffix =
            "# * *out* (`NDArray`, optional)
             #   The output array.".id
        elsif @type == MXNet::Symbol
          type = "Symbol".id
          prefix =
            "# * *data* (`Symbol`, required)
             #   Input data.".id
          suffix =
            "# * *name* (`String`, optional)
             #   Name of the symbol.".id
        else
          type = "".id
          prefix = "".id
          suffix = "".id
        end
      %}

      # Returns the element-wise absolute value of the input.
      #
      # Assume *x* is an array with the following elements:
      #     [-2, 0, 3]
      #
      # Then:
      #     abs(x) # => [2, 0, 3]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, abs)

      # Applies an activation function element-wise to the input.
      #
      # The following activation functions are supported:
      #   * **relu**: Rectified Linear Unit, _y = max(x, 0)_
      #   * **softrelu**: Soft ReLU or SoftPlus, _y = log(1 + exp(x))_
      #   * **tanh**: Hyperbolic tangent, _y = exp(x) − exp(−x) / exp(x) + exp(−x)_
      #   * **sigmoid**: _y = 1 / 1 + exp(−x)_
      #   * **softsign**: _y = x / 1 + abs(x)_
      #
      # ### Parameters
      # * *data* (`{{type}}`, required)
      #   The input array.
      # * *act_type* (`::Symbol`, `:relu`, `:softrelu`, `:tanh`, `:sigmoid`, or `:softsign`, required)
      #   Activation function to be applied.
      {{suffix}}
      #
      def self.activation(data : self, act_type, **kwargs)
        Ops._Activation(data, **kwargs.merge({act_type: act_type}))
      end

      # Adds all input arguments element-wise.
      #
      # *add_n(a1,a2,...,an)=a1+a2+...+an*
      #
      # `.add_n` is potentially more efficient than calling `.add` *n* times.
      #
      # ### Parameters
      # * *data* (`Array({{type}})`, required)
      #   List of arrays to add.
      {{suffix}}
      #
      def self.add_n(data : Array(self), **kwargs)
        Ops._add_n(data, **kwargs.merge({num_args: data.size}))
      end

      # Returns evenly spaced values within a given interval.
      #
      # Values are generated within the half-open interval `[start,
      # stop)`. In other words, the interval includes start but
      # excludes stop.
      #
      # Examples:
      #     arange(3)                                       # => [0.0, 1.0, 2.0]
      #     arange(2, 6)                                    # => [2.0, 3.0, 4.0, 5.0]
      #     arange(2, 6, step: 2)                           # => [2.0, 4.0]
      #     arange(2, 6, step: 1.5, repeat: 2)              # => [2.0, 2.0, 3.5, 3.5, 5.0 , 5.0]
      #     arange(2, 6, step: 2, repeat: 3, dtype: :int32) # => [2, 2, 2, 4, 4, 4]
      #
      # ### Parameters
      {{prefix}}
      # * *start* (`Number`, optional, default = `0.0`)
      #   Start of interval.
      # * *stop* (`Number`, required)
      #   End of interval.
      # * *step* (`Number`, optional, default = `1.0`)
      #   Spacing between values.
      # * *repeat* (`Int`, optional, default = `1`)
      #   Number of times to repeat each value.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output array.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.arange(start : Number, stop : Number? = nil, ctx = Context.current, **kwargs)
        Internal._arange(**kwargs.merge({start: start, stop: stop, ctx: ctx}))
      end

      # Converts each element of the input array from radians to
      # degrees.
      #
      #     degrees([0, 𝜋/2, 𝜋, 3𝜋/2, 2𝜋]) = [0, 90, 180, 270, 360]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, degrees)

      # Returns element-wise inverse cosine of the input array.
      #
      # The input should be in range `[-1, 1]`.
      # The output is in the closed interval `[0, 𝜋]`
      #
      #     arccos([-1, -.707, 0, .707, 1]) = [𝜋, 3𝜋/4, 𝜋/2, 𝜋/4, 0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, arccos)

      # Returns the inverse hyperbolic cosine of the input array,
      # computed element-wise.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, arccosh)

      # Returns element-wise inverse sine of the input array.
      #
      # The input should be in the range `[-1, 1]`.
      # The output is in the closed interval `[-𝜋/2, 𝜋/2]`.
      #
      #     arcsin([-1, -.707, 0, .707, 1]) = [-𝜋/2, -𝜋/4, 0, 𝜋/4, 𝜋/2]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, arcsin)

      # Returns the inverse hyperbolic sine of the input array,
      # computed element-wise.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, arcsinh)

      # Returns element-wise inverse tangent of the input array.
      #
      # The output is in the closed interval `[-𝜋/2, 𝜋/2]`
      #
      #     arctan([-1, 0, 1]) = [-𝜋/4, 0, 𝜋/4]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, arctan)

      # Returns the inverse hyperbolic tangent of the input array,
      # computed element-wise.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, arctanh)

      # Returns indices of the maximum values along an axis.
      #
      # In the case of multiple occurrences of maximum values, the
      # indices corresponding to the first occurrence are returned.
      #
      # Assume *x* is an array with the following elements:
      #     [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
      #
      # Then:
      #     argmax(x, axis: 0) = [1.0, 1.0, 1.0]
      #     argmax(x, axis: 1) = [2.0, 2.0]
      #     argmax(x, axis: 1, keepdims: true) = [[2.0], [2.0]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, optional, default = `-1`)
      #   The axis along which to perform the reduction. If omitted,
      #   the last axis is used.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If true, the reduced axis is left in the result as a
      #   dimension with size one.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, argmax)

      # Returns indices of the minimum values along an axis.
      #
      # In the case of multiple occurrences of minimum values, the
      # indices corresponding to the first occurrence are returned.
      #
      # Assume *x* is an array with the following elements:
      #     [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
      #
      # Then:
      #     argmin(x, axis: 0) = [0.0, 0.0, 0.0]
      #     argmin(x, axis: 1) = [0.0, 0.0]
      #     argmin(x, axis: 1, keepdims: true) = [[0.0], [0.0]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, optional, default = `-1`)
      #   The axis along which to perform the reduction. If omitted,
      #   the last axis is used.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If true, the reduced axis is left in the result as a
      #   dimension with size one.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, argmin)

      # Returns the indices that would sort an input array along the
      # given axis.
      #
      # This function performs sorting along the given axis and
      # returns an array of indices having the same shape as an input
      # array that index data in the sorted order.
      #
      # Assume *x* is an array with the following elements:
      #     [[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]]
      #
      # Then:
      #     argsort(x) = [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]]
      #     argsort(x, axis: 0) = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
      #     argsort(x, axis: None) = [3.0, 1.0, 5.0, 0.0, 4.0, 2.0]
      #     argsort(x, is_ascend: false) = [[2.0, 0.0, 1.0], [1.0, 2.0, 0.0]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `None`, optional, default = `-1`)
      #   The axis along which to choose sort the input tensor. If
      #   omitted, the last axis is used. If `None`, the flattened
      #   array is used.
      # * *is_ascend* (`Bool`, optional, default = false)
      #   Whether to sort in ascending or descending order.
      # * *dtype* (`::Symbol`, optional, default = `:float32`)
      #   The data type of the output indices.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, argsort)

      # Returns element-wise sum of the input arrays with broadcasting.
      #
      # `.broadcast_add` is an alias for `.broadcast_plus`.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_add(x, y) # => [[1, 1, 1], [2, 2, 2]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_add)

      # Broadcasts the input array over particular axis.
      #
      # Broadcasting is allowed on axes with size 1, such as from `[2, 1, 3, 1]`
      # to `[2, 8, 3, 9]`. Elements will be duplicated on the broadcasted
      # axis.
      #
      # Assume *x* is an array with the following elements:
      #     [[[1], [2]]]
      #
      # Then:
      #     broadcast_axis(x, axis: 2, size: 3) = [[[1, 1, 1], [2, 2, 2]]]
      #     broadcast_axis(x, axis: [0, 2], size: [2, 3]) = [[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]]
      #
      # ### Parameters
      {{suffix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis on which to perform the broadcasting.
      # * *size* (`Int` or `Array(Int)`, optional)
      #   Target sizes of the broadcasting axis.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_axis)

      # Returns element-wise division of the input arrays with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[6, 6, 6], [6, 6, 6]] # x
      #     [[2], [3]]             # y
      #
      # Then:
      #     broadcast_div(x, y) # => [[3, 3, 3], [2, 2, 2]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_div)

      # Returns the result of element-wise equal to (`==`) comparison
      # operation with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_equal(x, y) # => [[0, 0, 0], [1, 1, 1]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_equal)

      # Returns the result of element-wise greater than (`>`) comparison
      # operation with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_greater(x, y) # => [[1, 1, 1], [0, 0, 0]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_greater)

      # Returns the result of element-wise greater than or equal to
      # (`>=`) comparison operation with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_greater_equal(x, y) # => [[1, 1, 1], [1, 1, 1]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_greater_equal)

      # Returns the result of element-wise less than (`<`) comparison
      # operation with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_lesser(x, y) # => [[0, 0, 0], [0, 0, 0]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_lesser)

      # Returns the result of element-wise less than or equal to (`<=`)
      # comparison operation with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_lesser_equal(x, y) # => [[0, 0, 0], [1, 1, 1]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_lesser_equal)

      # Broadcasts the left hand side to have the same shape as right
      # hand side.
      #
      # Broadcasting is a mechanism that allows `NDArray` to perform
      # arithmetic operations with other arrays of different shapes
      # efficiently without creating multiple copies of arrays. See:
      # [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
      # for explanation.
      #
      # Broadcasting is allowed on axes with size 1, such as from `[2, 1, 3, 1]`
      # to `[2, 8, 3, 9]`. Elements will be duplicated on the broadcasted
      # axes.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 2, 3]]            # x
      #     [[5, 6, 7], [7, 8, 9]] # y
      #
      # Then:
      #     broadcast_like(x, y) = [[1, 2, 3], [1, 2, 3]])
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      # * *lhs_axes* (`Array(Int)`, optional)
      #   Axes to perform broadcast on in the first input array.
      # * *rhs_axes* (`Array(Int)`, optional)
      #   Axes to copy from the second input array.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_like)

      # Returns element-wise maximum of the input arrays with broadcasting.
      #
      # This function compares two input arrays and returns a new array
      # having the element-wise maxima.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_maximum(x, y) # => [[1, 1, 1], [1, 1, 1]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_maximum)

      # Returns element-wise minimum of the input arrays with broadcasting.
      #
      # This function compares two input arrays and returns a new array
      # having the element-wise minima.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_minimum(x, y) # => [[0, 0, 0], [1, 1, 1]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_minimum)

      # Returns element-wise difference of the input arrays with broadcasting.
      #
      # `.broadcast_minus` is an alias to the function `.broadcast_sub`.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_minus(x, y) # => [[1, 1, 1], [0, 0, 0]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_minus)

      # Returns element-wise product of the input arrays with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_mul(x, y) # => [[0, 0, 0], [1, 1, 1]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_mul)

      # Returns the result of element-wise not equal to (`!=`)
      # comparison operation with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_not_equal(x, y) # => [[1, 1, 1], [0, 0, 0]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input to be compared.
      # * *rhs* (`{{type}}`, required)
      #   The second input to be compared.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_not_equal)

      # Returns element-wise sum of the input arrays with broadcasting.
      #
      # `.broadcast_plus` is an alias for `.broadcast_add`.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_plus(x, y) # => [[1, 1, 1], [2, 2, 2]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_plus)

      # Returns result of first array elements raised to powers from
      # second array, element-wise with broadcasting.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[2, 2, 2], [2, 2, 2]] # x
      #     [[1], [2]]             # y
      #
      # Then:
      #     broadcast_power(x, y) # => [[2, 2, 2], [4, 4, 4]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The base input.
      # * *rhs* (`{{type}}`, required)
      #   The exponent input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_power)

      # Returns element-wise difference of the input arrays with broadcasting.
      #
      # `.broadcast_sub` is an alias to the function `.broadcast_minus`.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 1, 1], [1, 1, 1]] # x
      #     [[0], [1]]             # y
      #
      # Then:
      #     broadcast_sub(x, y) # => [[1, 1, 1], [0, 0, 0]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_sub)

      # Broadcasts the input array to a new shape.
      #
      # Broadcasting is a mechanism that allows `NDArray` to perform
      # arithmetic operations with other arrays of different shapes
      # efficiently without creating multiple copies of arrays. See:
      # [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
      # for explanation.
      #
      # Broadcasting is allowed on axes with size 1, such as from `[2, 1, 3, 1]`
      # to `[2, 8, 3, 9]`. Elements will be duplicated on the broadcasted
      # axes.
      #
      # Assume *x* is an array with the following elements:
      #     [[1, 2, 3]]
      #
      # Then:
      #     broadcast_to(x, shape: [2, 3]) = [[1, 2, 3], [1, 2, 3]])
      #
      # The dimension which you do not want to change can also be
      # specified as `0`. So with `shape: [2, 0]`, we will obtain the
      # same result as in the above example.
      #
      # ### Parameters
      {{prefix}}
      # * *shape* (`Int` or `Array(Int)`, required)
      #   The shape of the desired array.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, broadcast_to)

      # Returns element-wise cube-root value of the input.
      #
      # Assume *x* is an array with the following elements:
      #     [1, 8, -125]
      #
      # Then:
      #     cbrt(x) = [1, 2, -5]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, cbrt)

      # Returns element-wise ceiling of the input.
      #
      # The ceiling  `x` is the smallest integer `i`, such that `i >= x`.
      #
      # Assume *x* is an array with the following elements:
      #     [-2.1, -1.9, 1.5, 1.9, 2.1]
      #
      # Then:
      #     ceil(x) = [-2.0, -1.9, 2.0, 2.0, 3.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, ceil)

      # Clips (limits) the values in an array.
      #
      # Given an interval, values outside the interval are clipped to
      # the interval edges. Clipping *x* between *a_min* and *a_x*
      # would be:
      #
      #     clip(x, a_min, a_max) = max(min(x, a_max), a_min))
      #
      # Assume *x* is an array with the following elements:
      #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      #
      # Then:
      #     clip(x, 1, 8) # => [1, 1, 2, 3, 4, 5, 6, 7, 8, 8]
      #
      # ### Parameters
      {{prefix}}
      # * *a_min* (`Float`, required)
      #   Minimum value.
      # * *a_max* (`Float`, required)
      #   Maximum value.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, clip)

      # Joins input arrays along a given axis.
      #
      # The dimensions of the input arrays should be the same except
      # for the axis along which they will be concatenated. The
      # dimension of the output array along the concatenated axis will
      # be equal to the sum of the corresponding dimensions of the
      # input arrays.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 2], [3, 4]] # x
      #     [[1, 4], [1, 1]] # y
      #
      # Then:
      #     concat(x, y) # => [[1, 2, 1, 4], [3, 4, 1, 1]]
      #
      # ### Parameters
      # * *data* (`Array({{type}})`, required)
      #   List of arrays to concatenate.
      # * *dim* (`Int`, default = 1)
      #   The dimension to be concated.
      {{suffix}}
      #
      def self.concat(data : Array(self), **kwargs)
        Ops._concat(data, **kwargs.merge({num_args: data.size}))
      end

      # Compute *N*-D convolution on *(N+2)*-D input.
      #
      # For general 2-D convolution, the shapes are:
      #   * **data**: *[batch_size, channel, height, width]*
      #   * **weight**: *[num_filter, channel, kernel[0], kernel[1]]*
      #   * **bias**: *[num_filter]*
      #   * **out**: *[batch_size, num_filter, out_height, out_width]*
      #
      # If *no_bias* is set to be true, then the *bias* term is
      # ignored.
      #
      # The default data *layout* is *NCHW*, namely *(batch_size,
      # channel, height, width)*. We can choose other layouts such as
      # *NWC*.
      #
      # If *num_group* is larger than 1, denoted by *g*, then split
      # the input data evenly into *g* parts along the channel axis,
      # and also evenly split *weight* along the first dimension. Next
      # compute the convolution on the *i*-th part of the data with
      # the *i*-th weight part. The output is obtained by
      # concatenating all the *g* results.
      #
      # 1-D convolution does not have *height* dimension but only
      # *width* in space.  The shapes are:
      #   * **data**: *[batch_size, channel, width]*
      #   * **weight**: *[num_filter, channel, kernel[0]]*
      #   * **bias**: *[num_filter]*
      #   * **out**: *[batch_size, num_filter, out_width]*
      #
      # 3-D convolution adds an additional *depth* dimension besides
      # *height* and *width*. The shapes are:
      #   * **data**: *[batch_size, channel, depth, height, width]*
      #   * **weight**: *[num_filter, channel, kernel[0], kernel[1], kernel[2]]*
      #   * **bias**: *[num_filter]*
      #   * **out**: *[batch_size, num_filter, out_depth, out_height, out_width]*
      #
      # Both *weight* and *bias* are learnable parameters.
      #
      # There are other options to tune the performance:
      #   * **cudnn_tune**: enabling this option leads to higher
      #   startup time but may give faster speed. Options are: "off" -
      #   no tuning, "limited_workspace" - run test and pick the
      #   fastest algorithm that doesn't exceed workspace limit,
      #   "fastest" - pick the fastest algorithm and ignore workspace
      #   limit, `nil` (default) - the behavior is determined by the
      #   environment variable "MXNET_CUDNN_AUTOTUNE_DEFAULT" -- 0 for
      #   off, 1 for limited workspace (default), 2 for fastest.
      #   * **workspace**: a larger number leads to more (GPU) memory
      #   usage but may improve the performance.
      #
      # ### Parameters
      # * *data* (`{{type}}`, required)
      #   Input data.
      # * *weight* (`{{type}}`, required)
      #   Weight matrix.
      # * *bias* (`{{type}}`, required)
      #   Bias parameter.
      # * *kernel* (`Array(Int)`, shape, required)
      #   Convolution kernel size: `[w]`, `[h, w]` or `[d, h, w]`.
      # * *stride* (`Array(Int)`, shape, optional, default = [])
      #   Convolution stride: `[w]`, `[h, w]` or `[d, h, w]`. Defaults
      #   to 1 for each dimension.
      # * *dilate* (`Array(Int)`, shape, optional, default = [])
      #   Convolution dilation: `[w]`, `[h, w]` or `[d, h, w]`.
      #   Defaults to 1 for each dimension.
      # * *pad* (`Array(Int)`, shape, optional, default = [])
      #   Zero pad for convolution: `[w]`, `[h, w]` or `[d, h, w]`.
      #   Defaults to no padding.
      # * *num_filter* (`Int::Unsigned`, required)
      #   Convolution filter (channel) number.
      # * *num_group* (`Int::Unsigned`, optional, default = 1)
      #   Number of group partitions.
      # * *workspace* (`Int::Unsigned`, optional, default = 1024)
      #   Maximum temporary workspace allowed (MB) for convolution.
      #   This parameter has two usages. When CUDNN is not used, it
      #   determines the effective batch size of the convolution
      #   kernel. When CUDNN is used, it controls the maximum
      #   temporary storage used for tuning the best CUDNN kernel
      #   when "limited_workspace" strategy is used.
      # * *no_bias* (`Bool`, optional, default = false)
      #   Whether to disable bias parameter.
      # * *cudnn_tune* (`::Symbol`, `:fastest`, `:limited_workspace`, `:off` or `nil`, optional)
      #   Whether to pick the convolution algorithm by running a
      #   performance test.
      # * *cudnn_off* (`Bool`, optional, default = false)
      #   Turn off cudnn for this layer.
      # * *layout* (`String`, `"NCDHW"`, `"NCHW"`, `"NCW"`, `"NDHWC"`, `"NHWC"`, `"NWC"` or `nil`, optional)
      #   Set layout for input, output and weight. Empty for default
      #   layout: "NCW" for 1D, "NCHW" for 2D and "NCDHW" for
      #   3D. "NHWC" and "NDHWC" are only supported on GPU.
      {{suffix}}
      #
      def self.convolution(data : self, weight : self?, bias : self?, kernel, num_filter, **kwargs)
        Ops._Convolution(data, weight, bias, **kwargs.merge({kernel: kernel, num_filter: num_filter}))
      end

      # Computes the element-wise cosine of the input array.
      #
      # The input should be in radians (`2\𝜋` radians equals 360 degrees).
      #
      #     cos([0, 𝜋/4, 𝜋/2]) = [1, 0.707, 0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, cos)

      # Returns the hyperbolic cosine of the input array, computed element-wise.
      #
      #     cosh(x) = (exp(x) + exp(-x)) / 2
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, cosh)

      # Extracts a diagonal or constructs a diagonal array.
      #
      # `.diag`‘s behavior depends on the input array dimensions:
      #   * *1-D* arrays: constructs a 2-D array with the input as its
      #   diagonal, all other elements are zero.
      #   * *N-D* arrays: extracts the diagonals of the sub-arrays
      #   with axes specified by *axis1* and *axis2*. The output shape
      #   is decided by removing the axes numbered *axis1* and *axis2*
      #   from the input shape and appending to the result a new axis
      #   with the size of the diagonals in question.
      #
      # For example, when the input shape is `[2, 3, 4, 5]`, *axis1*
      # and *axis2* are 0 and 2 respectively and *k* is 0, the
      # resulting shape is `[3, 5, 2]`.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 2, 3], [4, 5, 6]]               # x
      #     [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] # y
      #
      # Then:
      #     diag(x) = [1, 5]
      #     diag(x, k: 1) = [2, 6]
      #     diag(x, k: -1) = [4]
      #
      #     diag(y) = [[1, 7], [2, 8]]
      #     diag(y, k: 1) = [[3], [4]]
      #     diag(y, axis1: -2, axis2: -1) = [[1, 4], [5, 8]]
      #
      # ### Parameters
      {{prefix}}
      # * *k* (`Int`, optional, default = 0)
      #   The diagonal in question. The default is 0. Use `k > 0` for
      #   diagonals above the main diagonal, and `k < 0` for diagonals
      #   below the main diagonal.
      # * *axis1* (`Int`, optional, default = 0)
      #   The first axis of the sub-arrays of interest. Ignored when
      #   the input is a 1-D array.
      # * *axis2* (`Int`, optional, default = 1)
      #   The second axis of the sub-arrays of interest. Ignored when
      #   the input is a 1-D array.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, diag)

      # Computes the dot product of two arrays.
      #
      # `.dot`‘s behavior depends on the input array dimensions:
      #   * *1-D* arrays: inner product of vectors
      #   * *2-D* arrays: matrix multiplication
      #   * *N-D* arrays: a sum product over the last axis of the first
      #   input and the first axis of the second input
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[1, 2], [3, 4]] # x
      #     [[4, 3], [1, 1]] # y
      #
      # Then:
      #     dot(x, y) # => [[8, 5], [20, 13]]
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      # * *transpose_a* (`Bool`, default = false)
      #   If true then transpose the first input before dot.
      # * *transpose_b* (`Bool`, default = false)
      #   If true then transpose the second input before dot.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, dot)

      # Returns element-wise exponential value of the input.
      #
      # Assume *x* is an array with the following elements:
      #     [0.0, 1.0, 2.0]
      #
      # Then:
      #     exp(x) = [1.0, 2.71828175, 7.38905621]
      #
      # The storage type of `.exp` output is always dense.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, exp)

      # Returns `exp(x) - 1` computed element-wise on the input.
      #
      # This function provides greater precision than explicitly
      # calculating `exp(x) - 1` for small values of *x*.
      #
      # Assume *x* is an array with the following elements:
      #     [0.0, 1.0, 2.0]
      #
      # Then:
      #     expm1(x) = [0.0, 1.71828182, 6.38905609]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, expm1)

      # Inserts a new axis of size 1 into the array shape.
      #
      # For example, given *x* with shape *[2, 3, 4]*, then
      # `expand_dims(x, axis: 1)` will return a new array with shape
      # *[2, 1, 3, 4]*.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, required)
      #   Position where new axis is to be inserted. Suppose that the
      #   input array‘s dimension is `ndim`, the range of the inserted
      #   axis is `[-ndim, ndim]`.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, expand_dims)

      # Returns element-wise rounded value to the nearest integer
      # towards zero.
      #
      # Assume *x* is an array with the following elements:
      #     [-2.1, -1.9, 1.5, 1.9, 2.1]
      #
      # Then:
      #     fix(x) = [-2.0, -1.0, 1.0, 1.0, 2.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, fix)

      # Flattens the input array into a 2-D array by collapsing the
      # higher dimensions.
      #
      # For an input array with shape *(d1, d2, ..., dk)*, `.flatten`
      # reshapes the input array into an output array of shape
      # _(d1, d2 * ... * dk)_.
      #
      # Note that the bahavior of this function is different from
      # `Array#flatten`, which behaves similar to `.reshape(shape: [-1])`.
      #
      # Assume *x* is an array with the following elements:
      #     [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]
      #
      # Then:
      #     flatten(x).shape # => [2, 6]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, flatten)

      # Reverses the order of elements along given axis while preserving array shape.
      #
      # Assume *x* is an array with the following elements:
      #     [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
      #
      # Then:
      #     flip(x, axis: 0) # => [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
      #     flip(x, axis: 1) # => [[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, required)
      #   The axis on which to reverse elements.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, flip)

      # Returns the element-wise floor of the input.
      #
      # The floor of `x` is the largest integer `i`, such that `i <= x`.
      #
      # Assume *x* is an array with the following elements:
      #     [-2.1, -1.9, 1.5, 1.9, 2.1]
      #
      # Then:
      #     floor(x) = [-3.0, -2.0, 1.0, 1.0, 2.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, floor)

      # Applies a linear transformation: _Y = XWᵀ + b_.
      #
      # If *flatten* is true, then the shapes are:
      #   * **data**: *[batch_size, x1, x2, ..., xn]*
      #   * **weight**: *[num_hidden, x1 * x2 * ... * xn]*
      #   * **bias**: *[num_hidden]*
      #   * **out**: *[batch_size, num_hidden]*
      #
      # If *flatten* is false, then the shapes are:
      #   * **data**: *[x1, x2, ..., xn, input_dim]*
      #   * **weight**: *[num_hidden, input_dim]*
      #   * **bias**: *[num_hidden]*
      #   * **out**: *[x1, x2, ..., xn, num_hidden]*
      #
      # The learnable parameters include both *weight* and *bias*.
      #
      # If *no_bias* is true, then the *bias* term is ignored.
      #
      # ### Parameters
      # * *data* (`{{type}}`, required)
      #   Input data.
      # * *weight* (`{{type}}`, required)
      #   Weight matrix.
      # * *bias* (`{{type}}`, required)
      #   Bias parameter.
      # * *num_hidden* (`Int`, required)
      #   Number of hidden nodes of the output.
      # * *no_bias* (`Bool`, optional, default = false)
      #   Whether to disable bias parameter.
      # * *flatten* (`Bool`, optional, default = true)
      #   Whether to collapse all but the first axis of the input data
      #   tensor.
      {{suffix}}
      #
      def self.fully_connected(data : self, weight : self?, bias : self?, num_hidden : Int, **kwargs)
        Ops._FullyConnected(data, weight, bias, **kwargs.merge({num_hidden: num_hidden}))
      end

      # Returns element-wise natural logarithmic value of the input.
      #
      # The natural logarithm is the logarithm in base *e*, so that
      # `log(exp(x)) = x`.
      #
      # The storage type of `.log` output is always dense.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, log)

      # Returns `log(1 + x)` computed element-wise on the input.
      #
      # This function is more accurate than explicitly calculating
      # `log(1 + x)` for small *x*.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, log1p)

      # Returns element-wise base-10 logarithmic value of the input.
      #
      #     10**log10(x) = x
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, log10)

      # Returns element-wise base-2 logarithmic value of the input.
      #
      #     2**log2(x) = x
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, log2)

      # Computes the log softmax of the input.
      #
      # This is equivalent to computing `.softmax` followed by `.log`.
      #
      # Assume *x* is an array with the following elements:
      #     [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
      #
      # Then:
      #     softmax(x, axis: 0) # => [[-0.6931, -0.6931, -0.6931], [-0.6931, -0.6931, -0.6931]]
      #     softmax(x, axis: 1) # => [[-1.0986, -1.0986, -1.0986], [-1.0986, -1.0986, -1.0986]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, optional, default = -1)
      #   The axis along which to compute softmax.
      # * *temperature* (`Float`, optional, default = 1.0)
      #   Temperature parameter in softmax.
      # * *dtype* (`::Symbol`, `:float16`, `:float32` or `:float64`, optional)
      #   Type of the output in case this can't be inferred. Defaults
      #   to the same type as the input if not defined.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, log_softmax)

      # Computes the max of array elements over given axes.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the reduction.
      #   By default it computes over all elements into a scalar array
      #   with shape `[1]`. If axis is `Int`, a reduction is performed
      #   on a particular axis. If axis is `Array(Int)`, a reduction is
      #   performed on all the axes specified in the list. If *exclude*
      #   is `true`, reduction will be performed on the axes that are
      #   **not** in axis instead. Negative values means indexing from
      #   right to left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If `true`, the reduced axes are left in the result as
      #   a dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axes that are not in *axis*
      #   instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, max)

      # Computes the mean of array elements over given axes.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the reduction.
      #   By default it computes over all elements into a scalar array
      #   with shape `[1]`. If axis is `Int`, a reduction is performed
      #   on a particular axis. If axis is `Array(Int)`, a reduction is
      #   performed on all the axes specified in the list. If *exclude*
      #   is `true`, reduction will be performed on the axes that are
      #   **not** in axis instead. Negative values means indexing from
      #   right to left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If `true`, the reduced axes are left in the result as
      #   a dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axes that are not in *axis*
      #   instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, mean)

      # Computes the min of array elements over given axes.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the reduction.
      #   By default it computes over all elements into a scalar array
      #   with shape `[1]`. If axis is `Int`, a reduction is performed
      #   on a particular axis. If axis is `Array(Int)`, a reduction is
      #   performed on all the axes specified in the list. If *exclude*
      #   is `true`, reduction will be performed on the axes that are
      #   **not** in axis instead. Negative values means indexing from
      #   right to left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If `true`, the reduced axes are left in the result as
      #   a dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axes that are not in *axis*
      #   instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, min)

      # Computes the product of array elements over given axes
      # treating not-a-number values (*NaN*) as one.
      #
      # See `.prod`.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the
      #   reduction. `axis: []` or `axis: nil` will compute over all
      #   elements into a scalar array with shape `[1]`. If *axis* is
      #   an `Int`, a reduction is performed on a particular axis. If
      #   *axis* is an array of `Int`, a reduction is performed on all
      #   the axes specified in the array. If *exclude* is true,
      #   reduction will be performed on the axes that are **not** in
      #   *axis* instead. Negative values means indexing from right to
      #   left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If this is set to true, the reduced axes are left in the
      #   result as dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axis that are **not** in
      #   axis instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, nanprod)

      # Computes the sum of array elements over given axes treating
      # not-a-number values (*NaN*) as zero.
      #
      # See `.sum`.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the
      #   reduction. `axis: []` or `axis: nil` will compute over all
      #   elements into a scalar array with shape `[1]`. If *axis* is
      #   an `Int`, a reduction is performed on a particular axis. If
      #   *axis* is an array of `Int`, a reduction is performed on all
      #   the axes specified in the array. If *exclude* is true,
      #   reduction will be performed on the axes that are **not** in
      #   *axis* instead. Negative values means indexing from right to
      #   left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If this is set to true, the reduced axes are left in the
      #   result as dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axis that are **not** in
      #   axis instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, nansum)

      # Computes the norm.
      #
      # This operator computes the norm on an array with the specified
      # axis, depending on the value of the `ord` parameter. By default,
      # it computes the L2 norm on the entire array. Currently only
      # `ord: 2` supports sparse arrays.
      #
      # Assume *x* is an array with the following elements:
      #     [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 2.0], [5.0, 6.0]]]
      #
      # Then:
      #     norm(x, ord: 2, axis: 1) # => [[3.1622, 4.4721], [5.3851, 6.3245]]
      #     norm(x, ord: 1, axis: 1) # => [[40., 6.0], [7.0, 8.0]]
      #
      # ### Parameters
      {{prefix}}
      # * *ord* (`Int`, optional, default = `2`)
      #   Order of the norm. Currently `ord: 1` and `ord: 2` are
      #   supported.
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the reduction.
      #   By default it computes over all elements into a scalar array
      #   with shape `[1]`. If axis is `Int`, a reduction is performed
      #   on a particular axis. If axis is `Array(Int)`, it specifies
      #   the axes that hold 2-D matrices, and the matrix norms of
      #   these matrices are computed.
      # * *out_dtype* (`::Symbol`, `:float16`, `:float32`, `:float64`, `:int32`, `:int64` or `:int8`, optional)
      #   The data type of the output.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If `true`, the reduced axes are left in the result as
      #   a dimension with size one.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, norm)

      # Returns a one-hot array.
      #
      # The locations represented by *indices* take value *on_value*,
      # while all other locations take value *off_value*.
      #
      # `.one_hot` with *indices* of shape `[i0, i1]` and depth of `d`
      # would result in an output array of shape `[i0, i1, d]` with:
      #     output[i, j, 0..-1] = off_value
      #     output[i, j, indices[i, j]] = on_value
      #
      # Assume *x* is an array with the following elements:
      #     [1, 0, 2, 0]
      #
      # Then:
      #     one_hot(x, 3) # => [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
      #
      # ### Parameters
      # * *indices* (`{{type}}`, required)
      #   Array of locations where to set *on_value*.
      # * *depth* (`Int`, required)
      #   Depth of the one hot dimension.
      # * *on_value* (`Float`, optional, default = 1.0)
      #   The value assigned to the locations represented by indices.
      # * *off_value* (`Float`, optional, default = 0.0)
      #   The value assigned to the locations not represented by indices.
      # * *dtype* (`::Symbol`, optional, default = `:float32`)
      #   Type of the output.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, one_hot)

      # Returns an array filled with all ones, with the given shape.
      #
      # ### Parameters
      {{prefix}}
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the array.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output array.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.ones(shape : Int | Array(Int), ctx = Context.current, **kwargs)
        Internal._ones(**kwargs.merge({shape: shape, ctx: ctx}))
      end

      # Returns an array of ones with the same shape, data type and
      # storage type as the input array.
      #
      # Assume *x* is an array with the following elements:
      #     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      #
      # Then:
      #     ones_like(x) # => [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, ones_like)

      # Picks elements from an input array according to the indices
      # along the given axis.
      #
      # Given an input array of shape `[d0, d1]` and indices of shape
      # `[i0]`, the result will be an output array of shape `[i0]`
      # with:
      #     output[i] = input[i, indices[i]]
      #
      # By default, if any index mentioned is too large, it is
      # replaced by the index that addresses the last element along an
      # axis (clip mode).
      #
      # This function supports n-dimensional input and
      # (n-1)-dimensional indices arrays.
      #
      # Assume *x*, *i*, *j*, and *k* are arrays with the following
      # elements:
      #     [[1, 2], [3, 4], [5, 6]] # x
      #     [0, 1]                   # i
      #     [0, 1, 0]                # j
      #     [1, 0, 2]                # k
      #
      # Then:
      #     # pick elements with specified indices along axis 0
      #     pick(x, index: i, 0) # => [1, 4]
      #     # pick elements with specified indices along axis 1
      #     pick(x, index: j, 1) # => [1, 4, 5]
      #     # pick elements with specified indices along axis 1 --
      #     # dims are maintained
      #     pick(x, index: k, 1, keepdims: true) # => [[2], [3], [6]]
      #
      # ### Parameters
      # * *data* (`{{type}}`, required)
      #   The input array.
      # * *index* (`{{type}}`, required)
      #   The index array.
      # * *axis* (`Int` or `nil`, optional, default = -1)
      #   The axis to pick the elements. Negative values mean
      #   indexing from right to left. If `nil`, elements in the index
      #   with respect to the flattened input will be picked.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If true, the axis where we pick the elements is left in the
      #   result as a dimension with size one.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, pick)

      # Computes the product of array elements over given axes.
      #
      # Assume *x* is an array with the following elements:
      #     [[[1, 2], [2, 3], [1, 3]],
      #      [[1, 4], [4, 3], [5, 2]],
      #      [[7, 1], [7, 2], [7, 3]]]
      #
      # Then:
      #     prod(x, axis: 1) # => [[2, 18], [20, 24], [343, 6]]
      #     prod(x, axis: [1, 2]) # => [36, 480, 2058]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the
      #   reduction. `axis: []` or `axis: nil` will compute over all
      #   elements into a scalar array with shape `[1]`. If *axis* is
      #   an `Int`, a reduction is performed on a particular axis. If
      #   *axis* is an array of `Int`, a reduction is performed on all
      #   the axes specified in the array. If *exclude* is true,
      #   reduction will be performed on the axes that are **not** in
      #   *axis* instead. Negative values means indexing from right to
      #   left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If this is set to true, the reduced axes are left in the
      #   result as dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axis that are **not** in
      #   axis instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, prod)

      # Performs pooling on the input.
      #
      # The shapes for 1-D pooling are:
      #   * **data** and **out**:
      #   *[batch_size, channel, width]* ("NCW" layout) or
      #   *[batch_size, width, channel]* ("NWC" layout)
      #
      # The shapes for 2-D pooling are:
      #   * **data** and **out**:
      #   *[batch_size, channel, height, width]* ("NCHW" layout) or
      #   *[batch_size, height, width, channel]* ("NHWC" layout)
      #
      # Three pooling options are supported by *pool_type*:
      #   * **avg**: average pooling
      #   * **max**: max pooling
      #   * **sum**: sum pooling
      #   * **lp**: Lp pooling
      #
      # For 3-D pooling, an additional *depth* dimension is added
      # before *height*. Namely the input data and output will have
      # shape:
      #   *[batch_size, channel, depth, height, width]* ("NCDHW" layout) or
      #   *[batch_size, depth, height, width, channel]* ("NDHWC" layout).
      #
      # Notes on Lp pooling:
      #
      # Lp pooling was first introduced by this paper:
      # https://arxiv.org/pdf/1204.3968.pdf. L-1 pooling is simply
      # sum pooling, while L-inf pooling is simply max pooling. We can
      # see that Lp pooling stands between those two, in practice the
      # most common value for *p* is 2.
      #
      # ### Parameters
      # * *data* (`{{type}}`, required)
      #   Input data.
      # * *kernel* (`Array(Int)`, shape, optional, default = [])
      #   Pooling kernel size: *[y, x]* or *[d, y, x]*.
      # * *pool_type* (`::Symbol`, `:avg`, `:lp`, `:max` or `:sum`, optional, default = `:max`)
      #   Pooling type to be applied.
      # * *global_pool* (`Bool`, optional, default = false)
      #   Ignore kernel size; do global pooling based on current input
      #   feature map.
      # * *cudnn_off* (`Bool`, optional, default = false)
      #   Turn off cudnn pooling and use MXNet pooling operator.
      # * *pooling_convention* (`::Symbol`, `:full`, `:same`, or `:valid`, optional, default = `:valid`)
      #   Pooling convention to be applied.
      # * *stride* (`Array(Int)`, shape, optional, default = [])
      #   Stride for pooling: *[y, x]* or *[d, y, x]*. Defaults to 1
      #   for each dimension.
      # * *pad* (`Array(Int)`, shape, optional, default = [])
      #   Pad for pooling: *[y, x]* or *[d, y, x]*. Defaults to no
      #   padding.
      # * *p_value* (`Int`, optional)
      #   Value of *p* for Lp pooling, can be 1 or 2, required for Lp
      #   pooling.
      # * *count_include_pad* (`Bool`, optional)
      #   Only used for average pooling. Specify whether to count
      #   padding elements for average calculation. For example, with
      #   a 5*5 kernel on a 3*3 corner of a image, the sum of the 9
      #   valid elements will be divided by 25 if this is set to
      #   true, or it will be divided by 9 if this is set to
      #   false. Defaults to true.
      # * *layout* (`String`, `"NCDHW"`, `"NCHW"`, `"NCW"`, `"NDHWC"`, `"NHWC"`, `"NWC"` or `nil`, optional)
      #   Set layout for input, output and weight. Empty for default
      #   layout: "NCW" for 1D, "NCHW" for 2D and "NCDHW" for
      #   3D. "NHWC" and "NDHWC" are only supported on GPU.
      {{suffix}}
      #
      def self.pooling(data : self, **kwargs)
        Ops._Pooling(data, **kwargs)
      end

      # Converts each element of the input array from degrees to
      # radians.
      #
      #     radians([0, 90, 180, 270, 360]) = [0, 𝜋/2, 𝜋, 3𝜋/2, 2𝜋]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, radians)

      # Returns element-wise inverse cube-root value of the input.
      #
      #     rcbrt(x) = 1/cbrt(x)
      #
      # Assume *x* is an array with the following elements:
      #     [1, 8, -125]
      #
      # Then:
      #     rcbrt(x) = [1.0, 0.5, -0.2]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, rcbrt)

      # Returns the reciprocal of the argument, element-wise.
      #
      #     reciprocal(x) = 1/x
      #
      # Assume *x* is an array with the following elements:
      #     [-2, 1, 3, 1.6, 0.2]
      #
      # Then:
      #     reciprocal(x) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, reciprocal)

      # Computes the rectified linear activation.
      #
      # _y=max(input,0)_
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, relu)

      # Draws random samples from a uniform distribution.
      #
      # Samples are uniformly distributed over the half-open interval
      # `[low, high)` (includes low, but excludes high).
      #
      #     random_uniform(0.0, 1.0, shape: [2, 2]) # => [[0.60276335, 0.85794562], [0.54488319, 0.84725171]]
      #
      # ### Parameters
      # * *low* (`Float`, default = 0.0)
      #   Lower bound of the distribution.
      # * *high* (`Float`, default = 1.0)
      #   Upper bound of the distribution.
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the output.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.random_uniform(low : Number = 0.0, high : Number = 1.0, ctx : Context = Context.current, **kwargs)
        Internal._random_uniform(**kwargs.merge({low: low, high: high, ctx: ctx}))
      end

      # Draws random samples from a normal (Gaussian) distribution.
      #
      # Samples are distributed according to a normal distribution
      # parametrized by `loc` (mean) and `scale` (standard deviation).
      #
      #     random_normal(0.0, 1.0, shape: [2, 2]) # => [[1.89171135, -1.16881478], [-1.23474145, 1.55807114]]
      #
      # ### Parameters
      # * *loc* (`Float`, default = 0.0)
      #   Mean of the distribution.
      # * *scale* (`Float`, default = 1.0)
      #   Standard deviation of the distribution.
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the output.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.random_normal(loc : Number = 0.0, scale : Number = 1.0, ctx : Context = Context.current, **kwargs)
        Internal._random_normal(**kwargs.merge({loc: loc, scale: scale, ctx: ctx}))
      end

      # Draws random samples from a Poisson distribution.
      #
      # Samples are distributed according to a Poisson distribution
      # parametrized by `lam` (rate). Samples will always be returned
      # as a floating point data type.
      #
      #     random_poisson(4.0, shape: [2, 2]) # => [[5.0, 2.0], [4.0, 6.0]]
      #
      # ### Parameters
      # * *lam* (`Float`, default = 1.0)
      #   Lambda parameter (rate) of the Poisson distribution.
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the output.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.random_poisson(lam : Number = 1.0, ctx : Context = Context.current, **kwargs)
        Internal._random_poisson(**kwargs.merge({lam: lam, ctx: ctx}))
      end

      # Draws random samples from an exponential distribution.
      #
      # Samples are distributed according to an exponential distribution
      # parametrized by `lam` (rate).
      #
      #     random_exponential(4.0, shape: [2, 2]) # => [[0.0097189 , 0.08999364], [0.04146638, 0.31715935]]
      #
      # ### Parameters
      # * *lam* (`Float`, default = 1.0)
      #   Lambda parameter (rate) of the exponential distribution.
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the output.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.random_exponential(lam : Number = 1.0, ctx : Context = Context.current, **kwargs)
        Internal._random_exponential(**kwargs.merge({lam: lam, ctx: ctx}))
      end

      # Draws random samples from a gamma distribution.
      #
      # Samples are distributed according to a gamma distribution
      # parametrized by `alpha` (shape) and `beta` (scale).
      #
      #     random_gamma(9.0, 0.5, shape: [2, 2]) # => [[6.2806954, 6.1658335], [4.5625057, 6.479337]]
      #
      # ### Parameters
      # * *alpha* (`Float`, default = 1.0)
      #   Alpha parameter (shape) of the gamma distribution.
      # * *beta* (`Float`, default = 1.0)
      #   Beta parameter (scale) of the gamma distribution.
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the output.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.random_gamma(alpha : Number = 1.0, beta : Number = 1.0, ctx : Context = Context.current, **kwargs)
        Internal._random_gamma(**kwargs.merge({alpha: alpha, beta: beta, ctx: ctx}))
      end

      {% unless compare_versions(MXNet::Internal::MXNET_VERSION, "1.4.0") < 0 %}
        # Draws random samples from a discrete uniform distribution.
        #
        # Samples are uniformly distributed over the half-open interval
        # `[low, high)` (includes low, but excludes high).
        #
        #     random_randint(0, 5, shape: [2, 2]) # => [[0, 2], [3, 1]]
        #
        # ### Parameters
        # * *low* (`Int`, required)
        #   Lower boundary of the output interval.
        # * *high* (`Int`, required)
        #   Upper boundary of the output interval.
        # * *shape* (`Int` or `Array(Int)`)
        #   The shape of the output.
        # * *dtype* (`::Symbol`, default = `:int32`)
        #   The data type of the output.
        # * *ctx* (`Context`, optional)
        #   Device context (default is the current context). Only used
        #   for imperative calls.
        {{suffix}}
        #
        def self.random_randint(low : Int, high : Int, ctx : Context = Context.current, **kwargs)
          Internal._random_randint(**kwargs.merge({low: low, high: high, ctx: ctx}))
        end
      {% end %}

      # Draws concurrent samples from uniform distributions.
      #
      # Samples are drawn from multiple uniform distributions on the
      # intervals given by `[low, high)`.
      #
      # The parameters of the distributions are provided as input
      # arrays. Let `[s]` be the shape of the input arrays, `n` be the
      # dimension of `[s]`, `[t]` be the shape specified as the
      # parameter of the operator, and `m` be the dimension of `[t]`.
      # Then the output will be a (`n+m`)-dimensional array with shape
      # `[s]x[t]`.
      #
      # For any valid `n`-dimensional index `i` with respect to the
      # input arrays, `output[i]` will be an `m`-dimensional array
      # that holds randomly drawn samples from the distribution which
      # is parameterized by the input values at index `i`. If the
      # shape parameter of the operator is not set, then one sample
      # will be drawn per distribution and the output array has the
      # same shape as the input arrays.
      #
      # Assume *low* and *high* are arrays with the following elements:
      #     [0.0, 2.5] # low
      #     [1.0, 3.7] # high
      #
      # Then:
      #     sample_uniform(low, high)             # => [0.40451524, 3.18687344]
      #     sample_uniform(low, high, shape: [2]) # => [[0.40451524, 0.18017688], [3.18687344, 3.68352246]]
      #
      # ### Parameters
      # * *low* (`{{type}}`)
      #   Lower bounds of the distributions.
      # * *high* (`{{type}}`)
      #   Upper bounds of the distributions.
      # * *shape* (`Int` or `Array(Int)`)
      #   Shape to be sampled from each random distribution.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      {{suffix}}
      #
      def self.sample_uniform(low : self, high : self, shape = [] of Int32, **kwargs)
        Internal._sample_uniform(**kwargs.merge({low: low, high: high, shape: shape}))
      end

      # Draws concurrent samples from normal (Gaussian) distributions.
      #
      # Samples are drawn from multiple normal distributions with
      # parameters `mu` (mean) and `sigma` (standard deviation).
      #
      # The parameters of the distributions are provided as input
      # arrays. Let `[s]` be the shape of the input arrays, `n` be the
      # dimension of `[s]`, `[t]` be the shape specified as the
      # parameter of the operator, and `m` be the dimension of `[t]`.
      # Then the output will be a (`n+m`)-dimensional array with shape
      # `[s]x[t]`.
      #
      # For any valid `n`-dimensional index `i` with respect to the
      # input arrays, `output[i]` will be an `m`-dimensional array
      # that holds randomly drawn samples from the distribution which
      # is parameterized by the input values at index `i`. If the
      # shape parameter of the operator is not set, then one sample
      # will be drawn per distribution and the output array has the
      # same shape as the input arrays.
      #
      # Assume *mu* and *sigma* are arrays with the following elements:
      #     [0.0, 2.5] # mu
      #     [1.0, 3.7] # sigma
      #
      # Then:
      #     sample_normal(mu, sigma)             # => [-0.56410581, 0.95934606]
      #     sample_normal(mu, sigma, shape: [2]) # => [[-0.56410581, 0.2928229 ], [0.95934606, 4.48287058]]
      #
      # ### Parameters
      # * *mu* (`{{type}}`)
      #   Means of the distributions.
      # * *sigma* (`{{type}}`)
      #   Standard deviations of the distributions.
      # * *shape* (`Int` or `Array(Int)`)
      #   Shape to be sampled from each random distribution.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      {{suffix}}
      #
      def self.sample_normal(mu : self, sigma : self, shape = [] of Int32, **kwargs)
        Internal._sample_normal(**kwargs.merge({mu: mu, sigma: sigma, shape: shape}))
      end

      # Draws concurrent samples from Poisson distributions.
      #
      # Samples are drawn from multiple Poisson distributions with
      # parameters `lam` (rate). Samples will always be returned as
      # a floating point data type.
      #
      # The parameters of the distributions are provided as an input
      # array. Let `[s]` be the shape of the input array, `n` be the
      # dimension of `[s]`, `[t]` be the shape specified as the
      # parameter of the operator, and `m` be the dimension of `[t]`.
      # Then the output will be a (`n+m`)-dimensional array with shape
      # `[s]x[t]`.
      #
      # For any valid `n`-dimensional index `i` with respect to the
      # input array, output[i] will be an `m`-dimensional array that
      # holds randomly drawn samples from the distribution which is
      # parameterized by the input value at index `i`. If the shape
      # parameter of the operator is not set, then one sample will be
      # drawn per distribution and the output array has the same shape
      # as the input array.
      #
      # Assume *lam* is an array with the following elements:
      #     [1.0, 8.5]
      #
      # Then:
      #     sample_poisson(lam)             # => [0.0, 13.0]
      #     sample_poisson(lam, shape: [2]) # => [[0.0, 4.0], [13.0, 8.0]]
      #
      # ### Parameters
      # * *lam* (`{{type}}`)
      #   Lambda parameters (rates) of the Poisson distributions.
      # * *shape* (`Int` or `Array(Int)`)
      #   Shape to be sampled from each random distribution.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      {{suffix}}
      #
      def self.sample_poisson(lam : self, shape = [] of Int32, **kwargs)
        Internal._sample_poisson(**kwargs.merge({lam: lam, shape: shape}))
      end

      # Draws concurrent samples from exponential distributions.
      #
      # Samples are drawn from multiple exponential distributions with
      # parameters `lam` (rate).
      #
      # The parameters of the distributions are provided as an input
      # array. Let `[s]` be the shape of the input array, `n` be the
      # dimension of `[s]`, `[t]` be the shape specified as the
      # parameter of the operator, and `m` be the dimension of `[t]`.
      # Then the output will be a (`n+m`)-dimensional array with shape
      # `[s]x[t]`.
      #
      # For any valid `n`-dimensional index `i` with respect to the
      # input array, output[i] will be an `m`-dimensional array that
      # holds randomly drawn samples from the distribution which is
      # parameterized by the input value at index `i`. If the shape
      # parameter of the operator is not set, then one sample will be
      # drawn per distribution and the output array has the same shape
      # as the input array.
      #
      # Assume *lam* is an array with the following elements:
      #     [1.0, 8.5]
      #
      # Then:
      #     sample_exponential(lam)             # => [0.51837951, 0.09994757]
      #     sample_exponential(lam, shape: [2]) # => [[0.51837951, 0.19866663], [0.09994757, 0.50447971]]
      #
      # ### Parameters
      # * *lam* (`{{type}}`)
      #   Lambda parameters (rates) of the exponential distributions.
      # * *shape* (`Int` or `Array(Int)`)
      #   Shape to be sampled from each random distribution.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      {{suffix}}
      #
      def self.sample_exponential(lam : self, shape = [] of Int32, **kwargs)
        Internal._sample_exponential(**kwargs.merge({lam: lam, shape: shape}))
      end

      # Draws random samples from gamma distributions.
      #
      # Samples are drawn from multiple gamma distributions with
      # parameters `alpha` (shape) and `beta` (scale).
      #
      # The parameters of the distributions are provided as input
      # arrays. Let `[s]` be the shape of the input arrays, `n` be the
      # dimension of `[s]`, `[t]` be the shape specified as the
      # parameter of the operator, and `m` be the dimension of `[t]`.
      # Then the output will be a (`n+m`)-dimensional array with shape
      # `[s]x[t]`.
      #
      # For any valid `n`-dimensional index `i` with respect to the
      # input arrays, `output[i]` will be an `m`-dimensional array
      # that holds randomly drawn samples from the distribution which
      # is parameterized by the input values at index `i`. If the
      # shape parameter of the operator is not set, then one sample
      # will be drawn per distribution and the output array has the
      # same shape as the input arrays.
      #
      # Assume *alpha* and *beta* are arrays with the following elements:
      #     [0.0, 2.5] # alpha
      #     [1.0, 0.7] # beta
      #
      # Then:
      #     sample_gamma(alpha, beta)             # => [0.0, 2.25797319]
      #     sample_gamma(alpha, beta, shape: [2]) # => [[0.0, 0.0], [2.25797319, 1.70734084]]
      #
      # ### Parameters
      # * *alpha* (`{{type}}`)
      #   Alpha parameters (shapes) of the distributions.
      # * *beta* (`{{type}}`)
      #   Beta parameters (scales) of the distributions.
      # * *shape* (`Int` or `Array(Int)`)
      #   Shape to be sampled from each random distribution.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      {{suffix}}
      #
      def self.sample_gamma(alpha : self, beta : self, shape = [] of Int32, **kwargs)
        Internal._sample_gamma(**kwargs.merge({alpha: alpha, beta: beta, shape: shape}))
      end

      # Draws random samples from multinomial distributions.
      #
      # Samples are drawn from multiple multinomial distributions.
      # Note that the input distribution must be normalized (data must
      # sum to 1 along its last axis).
      #
      # `data` is an `n` dimensional array whose last dimension has
      # length `k`, where `k` is the number of possible outcomes of
      # each multinomial distribution. This operator will draw shape
      # samples from each distribution. If `shape` is empty one sample
      # will be drawn from each distribution.
      #
      # If `get_prob` is `true`, a second array containing log
      # likelihood of the drawn samples will also be returned. This is
      # usually used for reinforcement learning where you can provide
      # reward as head gradient for this array to estimate gradient.
      #
      # Given:
      #     probs = [[0.0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0.0]]
      #
      # Then:
      #     sample_multinomial(probs)                 # => [3, 0]
      #     sample_multinomial(probs, shape: [2])     # => [[4, 2], [0, 0]]
      #     sample_multinomial(probs, get_prob: true) # => [2, 1], [0.2, 0.3]
      #
      # ### Parameters
      # * *data* (`{{type}}`)
      #   Distribution probabilities. Must sum to one on the last axis.
      # * *get_prob* (`Bool`, default = false)
      #   Whether to also return the log probabilities of sampled
      #   results. This is usually used for differentiating through
      #   stochastic variables, e.g. in reinforcement learning.
      # * *shape* (`Int` or `Array(Int)`)
      #   Shape to be sampled from each random distribution.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output in case this can’t be inferred.
      {{suffix}}
      #
      def self.sample_multinomial(data : self, get_prob : Bool = false, **kwargs)
        Internal._sample_multinomial(**kwargs.merge({data: data, get_prob: get_prob}))
      end

      # Reshapes the input array.
      #
      # Returns a copy of the array with a new shape without altering
      # any data.
      #
      # Assume *x* is an array with the following elements:
      #     [1, 2, 3, 4]
      #
      # Then:
      #     reshape(shape: [2, 2]) # => [[1, 2], [3, 4]]
      #
      # Some dimensions of the shape can take special values from the
      # set *{0, -1, -2, -3, -4}*. The significance of each is explained
      # below:
      #
      # * *0* copies this dimension from the input to the output shape:
      #     zeros([2, 3, 4]).reshape([4, 0, 2]).shape # => [4, 3, 2]
      #     zeros([2, 3, 4]).reshape([2, 0, 0]).shape # => [2, 3, 4]
      # * *-1* infers the dimension of the output shape by using the
      #   remainder of the input dimensions, keeping the size of the
      #   new array the same as that of the input array. At most one
      #   dimension can be *-1*:
      #     zeros([2, 3, 4]).reshape([6, 1, -1]).shape # => [6, 1, 4]
      #     zeros([2, 3, 4]).reshape([3, -1, 8]).shape # => [3, 1, 8]
      #     zeros([2, 3, 4]).reshape([-1]).shape # => [24]
      # * *-2* copies all/the remainder of the input dimensions to the
      #   output shape:
      #     zeros([2, 3, 4]).reshape([-2]).shape # => [2, 3, 4]
      #     zeros([2, 3, 4]).reshape([2, -2]).shape # => [2, 3, 4]
      #     zeros([2, 3, 4]).reshape([-2, 1, 1]).shape # => [2, 3, 4, 1, 1]
      # * *-3* uses the product of two consecutive dimensions of the
      #   input shape as the output dimension:
      #     zeros([2, 3, 4]).reshape([-3, 4]).shape # => [6, 4]
      #     zeros([2, 3, 4, 5]).reshape([-3, -3]).shape # => [6, 20]
      #     zeros([2, 3, 4]).reshape([0, -3]).shape # => [2, 12]
      #     zeros([2, 3, 4]).reshape([-3, -2]).shape # => [6, 4]
      # * *-4* splits one dimension of the input into the two dimensions
      #   passed subsequent to *-4* (which can contain *-1*):
      #     zeros([2, 3, 4]).reshape([-4, 1, 2, -2]).shape # => [1, 2, 3, 4]
      #     zeros([2, 3, 4]).reshape([2, -4, -1, 3, -2]).shape # => [2, 1, 3, 4]
      #
      # ### Parameters
      {{prefix}}
      # * *shape* (`Int` or `Array(Int)`)
      #   The target shape.
      # * *reverse* (`Bool`, optional, default `false`)
      #   If `true` then the special values are inferred from right to left.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, reshape)

      # Reshape some or all dimensions of *lhs* to have the same shape
      # as some or all dimensions of *rhs*.
      #
      # Returns a view of the *lhs* array with a new shape without
      # altering any data.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [1, 2, 3, 4, 5, 6]        # x
      #     [[0, -4], [3, 2], [2, 2]] # y
      #
      # Then:
      #     reshape_like(x, y) # => [[1, 2], [3, 4], [5, 6]]
      #
      #
      # ### Parameters
      # * *lhs* (`{{type}}`, required)
      #   The first input.
      # * *rhs* (`{{type}}`, required)
      #   The second input.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, reshape_like)

      # Returns element-wise rounded value to the nearest integer.
      #
      # Note:
      #    - For input *N.5* *rint* returns *N* while *round* returns *N+1*.
      #    - For input *-N.5* both *rint* and *round* return *-N-1*.
      #
      # Assume *x* is an array with the following elements:
      #     [-2.1, -1.9, 1.5, 1.9, 2.1]
      #
      # Then:
      #     rint(x) = [-2.0, -2.0, 1.0, 2.0, 2.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, rint)

      # Returns element-wise rounded value to the nearest integer.
      #
      # Assume *x* is an array with the following elements:
      #     [-2.1, -1.9, 1.5, 1.9, 2.1]
      #
      # Then:
      #     round(x) = [-2.0, -2.0, 2.0, 2.0, 2.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, round)

      # Returns element-wise inverse square-root value of the input.
      #
      #     rsqrt(x) = 1/sqrt(x)
      #
      # Assume *x* is an array with the following elements:
      #     [4, 9, 16]
      #
      # Then:
      #     rsqrt(x) = [0.5, 0.33333, 0.25]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, rsqrt)

      # Update function for Stochastic Gradient Descent (SGD)
      # optimizer.
      #
      # SGD updates the weights using:
      #     weight = weight - learning_rate * (gradient + wd * weight)
      #
      # ### Parameters
      # * *weight* (`{{type}}`, required)
      #   Weights.
      # * *grad* (`{{type}}`, required)
      #   Gradients.
      # * *lr* (`Float`, required)
      #   Learning rate.
      # * *wd* (`Float`, optional, default = 0)
      #   Weight decay augments the objective function with a
      #   regularization term that penalizes large weights. The
      #   penalty scales with the square of the magnitude of each
      #   weight.
      # * *rescale_grad* (`Float`, optional, default = 1.0)
      #   Rescale gradient to `grad = rescale_grad * grad`.
      # * *clip_gradient* (`Float`, optional, default = -1.0)
      #   Clip gradient to the range of *[-clip_gradient,
      #   clip_gradient]*. If `clip_gradient <= 0`, gradient clipping
      #   is turned off.
      # * *lazy_update* (`Bool`, optional, default = true)
      #   If true, lazy updates are applied if gradient's stype is
      #   row_sparse.
      {{suffix}}
      #
      def self.sgd_update(weight : self, grad : self, lr : Float, **kwargs)
        Ops._sgd_update(weight, grad, **kwargs.merge({lr: lr}))
      end

      # Momentum update function for Stochastic Gradient Descent (SGD)
      # optimizer.
      #
      # Momentum update has better convergence rates on neural
      # networks.
      #
      # ### Parameters:
      # * *weight* (`{{type}}`, required)
      #   Weights.
      # * *grad* (`{{type}}`, required)
      #   Gradients.
      # * *mom* (`{{type}}`, required)
      #   Momentum.
      # * *lr* (`Float`, required)
      #   Learning rate.
      # * *momentum* (`Float`, optional, default = 0)
      #   The decay rate of momentum estimates at each epoch.
      # * *wd* (`Float`, optional, default = 0)
      #   Weight decay augments the objective function with a
      #   regularization term that penalizes large weights. The
      #   penalty scales with the square of the magnitude of each
      #   weight.
      # * *rescale_grad* (`Float`, optional, default = 1.0)
      #   Rescale gradient to `grad = rescale_grad * grad`.
      # * *clip_gradient* (`Float`, optional, default = -1.0)
      #   Clip gradient to the range of *[-clip_gradient,
      #   clip_gradient]*. If `clip_gradient <= 0`, gradient clipping
      #   is turned off.
      # * *lazy_update* (`Bool`, optional, default = true)
      #   If true, lazy updates are applied if gradient's stype is
      #   row_sparse.
      {{suffix}}
      #
      def self.sgd_mom_update(weight : self, grad : self, mom : self, lr : Float, **kwargs)
        Ops._sgd_mom_update(weight, grad, mom, **kwargs.merge({lr: lr}))
      end

      # Returns a 1-D array containing the shape of the data.
      #
      # Assume *x* is an array with the following elements:
      #     [[1, 2, 3, 4], [5, 6, 7, 8]]
      #
      # Then:
      #     shape_array(x) = [2, 4]
      #
      # ### Parameters:
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, shape_array)

      # Randomly shuffles the elements.
      #
      # Shuffles the array along the first axis. The order of the
      # elements in each subarray does not change. For example, if a
      # 2-D array is given, the order of the rows randomly changes,
      # but the order of the elements in each row does not change.
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, shuffle)

      # Computes the sigmoid activation.
      #
      # _y=1/(1+exp(−x))_
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sigmoid)

      # Returns the element-wise sign of the input.
      #
      # Assume *x* is an array with the following elements:
      #     [-2, 0, 3]
      #
      # Then:
      #     sign(x) # => [-1, 0, 1]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sign)

      # Computes the element-wise sine of the input array.
      #
      # The input should be in radians (`2\𝜋` radians equals 360 degrees).
      #
      #     sin([0, 𝜋/4, 𝜋/2]) = [0, 0.707, 1]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sin)

      # Returns the hyperbolic sine of the input array, computed element-wise.
      #
      #     sinh(x) = (exp(x) - exp(-x)) / 2
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sinh)

      # Returns a 1-D array containing the size of the data.
      #
      # Assume *x* is an array with the following elements:
      #     [[1, 2, 3, 4], [5, 6, 7, 8]]
      #
      # Then:
      #     size_array(x) = [8]
      #
      # ### Parameters:
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, size_array)

      # Slices a region of the array.
      #
      # This function returns a sliced array between the indices given
      # by *begin* and *end* with the corresponding *step*.
      #
      # For an input array of *shape=[d_0, d_1, ..., d_n-1]*, a slice
      # operation with *begin=[b_0, b_1, ..., b_m-1]*, *end=[e_0, e_1,
      # ..., e_m-1]*, and *step=[s_0, s_1, ..., s_m-1]*, where *m <= n*,
      # results in an array with the shape *(|e_0-b_0|/|s_0|, ...,
      # |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)*.
      #
      # The resulting array's _k_-th dimension contains elements from
      # the _k_-th dimension of the input array starting from index
      # *b_k* (inclusive) with step *s_k* until reaching *e_k*
      # (exclusive).
      #
      # If the _k_-th elements are `nil` in the sequence of *begin*,
      # *end*, and *step*, the following rule will be used to set
      # default values: if `s_k` is `nil`, set `s_k = 1`. If `s_k > 0`,
      # set `b_k = 0`, `e_k = d_k`, else set `b_k = d_k-1`, `e_k = -1`.
      #
      # ### Parameters
      {{prefix}}
      # * *begin* (`Array(Int)`, required)
      #   Beginning indices for the slice operation, supports negative
      #   indices.
      # * *end* (`Array(Int)`, required)
      #   Ending indices for the slice operation, supports negative
      #   indices.
      # * *step* (`Array(Int)`, optional)
      #   Step for the slice operation, supports negative values.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, slice)

      # Slices along a given axis.
      #
      # Returns an array slice along a given *axis* starting from the
      # *begin* index to the *end* index.
      #
      # Assume *x* is an array with the following elements:
      #     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
      #
      # Then:
      #     slice_axis(x, axis: 1, begin: 0, end: 2) # => [[1, 2], [5, 6], [9, 10]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, required)
      #   Axis along which to slice. Supports negative indexes.
      # * *begin* (`Int`, required)
      #   The beginning index along the axis to be sliced. Supports
      #   negative indexes.
      # * *end* (`Int` or `nil`, required)
      #   The ending index along the axis to be sliced. Supports
      #   negative indexes.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, slice_axis)

      # Applies the softmax function.
      #
      # The resulting array contains elements in the range *(0, 1)*
      # and the elements along the given axis sum up to 1.
      #
      # Assume *x* is an array with the following elements:
      #     [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
      #
      # Then:
      #     softmax(x, axis: 0) # => [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
      #     softmax(x, axis: 1) # => [[0.3334, 0.3334, 0.3334], [0.3334, 0.3334, 0.3334]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, optional, default = -1)
      #   The axis along which to compute softmax.
      # * *temperature* (`Float`, optional, default = 1.0)
      #   Temperature parameter in softmax.
      # * *dtype* (`::Symbol`, `:float16`, `:float32` or `:float64`, optional)
      #   Type of the output in case this can't be inferred. Defaults
      #   to the same type as the input if not defined.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, softmax)

      # Returns a sorted copy of an input array along the given axis.
      #
      # Assume *x* is an array with the following elements:
      #     [[1, 4], [3, 1]]
      #
      # Then:
      #     sort(x) = [[1, 4], [1, 3]]
      #     sort(x, axis: 0) = [[1, 1], [3, 4]]
      #     sort(x, axis: None) = [1, 1, 3, 4]
      #     sort(x, is_ascend: false) = [[4, 1], [3, 1]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `None`, optional, default = `-1`)
      #   The axis along which to choose sort the input tensor. If
      #   omitted, the last axis is used. If `None`, the flattened
      #   array is used.
      # * *is_ascend* (`Bool`, optional, default = false)
      #   Whether to sort in ascending or descending order.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sort)

      # Returns element-wise square-root value of the input.
      #
      # Assume *x* is an array with the following elements:
      #     [4, 9, 16]
      #
      # Then:
      #     sqrt(x) # => [2, 3, 4]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sqrt)

      # Returns element-wise squared value of the input.
      #
      # Assume *x* is an array with the following elements:
      #     [2, 3, 4]
      #
      # Then:
      #     square(x) # => [4, 9, 16]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, square)

      # Computes the sum of array elements over given axes.
      #
      # Assume *x* is an array with the following elements:
      #     [[[1, 2], [2, 3], [1, 3]],
      #      [[1, 4], [4, 3], [5, 2]],
      #      [[7, 1], [7, 2], [7, 3]]]
      #
      # Then:
      #     sum(x, axis: 1) # => [[4, 8], [10, 9], [21, 6]]
      #     sum(x, axis: [1, 2]) # => [12, 19, 27]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `Array(Int)`, optional)
      #   The axis or axes along which to perform the
      #   reduction. `axis: []` or `axis: nil` will compute over all
      #   elements into a scalar array with shape `[1]`. If *axis* is
      #   an `Int`, a reduction is performed on a particular axis. If
      #   *axis* is an array of `Int`, a reduction is performed on all
      #   the axes specified in the array. If *exclude* is true,
      #   reduction will be performed on the axes that are **not** in
      #   *axis* instead. Negative values means indexing from right to
      #   left.
      # * *keepdims* (`Bool`, optional, default = false)
      #   If this is set to true, the reduced axes are left in the
      #   result as dimension with size one.
      # * *exclude* (`Bool`, optional, default = false)
      #   Whether to perform reduction on axis that are **not** in
      #   axis instead.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, sum)

      # Takes elements from an input array along the given axis.
      #
      # This function slices the input array along a particular axis
      # with the provided indices.
      #
      # Given data tensor of rank *r >= 1*, and indices tensor of rank
      # *q*, gather entries of the axis dimension of data (by default
      # outer-most one as axis=0) indexed by indices, and concatenate
      # them in an output tensor of rank *q + (r - 1)*.
      #
      # Assume *x* and *i* are arrays with the following elements:
      #     [[1, 2], [3, 4], [5, 6]] # x
      #     [[0, 1], [1, 2]]]        # i
      #
      # Then:
      #     # get rows 0 and 1, then 1 and 2, along axis 0
      #     take(x, i) # => [[[1, 2], [3, 4]], [[3, 4], [5, 6]]]
      #
      # ### Parameters
      # * *a* (`{{type}}`, required)
      #   The input array.
      # * *indices* (`{{type}}`, required)
      #   The indices of the values to be extracted.
      # * *axis* (`Int`, optional, default = 0)
      #   The axis of input array to be taken. For input tensor of
      #   rank *r*, it could be in the range of *[-r, r-1]*.
      # * *mode* (`::Symbol`, `:clip` or `:wrap`, optional, default = :clip)
      #   Specify how out-of-bound indices bahave. *:clip* means to
      #   clip to the range. If all indices mentioned are too large,
      #   they are replaced by the index that addresses the last
      #   element along an axis. *:wrap* means to wrap around.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, take)

      # Computes the element-wise tangent of the input array.
      #
      # The input should be in radians (`2\𝜋` radians equals 360 degrees).
      #
      #     tan([0, 𝜋, 𝜋/2]) = [0, 1, -∞)]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, tan)

      # Returns the hyperbolic tangent of the input array, computed element-wise.
      #
      #     tanh(x) = sinh(x) / cosh(x)
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, tanh)

      # Repeats the array multiple times.
      #
      # Assume *x* is an array with the following elements:
      #     [[1, 2], [3, 4]]
      #
      # If *reps* has length *d*, and the input array has a
      # corresponding dimension of *n*. There are three cases:
      #
      # - **n=d**. Repeat *i*-th dimension of the input *reps[i]* times:
      #     tile(x, reps: [2, 3]) = [[1, 2, 1, 2, 1, 2],
      #                              [3, 4, 3, 4, 3, 4],
      #                              [1, 2, 1, 2, 1, 2],
      #                              [3, 4, 3, 4, 3, 4]]
      #
      # - **n>d**. *reps* is promoted to length *n* by pre-pending
      #   1's. For an input shape `[2, 3]`, `reps: [2]` is treated
      #   as `[1, 2]`:
      #     tile(x, reps: [2]) = [[1, 2, 1, 2],
      #                           [3, 4, 3, 4]]
      #
      # - **n<d**. The input is promoted to be d-dimensional by
      #   prepending new axes. A shape `[2, 2]` array is promoted
      #   to `[1, 2, 2]` for 3-D replication:
      #     tile(x, reps: [2, 2, 3]) = [[[1, 2, 1, 2, 1, 2],
      #                                  [3, 4, 3, 4, 3, 4],
      #                                  [1, 2, 1, 2, 1, 2],
      #                                  [3, 4, 3, 4, 3, 4]],
      #                                 [[1, 2, 1, 2, 1, 2],
      #                                  [3, 4, 3, 4, 3, 4],
      #                                  [1, 2, 1, 2, 1, 2],
      #                                  [3, 4, 3, 4, 3, 4]]]
      #
      # ### Parameters
      {{prefix}}
      # * *reps* (`Array(Int)`)
      #   The number of times to repeat the input array. Each
      #   element of *reps* must be a positive integer.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, tile)

      # Returns the top *k* elements in an input array along the given
      # axis.
      #
      # Examples::
      #
      # Assume *x* is an array with the following elements:
      #     [[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]]
      #
      # Then:
      #     topk(x) = [[2.0], [1.0]]
      #     topk(x, ret_typ: :value, k: 2) = [[0.4, 0.3], [0.3, 0.2]]
      #     topk(x, ret_typ: :value, k: 2, is_ascend: true) = [[0.2, 0.3], [0.1, 0.2]]
      #     topk(x, axis: 0, k: 2) = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int` or `None`, optional, default = `-1`)
      #   Axis along which to choose the top k indices. If omitted,
      #   the last axis is used. If `None`, the flattened array is
      #   used.
      # * *k* (`Int`, optional, default = `1`)
      #   Number of top elements to select. It should be always
      #   smaller than or equal to the element number in the given
      #   axis.
      # * *ret_typ* (`::Symbol`, `:value`, `:indices`, `:mask`, `:both`, optional, default = `:indices`)
      #   The return type. `:value` means to return the top *k*
      #   values, `:indices` means to return the indices of the top
      #   *k* values, `:mask` means to return a mask array containing
      #   0 and 1 (1 means the top *k* value). `:both` means to return
      #   a list of both values and indices of top *k* elements.
      # * *is_ascend* (`Bool`, optional, default = false)
      #   Whether to choose *k* largest or *k* smallest elements. Top
      #   *k* largest elements will be chosen if set to `false`.
      # * *dtype* (`::Symbol`, optional, default = `:float32`)
      #   The data type of the output indices when *ret_typ* is
      #   `:indices` or `:both`.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, topk)

      # Permutes the dimensions of an array.
      #
      # Assume *x* and *y* are arrays with the following elements:
      #     [[[1, 2], [3, 4], [5, 6], [7, 8]]] # x
      #     [[1, 2], [3, 4]]                   # y
      #
      # Then:
      #     transpose(x) # => [[[1], [3], [5], [7]], [[2], [4], [6], [8]]]
      #     transpose(x, axes: [1, 0, 2]) # => [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]
      #     transpose(y) # => [[1, 3], [2, 4]]
      #
      # ### Parameters
      {{prefix}}
      # * *axes* (`Int` or `Array(Int)`, optional)
      #   Target axis order. By default the axes will be inverted.
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, transpose)

      # Return the element-wise truncated value of the input.
      #
      # The truncated value of `x` is the nearest integer `i` which is
      # closer to zero than `x` is. In short, the fractional part of
      # the signed number `x` is discarded.
      #
      # Assume *x* is an array with the following elements:
      #     [-2.1, -1.9, 1.5, 1.9, 2.1]
      #
      # Then:
      #     trunc(x) = [-2.0, -1.0, 1.0, 1.0, 2.0]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, trunc)

      # Returns an array filled with all zeros, with the given shape.
      #
      # ### Parameters
      {{prefix}}
      # * *shape* (`Int` or `Array(Int)`)
      #   The shape of the array.
      # * *dtype* (`::Symbol`, default = `:float32`)
      #   The data type of the output array.
      # * *ctx* (`Context`, optional)
      #   Device context (default is the current context). Only used
      #   for imperative calls.
      {{suffix}}
      #
      def self.zeros(shape : Int | Array(Int), ctx = Context.current, **kwargs)
        Internal._zeros(**kwargs.merge({shape: shape, ctx: ctx}))
      end

      # Returns an array of zeros with the same shape, data type and
      # storage type as the input array.
      #
      # Assume *x* is an array with the following elements:
      #     [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
      #
      # Then:
      #     zeros_like(x) # => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_method(Ops, zeros_like)
    end
  end
end
