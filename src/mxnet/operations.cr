module MXNet
  # Extends `MXNet::NDArray` and `MXNet::Symbol` classes with
  # wrappers for native MXNet operations.
  #
  module Operations
    # :nodoc:
    OP_INFO = {
      {"Activation","Activation",["data"],["act_type"],nil},
      {"BatchNorm","BatchNorm",["data","gamma","beta","moving_mean","moving_var"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var","axis","cudnn_off"]},
      {"BatchNorm_v1","BatchNorm_v1",["data","gamma","beta"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var"]},
      {"BilinearSampler","BilinearSampler",["data","grid"],nil,nil},
      {"BlockGrad","BlockGrad",["data"],nil,nil},
      {"Cast","Cast",["data"],["dtype"],nil},
      {"Concat","Concat",["*data"],["num_args"],["dim"]},
      {"Convolution","Convolution",["data","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      {"Convolution_v1","Convolution_v1",["data","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      {"Correlation","Correlation",["data1","data2"],nil,["kernel_size","max_displacement","stride1","stride2","pad_size","is_multiply"]},
      {"Crop","Crop",nil,["num_args"],["offset","h_w","center_crop"]},
      {"CuDNNBatchNorm","CuDNNBatchNorm",["data","gamma","beta","moving_mean","moving_var"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var","axis","cudnn_off"]},
      {"Custom","Custom",["*data"],["op_type"],nil},
      {"Deconvolution","Deconvolution",["data","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","adj","target_shape","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      {"Dropout","Dropout",["data"],nil,["p","mode","axes"]},
      {"ElementWiseSum","add_n",["*args"],nil,nil},
      {"Embedding","Embedding",["data","weight"],["input_dim","output_dim"],["dtype","sparse_grad"]},
      {"Flatten","Flatten",["data"],nil,nil},
      {"FullyConnected","FullyConnected",["data","weight","bias"],["num_hidden"],["no_bias","flatten"]},
      {"GridGenerator","GridGenerator",["data"],["transform_type"],["target_shape"]},
      {"IdentityAttachKLSparseReg","IdentityAttachKLSparseReg",["data"],nil,["sparseness_target","penalty","momentum"]},
      {"InstanceNorm","InstanceNorm",["data","gamma","beta"],nil,["eps"]},
      {"L2Normalization","L2Normalization",["data"],nil,["eps","mode"]},
      {"LRN","LRN",["data"],["nsize"],["alpha","beta","knorm"]},
      {"LayerNorm","LayerNorm",["data","gamma","beta"],nil,["axis","eps","output_mean_var"]},
      {"LeakyReLU","LeakyReLU",["data","gamma"],nil,["act_type","slope","lower_bound","upper_bound"]},
      {"LinearRegressionOutput","LinearRegressionOutput",["data","label"],nil,["grad_scale"]},
      {"LogisticRegressionOutput","LogisticRegressionOutput",["data","label"],nil,["grad_scale"]},
      {"MAERegressionOutput","MAERegressionOutput",["data","label"],nil,["grad_scale"]},
      {"MakeLoss","MakeLoss",["data"],nil,["grad_scale","valid_thresh","normalization"]},
      {"Pad","Pad",["data"],["mode","pad_width"],["constant_value"]},
      {"Pooling","Pooling",["data"],nil,["kernel","pool_type","global_pool","cudnn_off","pooling_convention","stride","pad","p_value","count_include_pad"]},
      {"Pooling_v1","Pooling_v1",["data"],nil,["kernel","pool_type","global_pool","pooling_convention","stride","pad"]},
      {"RNN","RNN",["data","parameters","state","state_cell"],["state_size","num_layers","mode"],["bidirectional","p","state_outputs"]},
      {"ROIPooling","ROIPooling",["data","rois"],["pooled_size","spatial_scale"],nil},
      {"Reshape","Reshape",["data"],nil,["shape","reverse","target_shape","keep_highest"]},
      {"SVMOutput","SVMOutput",["data","label"],nil,["margin","regularization_coefficient","use_linear"]},
      {"SequenceLast","SequenceLast",["data","sequence_length"],nil,["use_sequence_length","axis"]},
      {"SequenceMask","SequenceMask",["data","sequence_length"],nil,["use_sequence_length","value","axis"]},
      {"SequenceReverse","SequenceReverse",["data","sequence_length"],nil,["use_sequence_length","axis"]},
      {"SliceChannel","SliceChannel",["data"],["num_outputs"],["axis","squeeze_axis"]},
      {"Softmax","Softmax",["data"],nil,["grad_scale","ignore_label","multi_output","use_ignore","preserve_shape","normalization","out_grad","smooth_alpha"]},
      {"SoftmaxActivation","SoftmaxActivation",["data"],nil,["mode"]},
      {"SoftmaxOutput","SoftmaxOutput",["data","label"],nil,["grad_scale","ignore_label","multi_output","use_ignore","preserve_shape","normalization","out_grad","smooth_alpha"]},
      {"SpatialTransformer","SpatialTransformer",["data","loc"],["transform_type","sampler_type"],["target_shape"]},
      {"SwapAxis","SwapAxis",["data"],nil,["dim1","dim2"]},
      {"UpSampling","UpSampling",["*data"],["scale","sample_type","num_args"],["num_filter","multi_input_mode","workspace"]},
      {"_CachedOp","_CachedOp",nil,nil,nil},
      {"_CrossDeviceCopy","_CrossDeviceCopy",nil,nil,nil},
      {"_CustomFunction","_CustomFunction",nil,nil,nil},
      {"_Div","elemwise_div",["lhs","rhs"],nil,nil},
      {"_DivScalar","_div_scalar",["data"],["scalar"],nil},
      {"_Equal","_equal",["lhs","rhs"],nil,nil},
      {"_EqualScalar","_equal_scalar",["data"],["scalar"],nil},
      {"_Greater","_greater",["lhs","rhs"],nil,nil},
      {"_GreaterEqualScalar","_greater_equal_scalar",["data"],["scalar"],nil},
      {"_GreaterScalar","_greater_scalar",["data"],["scalar"],nil},
      {"_Greater_Equal","_greater_equal",["lhs","rhs"],nil,nil},
      {"_Hypot","_hypot",["lhs","rhs"],nil,nil},
      {"_HypotScalar","_hypot_scalar",["data"],["scalar"],nil},
      {"_Lesser","_lesser",["lhs","rhs"],nil,nil},
      {"_LesserEqualScalar","_lesser_equal_scalar",["data"],["scalar"],nil},
      {"_LesserScalar","_lesser_scalar",["data"],["scalar"],nil},
      {"_Lesser_Equal","_lesser_equal",["lhs","rhs"],nil,nil},
      {"_LogicalAndScalar","_logical_and_scalar",["data"],["scalar"],nil},
      {"_LogicalOrScalar","_logical_or_scalar",["data"],["scalar"],nil},
      {"_LogicalXorScalar","_logical_xor_scalar",["data"],["scalar"],nil},
      {"_Logical_And","_logical_and",["lhs","rhs"],nil,nil},
      {"_Logical_Or","_logical_or",["lhs","rhs"],nil,nil},
      {"_Logical_Xor","_logical_xor",["lhs","rhs"],nil,nil},
      {"_Maximum","_maximum",["lhs","rhs"],nil,nil},
      {"_MaximumScalar","_maximum_scalar",["data"],["scalar"],nil},
      {"_Minimum","_minimum",["lhs","rhs"],nil,nil},
      {"_MinimumScalar","_minimum_scalar",["data"],["scalar"],nil},
      {"_Minus","elemwise_sub",["lhs","rhs"],nil,nil},
      {"_MinusScalar","_minus_scalar",["data"],["scalar"],nil},
      {"_Mod","_mod",["lhs","rhs"],nil,nil},
      {"_ModScalar","_mod_scalar",["data"],["scalar"],nil},
      {"_Mul","elemwise_mul",["lhs","rhs"],nil,nil},
      {"_MulScalar","_mul_scalar",["data"],["scalar"],nil},
      {"_NDArray","_NDArray",["*data"],["info"],nil},
      {"_Native","_Native",["*data"],["info"],["need_top_grad"]},
      {"_NoGradient","_NoGradient",nil,nil,nil},
      {"_NotEqualScalar","_not_equal_scalar",["data"],["scalar"],nil},
      {"_Not_Equal","_not_equal",["lhs","rhs"],nil,nil},
      {"_Plus","elemwise_add",["lhs","rhs"],nil,nil},
      {"_PlusScalar","_plus_scalar",["data"],["scalar"],nil},
      {"_Power","_power",["lhs","rhs"],nil,nil},
      {"_PowerScalar","_power_scalar",["data"],["scalar"],nil},
      {"_RDivScalar","_rdiv_scalar",["data"],["scalar"],nil},
      {"_RMinusScalar","_rminus_scalar",["data"],["scalar"],nil},
      {"_RModScalar","_rmod_scalar",["data"],["scalar"],nil},
      {"_RPowerScalar","_rpower_scalar",["data"],["scalar"],nil},
      {"_add","elemwise_add",["lhs","rhs"],nil,nil},
      {"_arange","_arange",nil,["start"],["stop","step","repeat","ctx","dtype"]},
      {"_broadcast_backward","_broadcast_backward",nil,nil,nil},
      {"_cond","_cond",["*data"],["num_args","num_outputs","cond_input_locs","then_input_locs","else_input_locs"],nil},
      {"_contrib_AdaptiveAvgPooling2D","_contrib_AdaptiveAvgPooling2D",["data"],nil,["output_size"]},
      {"_contrib_BilinearResize2D","_contrib_BilinearResize2D",["data"],["height","width"],nil},
      {"_contrib_CTCLoss","_contrib_CTCLoss",["data","label","data_lengths","label_lengths"],nil,["use_data_lengths","use_label_lengths","blank_label"]},
      {"_contrib_DeformableConvolution","_contrib_DeformableConvolution",["data","offset","weight","bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","num_deformable_group","workspace","no_bias","layout"]},
      {"_contrib_DeformablePSROIPooling","_contrib_DeformablePSROIPooling",nil,["spatial_scale","output_dim","group_size","pooled_size"],["part_size","sample_per_part","trans_std","no_trans"]},
      {"_contrib_MultiBoxDetection","_contrib_MultiBoxDetection",["cls_prob","loc_pred","anchor"],nil,["clip","threshold","background_id","nms_threshold","force_suppress","variances","nms_topk"]},
      {"_contrib_MultiBoxPrior","_contrib_MultiBoxPrior",["data"],nil,["sizes","ratios","clip","steps","offsets"]},
      {"_contrib_MultiBoxTarget","_contrib_MultiBoxTarget",["anchor","label","cls_pred"],nil,["overlap_threshold","ignore_label","negative_mining_ratio","negative_mining_thresh","minimum_negative_samples","variances"]},
      {"_contrib_MultiProposal","_contrib_MultiProposal",["cls_prob","bbox_pred","im_info"],nil,["rpn_pre_nms_top_n","rpn_post_nms_top_n","threshold","rpn_min_size","scales","ratios","feature_stride","output_score","iou_loss"]},
      {"_contrib_PSROIPooling","_contrib_PSROIPooling",nil,["spatial_scale","output_dim","pooled_size"],["group_size"]},
      {"_contrib_Proposal","_contrib_Proposal",["cls_prob","bbox_pred","im_info"],nil,["rpn_pre_nms_top_n","rpn_post_nms_top_n","threshold","rpn_min_size","scales","ratios","feature_stride","output_score","iou_loss"]},
      {"_contrib_ROIAlign","_contrib_ROIAlign",["data","rois"],["pooled_size","spatial_scale"],["sample_ratio"]},
      {"_contrib_SparseEmbedding","_contrib_SparseEmbedding",["data","weight"],["input_dim","output_dim"],["dtype","sparse_grad"]},
      {"_contrib_SyncBatchNorm","_contrib_SyncBatchNorm",["data","gamma","beta","moving_mean","moving_var"],nil,["eps","momentum","fix_gamma","use_global_stats","output_mean_var","ndev","key"]},
      {"_contrib_backward_quadratic","_contrib_backward_quadratic",nil,nil,nil},
      {"_contrib_bipartite_matching","_contrib_bipartite_matching",["data"],["threshold"],["is_ascend","topk"]},
      {"_contrib_box_iou","_contrib_box_iou",["lhs","rhs"],nil,["format"]},
      {"_contrib_box_nms","_contrib_box_nms",["data"],nil,["overlap_thresh","valid_thresh","topk","coord_start","score_index","id_index","force_suppress","in_format","out_format"]},
      {"_contrib_box_non_maximum_suppression","_contrib_box_nms",["data"],nil,["overlap_thresh","valid_thresh","topk","coord_start","score_index","id_index","force_suppress","in_format","out_format"]},
      {"_contrib_count_sketch","_contrib_count_sketch",["data","h","s"],["out_dim"],["processing_batch_size"]},
      {"_contrib_ctc_loss","_contrib_CTCLoss",["data","label","data_lengths","label_lengths"],nil,["use_data_lengths","use_label_lengths","blank_label"]},
      {"_contrib_dequantize","_contrib_dequantize",["data","min_range","max_range"],nil,["out_type"]},
      {"_contrib_div_sqrt_dim","_contrib_div_sqrt_dim",["data"],nil,nil},
      {"_contrib_fft","_contrib_fft",["data"],nil,["compute_size"]},
      {"_contrib_ifft","_contrib_ifft",["data"],nil,["compute_size"]},
      {"_contrib_quadratic","_contrib_quadratic",["data"],nil,["a","b","c"]},
      {"_contrib_quantize","_contrib_quantize",["data","min_range","max_range"],nil,["out_type"]},
      {"_contrib_quantized_conv","_contrib_quantized_conv",["data","weight","bias","min_data","max_data","min_weight","max_weight","min_bias","max_bias"],["kernel","num_filter"],["stride","dilate","pad","num_group","workspace","no_bias","cudnn_tune","cudnn_off","layout"]},
      {"_contrib_quantized_flatten","_contrib_quantized_flatten",["data","min_data","max_data"],nil,nil},
      {"_contrib_quantized_fully_connected","_contrib_quantized_fully_connected",["data","weight","bias","min_data","max_data","min_weight","max_weight","min_bias","max_bias"],["num_hidden"],["no_bias","flatten"]},
      {"_contrib_quantized_pooling","_contrib_quantized_pooling",["data","min_data","max_data"],nil,["kernel","pool_type","global_pool","cudnn_off","pooling_convention","stride","pad","p_value","count_include_pad"]},
      {"_contrib_requantize","_contrib_requantize",["data","min_range","max_range"],nil,["min_calib_range","max_calib_range"]},
      {"_copy","_copy",["data"],nil,nil},
      {"_copyto","_copyto",nil,nil,nil},
      {"_crop_assign","_slice_assign",["lhs","rhs"],["begin","end"],["step"]},
      {"_crop_assign_scalar","_slice_assign_scalar",["data"],["begin","end"],["scalar","step"]},
      {"_cvcopyMakeBorder","_cvcopyMakeBorder",nil,["top","bot","left","right"],["type","value","values"]},
      {"_cvimdecode","_cvimdecode",nil,nil,["flag","to_rgb"]},
      {"_cvimread","_cvimread",nil,["filename"],["flag","to_rgb"]},
      {"_cvimresize","_cvimresize",nil,["w","h"],["interp"]},
      {"_div","elemwise_div",["lhs","rhs"],nil,nil},
      {"_div_scalar","_div_scalar",["data"],["scalar"],nil},
      {"_equal","_equal",["lhs","rhs"],nil,nil},
      {"_equal_scalar","_equal_scalar",["data"],["scalar"],nil},
      {"_eye","_eye",nil,["N"],["M","k","ctx","dtype"]},
      {"_foreach","_foreach",["*data"],["num_args","num_outputs","num_out_data","in_state_locs","in_data_locs","remain_locs"],nil},
      {"_full","_full",nil,["value"],["shape","ctx","dtype"]},
      {"_grad_add","_grad_add",["lhs","rhs"],nil,nil},
      {"_greater","_greater",["lhs","rhs"],nil,nil},
      {"_greater_equal","_greater_equal",["lhs","rhs"],nil,nil},
      {"_greater_equal_scalar","_greater_equal_scalar",["data"],["scalar"],nil},
      {"_greater_scalar","_greater_scalar",["data"],["scalar"],nil},
      {"_histogram","_histogram",["data","bins"],nil,["bin_cnt","range"]},
      {"_hypot","_hypot",["lhs","rhs"],nil,nil},
      {"_hypot_scalar","_hypot_scalar",["data"],["scalar"],nil},
      {"_identity_with_attr_like_rhs","_identity_with_attr_like_rhs",["lhs","rhs"],nil,nil},
      {"_image_adjust_lighting","_image_adjust_lighting",["data"],["alpha"],nil},
      {"_image_flip_left_right","_image_flip_left_right",["data"],nil,nil},
      {"_image_flip_top_bottom","_image_flip_top_bottom",["data"],nil,nil},
      {"_image_normalize","_image_normalize",["data"],["mean","std"],nil},
      {"_image_random_brightness","_image_random_brightness",["data"],["min_factor","max_factor"],nil},
      {"_image_random_color_jitter","_image_random_color_jitter",["data"],["brightness","contrast","saturation","hue"],nil},
      {"_image_random_contrast","_image_random_contrast",["data"],["min_factor","max_factor"],nil},
      {"_image_random_flip_left_right","_image_random_flip_left_right",["data"],nil,nil},
      {"_image_random_flip_top_bottom","_image_random_flip_top_bottom",["data"],nil,nil},
      {"_image_random_hue","_image_random_hue",["data"],["min_factor","max_factor"],nil},
      {"_image_random_lighting","_image_random_lighting",["data"],nil,["alpha_std"]},
      {"_image_random_saturation","_image_random_saturation",["data"],["min_factor","max_factor"],nil},
      {"_image_to_tensor","_image_to_tensor",["data"],nil,nil},
      {"_imdecode","_imdecode",["mean"],["index","x0","y0","x1","y1","c","size"],nil},
      {"_lesser","_lesser",["lhs","rhs"],nil,nil},
      {"_lesser_equal","_lesser_equal",["lhs","rhs"],nil,nil},
      {"_lesser_equal_scalar","_lesser_equal_scalar",["data"],["scalar"],nil},
      {"_lesser_scalar","_lesser_scalar",["data"],["scalar"],nil},
      {"_linalg_gelqf","_linalg_gelqf",["A"],nil,nil},
      {"_linalg_gemm","_linalg_gemm",["A","B","C"],nil,["transpose_a","transpose_b","alpha","beta","axis"]},
      {"_linalg_gemm2","_linalg_gemm2",["A","B"],nil,["transpose_a","transpose_b","alpha","axis"]},
      {"_linalg_potrf","_linalg_potrf",["A"],nil,nil},
      {"_linalg_potri","_linalg_potri",["A"],nil,nil},
      {"_linalg_sumlogdiag","_linalg_sumlogdiag",["A"],nil,nil},
      {"_linalg_syevd","_linalg_syevd",["A"],nil,nil},
      {"_linalg_syrk","_linalg_syrk",["A"],nil,["transpose","alpha"]},
      {"_linalg_trmm","_linalg_trmm",["A","B"],nil,["transpose","rightside","alpha"]},
      {"_linalg_trsm","_linalg_trsm",["A","B"],nil,["transpose","rightside","alpha"]},
      {"_logical_and","_logical_and",["lhs","rhs"],nil,nil},
      {"_logical_and_scalar","_logical_and_scalar",["data"],["scalar"],nil},
      {"_logical_or","_logical_or",["lhs","rhs"],nil,nil},
      {"_logical_or_scalar","_logical_or_scalar",["data"],["scalar"],nil},
      {"_logical_xor","_logical_xor",["lhs","rhs"],nil,nil},
      {"_logical_xor_scalar","_logical_xor_scalar",["data"],["scalar"],nil},
      {"_maximum","_maximum",["lhs","rhs"],nil,nil},
      {"_maximum_scalar","_maximum_scalar",["data"],["scalar"],nil},
      {"_minimum","_minimum",["lhs","rhs"],nil,nil},
      {"_minimum_scalar","_minimum_scalar",["data"],["scalar"],nil},
      {"_minus","elemwise_sub",["lhs","rhs"],nil,nil},
      {"_minus_scalar","_minus_scalar",["data"],["scalar"],nil},
      {"_mod","_mod",["lhs","rhs"],nil,nil},
      {"_mod_scalar","_mod_scalar",["data"],["scalar"],nil},
      {"_mul","elemwise_mul",["lhs","rhs"],nil,nil},
      {"_mul_scalar","_mul_scalar",["data"],["scalar"],nil},
      {"_not_equal","_not_equal",["lhs","rhs"],nil,nil},
      {"_not_equal_scalar","_not_equal_scalar",["data"],["scalar"],nil},
      {"_onehot_encode","_onehot_encode",nil,nil,nil},
      {"_ones","_ones",nil,nil,["shape","ctx","dtype"]},
      {"_plus","elemwise_add",["lhs","rhs"],nil,nil},
      {"_plus_scalar","_plus_scalar",["data"],["scalar"],nil},
      {"_power","_power",["lhs","rhs"],nil,nil},
      {"_power_scalar","_power_scalar",["data"],["scalar"],nil},
      {"_random_exponential","_random_exponential",nil,nil,["lam","shape","ctx","dtype"]},
      {"_random_gamma","_random_gamma",nil,nil,["alpha","beta","shape","ctx","dtype"]},
      {"_random_generalized_negative_binomial","_random_generalized_negative_binomial",nil,nil,["mu","alpha","shape","ctx","dtype"]},
      {"_random_negative_binomial","_random_negative_binomial",nil,nil,["k","p","shape","ctx","dtype"]},
      {"_random_normal","_random_normal",nil,nil,["loc","scale","shape","ctx","dtype"]},
      {"_random_poisson","_random_poisson",nil,nil,["lam","shape","ctx","dtype"]},
      {"_random_uniform","_random_uniform",nil,nil,["low","high","shape","ctx","dtype"]},
      {"_ravel_multi_index","_ravel_multi_index",["data"],nil,["shape"]},
      {"_rdiv_scalar","_rdiv_scalar",["data"],["scalar"],nil},
      {"_rminus_scalar","_rminus_scalar",["data"],["scalar"],nil},
      {"_rmod_scalar","_rmod_scalar",["data"],["scalar"],nil},
      {"_rpower_scalar","_rpower_scalar",["data"],["scalar"],nil},
      {"_sample_exponential","_sample_exponential",["lam"],nil,["shape","dtype"]},
      {"_sample_gamma","_sample_gamma",["alpha","beta"],nil,["shape","dtype"]},
      {"_sample_generalized_negative_binomial","_sample_generalized_negative_binomial",["mu","alpha"],nil,["shape","dtype"]},
      {"_sample_multinomial","_sample_multinomial",["data"],nil,["shape","get_prob","dtype"]},
      {"_sample_negative_binomial","_sample_negative_binomial",["k","p"],nil,["shape","dtype"]},
      {"_sample_normal","_sample_normal",["mu","sigma"],nil,["shape","dtype"]},
      {"_sample_poisson","_sample_poisson",["lam"],nil,["shape","dtype"]},
      {"_sample_uniform","_sample_uniform",["low","high"],nil,["shape","dtype"]},
      {"_scatter_elemwise_div","_scatter_elemwise_div",["lhs","rhs"],nil,nil},
      {"_scatter_minus_scalar","_scatter_minus_scalar",["data"],["scalar"],nil},
      {"_scatter_plus_scalar","_scatter_plus_scalar",["data"],["scalar"],nil},
      {"_scatter_set_nd","_scatter_set_nd",["lhs","rhs","indices"],["shape"],nil},
      {"_set_value","_set_value",nil,nil,nil},
      {"_shuffle","_shuffle",["data"],nil,nil},
      {"_slice_assign","_slice_assign",["lhs","rhs"],["begin","end"],["step"]},
      {"_slice_assign_scalar","_slice_assign_scalar",["data"],["begin","end"],["scalar","step"]},
      {"_sparse_ElementWiseSum","add_n",["*args"],nil,nil},
      {"_sparse_Embedding","Embedding",["data","weight"],["input_dim","output_dim"],["dtype","sparse_grad"]},
      {"_sparse_FullyConnected","FullyConnected",["data","weight","bias"],["num_hidden"],["no_bias","flatten"]},
      {"_sparse_LinearRegressionOutput","LinearRegressionOutput",["data","label"],nil,["grad_scale"]},
      {"_sparse_LogisticRegressionOutput","LogisticRegressionOutput",["data","label"],nil,["grad_scale"]},
      {"_sparse_MAERegressionOutput","MAERegressionOutput",["data","label"],nil,["grad_scale"]},
      {"_sparse_abs","abs",["data"],nil,nil},
      {"_sparse_adagrad_update","_sparse_adagrad_update",["weight","grad","history"],["lr"],["epsilon","wd","rescale_grad","clip_gradient"]},
      {"_sparse_adam_update","adam_update",["weight","grad","mean","var"],["lr"],["beta1","beta2","epsilon","wd","rescale_grad","clip_gradient","lazy_update"]},
      {"_sparse_add_n","add_n",["*args"],nil,nil},
      {"_sparse_arccos","arccos",["data"],nil,nil},
      {"_sparse_arccosh","arccosh",["data"],nil,nil},
      {"_sparse_arcsin","arcsin",["data"],nil,nil},
      {"_sparse_arcsinh","arcsinh",["data"],nil,nil},
      {"_sparse_arctan","arctan",["data"],nil,nil},
      {"_sparse_arctanh","arctanh",["data"],nil,nil},
      {"_sparse_broadcast_add","broadcast_add",["lhs","rhs"],nil,nil},
      {"_sparse_broadcast_div","broadcast_div",["lhs","rhs"],nil,nil},
      {"_sparse_broadcast_minus","broadcast_sub",["lhs","rhs"],nil,nil},
      {"_sparse_broadcast_mul","broadcast_mul",["lhs","rhs"],nil,nil},
      {"_sparse_broadcast_plus","broadcast_add",["lhs","rhs"],nil,nil},
      {"_sparse_broadcast_sub","broadcast_sub",["lhs","rhs"],nil,nil},
      {"_sparse_cast_storage","cast_storage",["data"],["stype"],nil},
      {"_sparse_cbrt","cbrt",["data"],nil,nil},
      {"_sparse_ceil","ceil",["data"],nil,nil},
      {"_sparse_clip","clip",["data"],["a_min","a_max"],nil},
      {"_sparse_concat","Concat",["*data"],["num_args"],["dim"]},
      {"_sparse_cos","cos",["data"],nil,nil},
      {"_sparse_cosh","cosh",["data"],nil,nil},
      {"_sparse_degrees","degrees",["data"],nil,nil},
      {"_sparse_dot","dot",["lhs","rhs"],nil,["transpose_a","transpose_b","forward_stype"]},
      {"_sparse_elemwise_add","elemwise_add",["lhs","rhs"],nil,nil},
      {"_sparse_elemwise_div","elemwise_div",["lhs","rhs"],nil,nil},
      {"_sparse_elemwise_mul","elemwise_mul",["lhs","rhs"],nil,nil},
      {"_sparse_elemwise_sub","elemwise_sub",["lhs","rhs"],nil,nil},
      {"_sparse_exp","exp",["data"],nil,nil},
      {"_sparse_expm1","expm1",["data"],nil,nil},
      {"_sparse_fix","fix",["data"],nil,nil},
      {"_sparse_floor","floor",["data"],nil,nil},
      {"_sparse_ftrl_update","ftrl_update",["weight","grad","z","n"],["lr"],["lamda1","beta","wd","rescale_grad","clip_gradient"]},
      {"_sparse_gamma","gamma",["data"],nil,nil},
      {"_sparse_gammaln","gammaln",["data"],nil,nil},
      {"_sparse_log","log",["data"],nil,nil},
      {"_sparse_log10","log10",["data"],nil,nil},
      {"_sparse_log1p","log1p",["data"],nil,nil},
      {"_sparse_log2","log2",["data"],nil,nil},
      {"_sparse_make_loss","make_loss",["data"],nil,nil},
      {"_sparse_mean","mean",["data"],nil,["axis","keepdims","exclude"]},
      {"_sparse_negative","negative",["data"],nil,nil},
      {"_sparse_norm","norm",["data"],nil,["ord","axis","keepdims"]},
      {"_sparse_radians","radians",["data"],nil,nil},
      {"_sparse_relu","relu",["data"],nil,nil},
      {"_sparse_retain","_sparse_retain",["data","indices"],nil,nil},
      {"_sparse_rint","rint",["data"],nil,nil},
      {"_sparse_round","round",["data"],nil,nil},
      {"_sparse_rsqrt","rsqrt",["data"],nil,nil},
      {"_sparse_sgd_mom_update","sgd_mom_update",["weight","grad","mom"],["lr"],["momentum","wd","rescale_grad","clip_gradient","lazy_update"]},
      {"_sparse_sgd_update","sgd_update",["weight","grad"],["lr"],["wd","rescale_grad","clip_gradient","lazy_update"]},
      {"_sparse_sigmoid","sigmoid",["data"],nil,nil},
      {"_sparse_sign","sign",["data"],nil,nil},
      {"_sparse_sin","sin",["data"],nil,nil},
      {"_sparse_sinh","sinh",["data"],nil,nil},
      {"_sparse_slice","slice",["data"],["begin","end"],["step"]},
      {"_sparse_sqrt","sqrt",["data"],nil,nil},
      {"_sparse_square","square",["data"],nil,nil},
      {"_sparse_stop_gradient","BlockGrad",["data"],nil,nil},
      {"_sparse_sum","sum",["data"],nil,["axis","keepdims","exclude"]},
      {"_sparse_tan","tan",["data"],nil,nil},
      {"_sparse_tanh","tanh",["data"],nil,nil},
      {"_sparse_trunc","trunc",["data"],nil,nil},
      {"_sparse_where","where",["condition","x","y"],nil,nil},
      {"_sparse_zeros_like","zeros_like",["data"],nil,nil},
      {"_square_sum","_square_sum",["data"],nil,["axis","keepdims","exclude"]},
      {"_sub","elemwise_sub",["lhs","rhs"],nil,nil},
      {"_unravel_index","_unravel_index",["data"],nil,["shape"]},
      {"_while_loop","_while_loop",["*data"],["num_args","num_outputs","num_out_data","max_iterations","cond_input_locs","func_input_locs","func_var_locs"],nil},
      {"_zeros","_zeros",nil,nil,["shape","ctx","dtype"]},
      {"abs","abs",["data"],nil,nil},
      {"adam_update","adam_update",["weight","grad","mean","var"],["lr"],["beta1","beta2","epsilon","wd","rescale_grad","clip_gradient","lazy_update"]},
      {"add_n","add_n",["*args"],nil,nil},
      {"arccos","arccos",["data"],nil,nil},
      {"arccosh","arccosh",["data"],nil,nil},
      {"arcsin","arcsin",["data"],nil,nil},
      {"arcsinh","arcsinh",["data"],nil,nil},
      {"arctan","arctan",["data"],nil,nil},
      {"arctanh","arctanh",["data"],nil,nil},
      {"argmax","argmax",["data"],nil,["axis","keepdims"]},
      {"argmax_channel","argmax_channel",["data"],nil,nil},
      {"argmin","argmin",["data"],nil,["axis","keepdims"]},
      {"argsort","argsort",["data"],nil,["axis","is_ascend"]},
      {"batch_dot","batch_dot",["lhs","rhs"],nil,["transpose_a","transpose_b","forward_stype"]},
      {"batch_take","batch_take",["a","indices"],nil,nil},
      {"broadcast_add","broadcast_add",["lhs","rhs"],nil,nil},
      {"broadcast_axes","broadcast_axis",["data"],nil,["axis","size"]},
      {"broadcast_axis","broadcast_axis",["data"],nil,["axis","size"]},
      {"broadcast_div","broadcast_div",["lhs","rhs"],nil,nil},
      {"broadcast_equal","broadcast_equal",["lhs","rhs"],nil,nil},
      {"broadcast_greater","broadcast_greater",["lhs","rhs"],nil,nil},
      {"broadcast_greater_equal","broadcast_greater_equal",["lhs","rhs"],nil,nil},
      {"broadcast_hypot","broadcast_hypot",["lhs","rhs"],nil,nil},
      {"broadcast_lesser","broadcast_lesser",["lhs","rhs"],nil,nil},
      {"broadcast_lesser_equal","broadcast_lesser_equal",["lhs","rhs"],nil,nil},
      {"broadcast_like","broadcast_like",["lhs","rhs"],nil,nil},
      {"broadcast_logical_and","broadcast_logical_and",["lhs","rhs"],nil,nil},
      {"broadcast_logical_or","broadcast_logical_or",["lhs","rhs"],nil,nil},
      {"broadcast_logical_xor","broadcast_logical_xor",["lhs","rhs"],nil,nil},
      {"broadcast_maximum","broadcast_maximum",["lhs","rhs"],nil,nil},
      {"broadcast_minimum","broadcast_minimum",["lhs","rhs"],nil,nil},
      {"broadcast_minus","broadcast_sub",["lhs","rhs"],nil,nil},
      {"broadcast_mod","broadcast_mod",["lhs","rhs"],nil,nil},
      {"broadcast_mul","broadcast_mul",["lhs","rhs"],nil,nil},
      {"broadcast_not_equal","broadcast_not_equal",["lhs","rhs"],nil,nil},
      {"broadcast_plus","broadcast_add",["lhs","rhs"],nil,nil},
      {"broadcast_power","broadcast_power",["lhs","rhs"],nil,nil},
      {"broadcast_sub","broadcast_sub",["lhs","rhs"],nil,nil},
      {"broadcast_to","broadcast_to",["data"],nil,["shape"]},
      {"cast","Cast",["data"],["dtype"],nil},
      {"cast_storage","cast_storage",["data"],["stype"],nil},
      {"cbrt","cbrt",["data"],nil,nil},
      {"ceil","ceil",["data"],nil,nil},
      {"choose_element_0index","choose_element_0index",nil,nil,nil},
      {"clip","clip",["data"],["a_min","a_max"],nil},
      {"concat","Concat",["*data"],["num_args"],["dim"]},
      {"cos","cos",["data"],nil,nil},
      {"cosh","cosh",["data"],nil,nil},
      {"crop","slice",["data"],["begin","end"],["step"]},
      {"degrees","degrees",["data"],nil,nil},
      {"depth_to_space","depth_to_space",["data"],["block_size"],nil},
      {"diag","diag",["data"],nil,["k"]},
      {"dot","dot",["lhs","rhs"],nil,["transpose_a","transpose_b","forward_stype"]},
      {"elemwise_add","elemwise_add",["lhs","rhs"],nil,nil},
      {"elemwise_div","elemwise_div",["lhs","rhs"],nil,nil},
      {"elemwise_mul","elemwise_mul",["lhs","rhs"],nil,nil},
      {"elemwise_sub","elemwise_sub",["lhs","rhs"],nil,nil},
      {"exp","exp",["data"],nil,nil},
      {"expand_dims","expand_dims",["data"],["axis"],nil},
      {"expm1","expm1",["data"],nil,nil},
      {"fill_element_0index","fill_element_0index",nil,nil,nil},
      {"fix","fix",["data"],nil,nil},
      {"flatten","Flatten",["data"],nil,nil},
      {"flip","reverse",["data"],["axis"],nil},
      {"floor","floor",["data"],nil,nil},
      {"ftml_update","ftml_update",["weight","grad","d","v","z"],["lr","t"],["beta1","beta2","epsilon","wd","rescale_grad","clip_grad"]},
      {"ftrl_update","ftrl_update",["weight","grad","z","n"],["lr"],["lamda1","beta","wd","rescale_grad","clip_gradient"]},
      {"gamma","gamma",["data"],nil,nil},
      {"gammaln","gammaln",["data"],nil,nil},
      {"gather_nd","gather_nd",["data","indices"],nil,nil},
      {"hard_sigmoid","hard_sigmoid",["data"],nil,["alpha","beta"]},
      {"identity","_copy",["data"],nil,nil},
      {"khatri_rao","khatri_rao",["*args"],nil,nil},
      {"linalg_gelqf","_linalg_gelqf",["A"],nil,nil},
      {"linalg_gemm","_linalg_gemm",["A","B","C"],nil,["transpose_a","transpose_b","alpha","beta","axis"]},
      {"linalg_gemm2","_linalg_gemm2",["A","B"],nil,["transpose_a","transpose_b","alpha","axis"]},
      {"linalg_potrf","_linalg_potrf",["A"],nil,nil},
      {"linalg_potri","_linalg_potri",["A"],nil,nil},
      {"linalg_sumlogdiag","_linalg_sumlogdiag",["A"],nil,nil},
      {"linalg_syrk","_linalg_syrk",["A"],nil,["transpose","alpha"]},
      {"linalg_trmm","_linalg_trmm",["A","B"],nil,["transpose","rightside","alpha"]},
      {"linalg_trsm","_linalg_trsm",["A","B"],nil,["transpose","rightside","alpha"]},
      {"log","log",["data"],nil,nil},
      {"log10","log10",["data"],nil,nil},
      {"log1p","log1p",["data"],nil,nil},
      {"log2","log2",["data"],nil,nil},
      {"log_softmax","log_softmax",["data"],nil,["axis","temperature"]},
      {"logical_not","logical_not",["data"],nil,nil},
      {"make_loss","make_loss",["data"],nil,nil},
      {"max","max",["data"],nil,["axis","keepdims","exclude"]},
      {"max_axis","max",["data"],nil,["axis","keepdims","exclude"]},
      {"mean","mean",["data"],nil,["axis","keepdims","exclude"]},
      {"min","min",["data"],nil,["axis","keepdims","exclude"]},
      {"min_axis","min",["data"],nil,["axis","keepdims","exclude"]},
      {"mp_sgd_mom_update","mp_sgd_mom_update",["weight","grad","mom","weight32"],["lr"],["momentum","wd","rescale_grad","clip_gradient","lazy_update"]},
      {"mp_sgd_update","mp_sgd_update",["weight","grad","weight32"],["lr"],["wd","rescale_grad","clip_gradient","lazy_update"]},
      {"nanprod","nanprod",["data"],nil,["axis","keepdims","exclude"]},
      {"nansum","nansum",["data"],nil,["axis","keepdims","exclude"]},
      {"negative","negative",["data"],nil,nil},
      {"norm","norm",["data"],nil,["ord","axis","keepdims"]},
      {"normal","_random_normal",nil,nil,["loc","scale","shape","ctx","dtype"]},
      {"one_hot","one_hot",["indices"],["depth"],["on_value","off_value","dtype"]},
      {"ones_like","ones_like",["data"],nil,nil},
      {"pad","Pad",["data"],["mode","pad_width"],["constant_value"]},
      {"pick","pick",["data","index"],nil,["axis","keepdims"]},
      {"prod","prod",["data"],nil,["axis","keepdims","exclude"]},
      {"radians","radians",["data"],nil,nil},
      {"random_exponential","_random_exponential",nil,nil,["lam","shape","ctx","dtype"]},
      {"random_gamma","_random_gamma",nil,nil,["alpha","beta","shape","ctx","dtype"]},
      {"random_generalized_negative_binomial","_random_generalized_negative_binomial",nil,nil,["mu","alpha","shape","ctx","dtype"]},
      {"random_negative_binomial","_random_negative_binomial",nil,nil,["k","p","shape","ctx","dtype"]},
      {"random_normal","_random_normal",nil,nil,["loc","scale","shape","ctx","dtype"]},
      {"random_poisson","_random_poisson",nil,nil,["lam","shape","ctx","dtype"]},
      {"random_uniform","_random_uniform",nil,nil,["low","high","shape","ctx","dtype"]},
      {"ravel_multi_index","_ravel_multi_index",["data"],nil,["shape"]},
      {"rcbrt","rcbrt",["data"],nil,nil},
      {"reciprocal","reciprocal",["data"],nil,nil},
      {"relu","relu",["data"],nil,nil},
      {"repeat","repeat",["data"],["repeats"],["axis"]},
      {"reshape","Reshape",["data"],nil,["shape","reverse","target_shape","keep_highest"]},
      {"reshape_like","reshape_like",["lhs","rhs"],nil,nil},
      {"reverse","reverse",["data"],["axis"],nil},
      {"rint","rint",["data"],nil,nil},
      {"rmsprop_update","rmsprop_update",["weight","grad","n"],["lr"],["gamma1","epsilon","wd","rescale_grad","clip_gradient","clip_weights"]},
      {"rmspropalex_update","rmspropalex_update",["weight","grad","n","g","delta"],["lr"],["gamma1","gamma2","epsilon","wd","rescale_grad","clip_gradient","clip_weights"]},
      {"round","round",["data"],nil,nil},
      {"rsqrt","rsqrt",["data"],nil,nil},
      {"sample_exponential","_sample_exponential",["lam"],nil,["shape","dtype"]},
      {"sample_gamma","_sample_gamma",["alpha","beta"],nil,["shape","dtype"]},
      {"sample_generalized_negative_binomial","_sample_generalized_negative_binomial",["mu","alpha"],nil,["shape","dtype"]},
      {"sample_multinomial","_sample_multinomial",["data"],nil,["shape","get_prob","dtype"]},
      {"sample_negative_binomial","_sample_negative_binomial",["k","p"],nil,["shape","dtype"]},
      {"sample_normal","_sample_normal",["mu","sigma"],nil,["shape","dtype"]},
      {"sample_poisson","_sample_poisson",["lam"],nil,["shape","dtype"]},
      {"sample_uniform","_sample_uniform",["low","high"],nil,["shape","dtype"]},
      {"scatter_nd","scatter_nd",["data","indices"],["shape"],nil},
      {"sgd_mom_update","sgd_mom_update",["weight","grad","mom"],["lr"],["momentum","wd","rescale_grad","clip_gradient","lazy_update"]},
      {"sgd_update","sgd_update",["weight","grad"],["lr"],["wd","rescale_grad","clip_gradient","lazy_update"]},
      {"shape_array","shape_array",["data"],nil,nil},
      {"shuffle","_shuffle",["data"],nil,nil},
      {"sigmoid","sigmoid",["data"],nil,nil},
      {"sign","sign",["data"],nil,nil},
      {"signsgd_update","signsgd_update",["weight","grad"],["lr"],["wd","rescale_grad","clip_gradient"]},
      {"signum_update","signum_update",["weight","grad","mom"],["lr"],["momentum","wd","rescale_grad","clip_gradient","wd_lh"]},
      {"sin","sin",["data"],nil,nil},
      {"sinh","sinh",["data"],nil,nil},
      {"size_array","size_array",["data"],nil,nil},
      {"slice","slice",["data"],["begin","end"],["step"]},
      {"slice_axis","slice_axis",["data"],["axis","begin","end"],nil},
      {"slice_like","slice_like",["data","shape_like"],nil,["axes"]},
      {"smooth_l1","smooth_l1",["data"],["scalar"],nil},
      {"softmax","softmax",["data"],nil,["axis","temperature"]},
      {"softmax_cross_entropy","softmax_cross_entropy",["data","label"],nil,nil},
      {"softsign","softsign",["data"],nil,nil},
      {"sort","sort",["data"],nil,["axis","is_ascend"]},
      {"space_to_depth","space_to_depth",["data"],["block_size"],nil},
      {"split","SliceChannel",["data"],["num_outputs"],["axis","squeeze_axis"]},
      {"sqrt","sqrt",["data"],nil,nil},
      {"square","square",["data"],nil,nil},
      {"squeeze","squeeze",["*data"],nil,["axis"]},
      {"stack","stack",["*data"],["num_args"],["axis"]},
      {"stop_gradient","BlockGrad",["data"],nil,nil},
      {"sum","sum",["data"],nil,["axis","keepdims","exclude"]},
      {"sum_axis","sum",["data"],nil,["axis","keepdims","exclude"]},
      {"swapaxes","SwapAxis",["data"],nil,["dim1","dim2"]},
      {"take","take",["a","indices"],nil,["axis","mode"]},
      {"tan","tan",["data"],nil,nil},
      {"tanh","tanh",["data"],nil,nil},
      {"tile","tile",["data"],["reps"],nil},
      {"topk","topk",["data"],nil,["axis","k","ret_typ","is_ascend"]},
      {"transpose","transpose",["data"],nil,["axes"]},
      {"trunc","trunc",["data"],nil,nil},
      {"uniform","_random_uniform",nil,nil,["low","high","shape","ctx","dtype"]},
      {"unravel_index","_unravel_index",["data"],nil,["shape"]},
      {"where","where",["condition","x","y"],nil,nil},
      {"zeros_like","zeros_like",["data"],nil,nil},
    }

    private macro extended
      class {{ @type }}::Ops end
      class {{ @type }}::Internal end
      class {{ @type }}::Contrib end
      class {{ @type }}::Linalg end
      class {{ @type }}::Sparse end

      {% for op in MXNet::Operations::OP_INFO %}
        {% keywords = {"begin", "end"} %}
        {% name = op[0].gsub(/^(_contrib_|_linalg_|_sparse_|_)/, "") %}
        {% prefix = {"_contrib_", "_linalg_", "_sparse_", "_"}.find { |pre| op[0].starts_with?(pre) } || "" %}
        {% mod = {"_contrib_": "Contrib", "_linalg_": "Linalg", "_sparse_": "Sparse", "_": "Internal"}[prefix] || "Ops" %}
        {% args = op[2] ? op[2].map { |a| keywords.includes?(a) ? "_#{a.downcase.id} : #{@type}".id : "#{a.downcase.id} : #{@type}".id } : nil %}
        {% kwargs = op[3] ? op[3].map { |a| keywords.includes?(a) ? "#{a.downcase.id} _#{a.downcase.id}".id : "#{a.downcase.id}".id } : nil %}
        {% if args && kwargs %}
          def {{ @type }}::{{ mod.id }}.{{ "_#{name.id}".id }}({{ *args }}, {{ *kwargs }}, **kwargs)
            {% args = op[2].map { |a| keywords.includes?(a) ? "_#{a.downcase.id}".id : a.downcase.id } %}
            {% kwargs = op[3].map { |a| keywords.includes?(a) ? "#{a.downcase.id}: _#{a.downcase.id}".id : "#{a.downcase.id}: #{a.downcase.id}".id } %}
            {% if @type == MXNet::NDArray %}
              {{ @type }}.imperative_invoke({{ op[0] }}, {{ *args }}, **kwargs.merge({{ *kwargs }}))
            {% elsif @type == MXNet::Symbol %}
              {{ @type }}.create_symbol({{ op[0] }}, {{ *args }}, **kwargs.merge({{ *kwargs }}))
            {% end %}
          end
        {% elsif args %}
          def {{ @type }}::{{ mod.id }}.{{ "_#{name.id}".id }}({{ *args }}, **kwargs)
            {% args = op[2].map { |a| keywords.includes?(a) ? "_#{a.downcase.id}".id : a.downcase.id } %}
            {% if @type == MXNet::NDArray %}
              {{ @type }}.imperative_invoke({{ op[0] }}, {{ *args }}, **kwargs)
            {% elsif @type == MXNet::Symbol %}
              {{ @type }}.create_symbol({{ op[0] }}, {{ *args }}, **kwargs)
            {% end %}
          end
        {% elsif kwargs %}
          def {{ @type }}::{{ mod.id }}.{{ "_#{name.id}".id }}({{ *kwargs }}, **kwargs)
            {% kwargs = op[3].map { |a| keywords.includes?(a) ? "#{a.downcase.id}: _#{a.downcase.id}".id : "#{a.downcase.id}: #{a.downcase.id}".id } %}
            {% if @type == MXNet::NDArray %}
              {{ @type }}.imperative_invoke({{ op[0] }}, **kwargs.merge({{ *kwargs }}))
            {% elsif @type == MXNet::Symbol %}
              {{ @type }}.create_symbol({{ op[0] }}, **kwargs.merge({{ *kwargs }}))
            {% end %}
          end
        {% else %}
          def {{ @type }}::{{ mod.id }}.{{ "_#{name.id}".id }}(**kwargs)
            {% if @type == MXNet::NDArray %}
              {{ @type }}.imperative_invoke({{ op[0] }}, **kwargs)
            {% elsif @type == MXNet::Symbol %}
              {{ @type }}.create_symbol({{ op[0] }}, **kwargs)
            {% end %}
          end
        {% end %}
      {% end %}
    end

    private macro def_class_and_fluent_methods(op, name, *args)
      {% if args.size > 0 %}
        def self.{{name}}(obj, {{ *args.map { |a| [:begin, :end].includes?(a) ? "#{a.id} _#{a.id}".id : a.id } }}, **kwargs)
          {{op}}._{{name}}(obj, {{ *args.map { |a| [:begin, :end].includes?(a) ? "_#{a.id}".id : a.id } }}, **kwargs)
        end
        # Convenience fluent method for `.{{name}}`.
        def {{name}}({{ *args.map { |a| [:begin, :end].includes?(a) ? "#{a.id} _#{a.id}".id : a.id } }}, **kwargs)
          {{@type}}.{{name}}(self, {{ *args.map { |a| [:begin, :end].includes?(a) ? "_#{a.id}".id : a.id } }}, **kwargs)
        end
      {% else %}
        def self.{{name}}(obj, **kwargs)
          {{op}}._{{name}}(obj, **kwargs)
        end
        # Convenience fluent method for `.{{name}}`.
        def {{name}}(**kwargs)
          {{@type}}.{{name}}(self, **kwargs)
        end
      {% end %}
    end

    private macro included
      {%
        if @type == MXNet::NDArray
          prefix =
            "# * *data* (`NDArray`)
             #   Input data.".id
          suffix =
            "# * *out* (`NDArray`, optional)
             #   The output array.".id
        elsif @type == MXNet::Symbol
          prefix =
            "# * *data* (`Symbol`)
             #   Input data.".id
          suffix =
            "# * *name* (`String`, optional)
             #   Name of the resulting symbol.".id
        else
          prefix = "".id
          suffix = "".id
        end
      %}

      # Reshapes the input array.
      #
      # Returns a copy of the array with a new shape without altering
      # any data.
      #
      # Assume `x` is an array with the following elements:
      #     [1, 2, 3, 4]
      #
      # Then:
      #     reshape(shape: [2, 2]) # => [[1, 2], [3, 4]]
      #
      # Some dimensions of the shape can take special values from the
      # set `{0, -1, -2, -3, -4}`. The significance of each is explained
      # below:
      #
      # * `0` copies this dimension from the input to the output shape:
      #     zeros([2, 3, 4]).reshape([4, 0, 2]).shape # => [4, 3, 2]
      #     zeros([2, 3, 4]).reshape([2, 0, 0]).shape # => [2, 3, 4]
      # * `-1` infers the dimension of the output shape by using the
      #   remainder of the input dimensions, keeping the size of the
      #   new array the same as that of the input array. At most one
      #   dimension can be `-1`:
      #     zeros([2, 3, 4]).reshape([6, 1, -1]).shape # => [6, 1, 4]
      #     zeros([2, 3, 4]).reshape([3, -1, 8]).shape # => [3, 1, 8]
      #     zeros([2, 3, 4]).reshape([-1]).shape # => [24]
      # * `-2` copies all/the remainder of the input dimensions to the
      #   output shape:
      #     zeros([2, 3, 4]).reshape([-2]).shape # => [2, 3, 4]
      #     zeros([2, 3, 4]).reshape([2, -2]).shape # => [2, 3, 4]
      #     zeros([2, 3, 4]).reshape([-2, 1, 1]).shape # => [2, 3, 4, 1, 1]
      # * `-3` uses the product of two consecutive dimensions of the
      #   input shape as the output dimension:
      #     zeros([2, 3, 4]).reshape([-3, 4]).shape # => [6, 4]
      #     zeros([2, 3, 4, 5]).reshape([-3, -3]).shape # => [6, 20]
      #     zeros([2, 3, 4]).reshape([0, -3]).shape # => [2, 12]
      #     zeros([2, 3, 4]).reshape([-3, -2]).shape # => [6, 4]
      # * `-4` splits one dimension of the input into the two dimensions
      #   passed subsequent to `-4` (which can contain `-1`):
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
      def_class_and_fluent_methods(Ops, reshape)

      # Flattens the input array into a 2-D array by collapsing the
      # higher dimensions.
      #
      # For an input array with shape `(d1, d2, ..., dk)`, `#flatten`
      # reshapes the input array into an output array of shape
      # `(d1, d2 * ... * dk)`.
      #
      # Note that the bahavior of this function is different from
      # `Array#flatten`, which behaves similar to `#reshape([-1])`.
      #
      # Assume `x` is an array with the following elements:
      #     [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]
      #
      # Then:
      #     flatten(x).shape # => [2, 6]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, flatten)

      # Inserts a new axis of size 1 into the array shape.
      #
      # For example, given `x` with shape `[2, 3, 4]`, then
      # `expand_dims(x, axis: 1)` will return a new array with shape
      # `[2, 1, 3, 4]`.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, required)
      #   Position where new axis is to be inserted. Suppose that the
      #   input array‘s dimension is `ndim`, the range of the inserted
      #   axis is `[-ndim, ndim]`.
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, expand_dims, :axis)

      # Computes the mean of array elements over given axes.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, `Array(Int)`, optional)
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
      def_class_and_fluent_methods(Ops, mean)

      # Computes the max of array elements over given axes.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, `Array(Int)`, optional)
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
      def_class_and_fluent_methods(Ops, max)

      # Computes the min of array elements over given axes.
      #
      # ### Parameters
      {{prefix}}
      # * *axis* (`Int`, `Array(Int)`, optional)
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
      def_class_and_fluent_methods(Ops, min)

      # Permutes the dimensions of an array.
      #
      # Assume `x` and `y` are arrays with the following elements:
      #     [[[1, 2], [3, 4], [5, 6], [7, 8]]] # x
      #     [[1, 2], [3, 4]] # y
      #
      # Then:
      #     transpose(x) # => [[[1], [3], [5], [7]], [[2], [4], [6], [8]]]
      #     transpose(x, axes: [1, 0, 2]) # => [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]
      #     transpose(y) # => [[1, 3], [2, 4]]
      #
      # ### Parameters
      {{prefix}}
      # * *axes* (`Int`, `Array(Int)`, optional)
      #   Target axis order. By default the axes will be inverted.
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, transpose)

      # Reverses the order of elements along given axis while preserving array shape.
      #
      # Assume `x` is an array with the following elements:
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
      def_class_and_fluent_methods(Ops, flip, :axis)

      # Returns element-wise square-root value of the input.
      #
      # Assume `x` is an array with the following elements:
      #     [4, 9, 16]
      #
      # Then:
      #     sqrt(x) # => [2, 3, 4]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, sqrt)

      # Returns element-wise squared value of the input.
      #
      # Assume `x` is an array with the following elements:
      #     [2, 3, 4]
      #
      # Then:
      #     square(x) # => [4, 9, 16]
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, square)

      # Computes the rectified linear activation.
      #
      # _y=max(input,0)_
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, relu)

      # Computes the sigmoid activation.
      #
      # _y=1/(1+exp(−x))_
      #
      # ### Parameters
      {{prefix}}
      {{suffix}}
      #
      def_class_and_fluent_methods(Ops, sigmoid)

      # Slices a region of the array.
      #
      # This function returns a sliced array between the indices given
      # by *begin* and *end* with the corresponding *step*.
      #
      # For an input array of `shape=[d_0, d_1, ..., d_n-1]`, a slice
      # operation with `begin=[b_0, b_1, ..., b_m-1]`, `end=[e_0, e_1,
      # ..., e_m-1]`, and `step=[s_0, s_1, ..., s_m-1]`, where `m <= n`,
      # results in an array with the shape `(|e_0-b_0|/|s_0|, ...,
      # |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)`.
      #
      # The resulting array’s _k_-th dimension contains elements from
      # the _k_-th dimension of the input array starting from index
      # `b_k` (inclusive) with step `s_k` until reaching `e_k`
      # (exclusive).
      #
      # If the _k_-th elements are `nil` in the sequence of *begin*,
      # *end*, and *step*, the following rule will be used to set
      # default values: if `s_k` is `nil`, set `s_k=1`. If `s_k > 0`,
      # set `b_k=0`, `e_k=d_k`, else set `b_k=d_k-1`, `e_k=-1`.
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
      def_class_and_fluent_methods(Ops, slice, :begin, :end)

      # Slices along a given axis.
      #
      # Returns an array slice along a given *axis* starting from the
      # *begin* index to the *end* index.
      #
      # Assume `x` is an array with the following elements:
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
      def_class_and_fluent_methods(Ops, slice_axis, :axis, :begin, :end)

      # Computes the dot product of two arrays.
      #
      # `dot`‘s behavior depends on the input array dimensions:
      #   * 1-D arrays: inner product of vectors
      #   * 2-D arrays: matrix multiplicationO
      #   * N-D arrays: a sum product over the last axis of the first
      #   input and the first axis of the second input
      #
      # Assume `x` and `y` are arrays with the following elements:
      #     [[1, 2], [3, 4]] # x
      #     [[4, 3], [1, 1]] # y
      #
      # Then:
      #     dot(x, y) # => [[8, 5], [20, 13]]
      #
      # ### Parameters
      # * *lhs* (`{{@type.stringify.split("::").last.id}}`, required)
      #   The first input.
      # * *rhs* (`{{@type.stringify.split("::").last.id}}`, required)
      #   The second input.
      # * *transpose_a* (`Bool`, default = false)
      #   If true then transpose the first input before dot.
      # * *transpose_b* (`Bool`, default = false)
      #   If true then transpose the second input before dot.
      {{suffix}}
      #
      def self.dot(lhs : self, rhs : self, **kwargs)
        Ops._dot(lhs, rhs, **kwargs)
      end

      # Joins input arrays along a given axis.
      #
      # The dimensions of the input arrays should be the same except
      # for the axis along which they will be concatenated. The
      # dimension of the output array along the concatenated axis will
      # be equal to the sum of the corresponding dimensions of the
      # input arrays.
      #
      # Assume `x` and `y` are arrays with the following elements:
      #     [[1, 2], [3, 4]] # x
      #     [[1, 4], [1, 1]] # y
      #
      # Then:
      #     concat(x, y) # => [[1, 2, 1, 4], [3, 4, 1, 1]]
      #
      # ### Parameters
      # * *data* (`Array({{@type.stringify.split("::").last.id}})`, required)
      #   List of arrays to concatenate.
      # * *dim* (`Int`, default = 1)
      #   The dimension to be concated.
      {{suffix}}
      #
      def self.concat(*data : self, **kwargs)
        Ops._concat(*data, **kwargs.merge({num_args: data.size}))
      end

      # Adds all input arguments element-wise.
      #
      # _add_n(a1,a2,...,an)=a1+a2+...+an_
      #
      # `add_n` is potentially more efficient than calling `add` _n_ times.
      #
      # ### Parameters
      # * *data* (`Array({{@type.stringify.split("::").last.id}})`, required)
      #   List of arrays to add.
      {{suffix}}
      #
      def self.add_n(*data : self, **kwargs)
        Ops._add_n(*data, **kwargs.merge({num_args: data.size}))
      end

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
      def self.shuffle(data : self, **kwargs)
        Ops._shuffle(data, **kwargs)
      end
    end
  end
end
