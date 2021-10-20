from yacs.config import CfgNode as CN
import os
import torch
import multiprocessing


def get_num_workers():
    NUM_GPUS = torch.cuda.device_count()
    NUM_CPUS = multiprocessing.cpu_count()
    if NUM_GPUS == 0:
        raise ValueError("you need gpus!")
    num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2 * NUM_GPUS) - 1
    print("NUM_GPUS: ", NUM_GPUS, "; NUM_CPUS: ", NUM_CPUS, "; num_workers: ", num_workers)
    return num_workers

# root_path = '../..'
root_path = '../../generate-it'

_C = CN()
_C.exp_id = ''
# _C.root_path = '../..'
_C.device = 'cuda'
_C.distributed = True
_C.fp16 = 1
_C.opt_level = 'O1'
_C.loss_scale = 128
_C.log_time = 400
_C.checkpoint_time = 10000
_C.test_interval = 10000
_C.save_dir = os.path.join(root_path, 'models')

_C.img_save_dir = os.path.join(root_path, 'images')
_C.gt_img_path = os.path.join(root_path, 'images', 'imgs_karpathy')

_C.data_dir_feat = os.path.join(root_path, 'data/samples')
_C.train_split = 'trainrestval'
_C.file_train_data = os.path.join(_C.data_dir_feat, f'{_C.train_split}_samples.pth')

_C.data_dir_anno = os.path.join(root_path, 'annotations')
_C.id2captions_test = os.path.join(_C.data_dir_anno, 'id2captions_test.json')
_C.test_samples = os.path.join(_C.data_dir_anno, 'test_samples.json')
_C.dataset_coco = os.path.join(_C.data_dir_anno, 'dataset_coco.json')
_C.test_gts_dict = os.path.join(_C.data_dir_anno, 'test_gts_dict.json')
_C.karpathy_test_result_example = os.path.join(_C.data_dir_anno, 'karpathy_test_result_example.json')

# feature from https://github.com/allenai/x-lxmert/blob/master/feature_extraction/README.md
clustering_dir = os.path.join(root_path, 'data/clustering/')
_C.grid_cluster_id_path_train = os.path.join(clustering_dir,
        'maskrcnn_mscoco_train_mscoco_train_img_id_to_cluster_id_10000_iter20_d2048_grid8.pkl')
_C.grid_cluster_id_path_valid = os.path.join(clustering_dir,
        'maskrcnn_mscoco_train_mscoco_valid_img_id_to_cluster_id_10000_iter20_d2048_grid8.pkl')
_C.grid_cluster_centroids_path = os.path.join(clustering_dir,
        'maskrcnn_mscoco_train_centroids10000_iter20_d2048_grid8.npy')

'''
# Grid features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_train_grid8.h5 -P grid_features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_valid_grid8.h5 -P grid_features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_test_grid8.h5 -P grid_features
'''
features_dir = os.path.join(root_path, 'data/grid_features/')
_C.grid_feat_path_train = os.path.join(features_dir, 'maskrcnn_train_grid8.h5')
_C.grid_feat_path_valid = os.path.join(features_dir, 'maskrcnn_valid_grid8.h5')

_C.num_workers = get_num_workers()
_C.samples_per_gpu = 32
_C.model_path = 'resume_from_latest_or_specific_path'
_C.resume_from_latest = 1
_C.load_model_only = 0  # if 1, only load the model from model_path, otherwise will also load optimizer and scheduler
_C.restart_iteration = 0  # if 1, restart the iteration to 1; if 0, will load the iteration from model_path

_C.pretrained_bert = os.path.join(root_path, 'models/pretrained/bert.pth')
_C.pretrained_generator = os.path.join(root_path, 'models/pretrained/G_60.pth')


_C.solver = CN()
_C.solver.lr = 5e-5
_C.solver.weight_decay = 1e-2
_C.solver.betas = (0.9, 0.999)
_C.solver.eps = 1e-6
_C.solver.grad_clip = 1.0

_C.scheduler = CN()
_C.scheduler.method = 'WarmupCosineSchedule'
_C.scheduler.warmup_steps = 10000
_C.scheduler.max_steps = 200000

_C.nonautoregressize_steps = 20

_C.num_hidden_layers = 12

# the number of (grids, caption(s)) pairs in training, set to 1 or 5
_C.num_sample_captions = 5

_C.autoregressive = 0

# unlikelihood training, refer to https://arxiv.org/pdf/1908.04319.pdf
_C.unlikelihood_training = 1
_C.rank_alpha = 5.0
# whether to use the frequency of tokens as weights in unlikelihood loss
# 1: use freq; -1: use 1/freq
_C.use_freq_weight = 1
# when using unlikelihood loss
# []: meaningless setting
# [1]/[2]: penalize 1-gram/2-gram repetition
# [1, 2]: penalize 1-gram and 2-gram repetition;
# [1, 2]: use_freq_weight should not be zero under this setting, otherwise the 2-gram penalty would not take effect
_C.penalize_ngram_repetition = [1]

_C.text_length = 16

_C.iteration_tasks = ['i2t','t2i']

# grid-level features: extracted with the butd faster rcnn detector, but with the grid boxes instead of region boxes
_C.use_mix_feature = 1  # when set to 1: dense+discrete features, use with use_grid_cluster 1 use_grid_feat 1
_C.use_grid_cluster = 1
_C.use_grid_feat = 1
_C.grid_cluster_num = 10000
_C.grid_size = 8
_C.grid_feat_dim = 2048

# Number of updates steps to accumulate before performing a backward/update pass.
_C.gradient_accumulation_steps = 1

_C.seed = 123

# -------------------------------------- Testing ----------------------------------
_C.custom_generation = 0
# setting "custom_generation 0" equals to setting
# "custom_generation 1 consider_generated_sentence 0 do_sample False num_beams 1 num_beam_groups 1 repetition_penalty 1.0"
# generation
_C.max_length = 25
_C.do_sample = True
_C.num_return_sequences = 1

# _get_logits_warper
_C.top_k = 3
_C.top_p = 0.8
_C.temperature = 0.75

# _get_logits_processor
_C.consider_generated_sentence = 0  # no bother for diverse group beam search
# 1: input all previous generated captions when generating the next token
# 0: input generated tokens of current captions when generating the next token
_C.repetition_penalty = 1.5
_C.no_repeat_ngram_size = 0
_C.bad_words_ids = None
_C.min_length = 7
_C.eos_token_id = 102
_C.prefix_allowed_tokens_fn = None
_C.diversity_penalty = 0.0

# beam search
_C.print_beams = 0
_C.num_beams = 1
_C.num_beam_groups = 1
_C.length_penalty = 1.0
_C.early_stopping = False

_C.inference_postfix = None
_C.only_eval_cider = True

_C.demo_captions = False

