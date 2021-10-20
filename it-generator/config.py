from yacs.config import CfgNode as CN
import os
import torch
import multiprocessing


def get_num_workers():
    NUM_GPUS = torch.cuda.device_count()
    NUM_CPUS = multiprocessing.cpu_count()
    if NUM_GPUS == 0:
        raise ValueError("you need gpus!")
    num_workers = min(2, (4 * NUM_GPUS if NUM_CPUS == 32 else 2 * NUM_GPUS) - 1)
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
_C.loss_scale = -1
_C.log_time = 400
_C.checkpoint_time = 10000
_C.test_interval = 10000
_C.save_dir = os.path.join(root_path, 'models')
_C.img_save_dir = os.path.join(root_path, 'images')
_C.gt_img_dir_val2014 = os.path.join(root_path, 'images', 'coco_val2014/')
_C.gt_img_dir_5k = os.path.join(root_path, 'images',  'gt_images_5k')
_C.gt_img_dir_3w = os.path.join(root_path, 'images',  'gt_images_3w')
_C.data_dir_feat = os.path.join(root_path, 'data/samples')
_C.train_split = 'train'
_C.file_train_data = os.path.join(_C.data_dir_feat, f'{_C.train_split}_samples.pth')

_C.data_dir_anno = os.path.join(root_path, 'annotations')
_C.id2captions_test = os.path.join(_C.data_dir_anno, 'id2captions_test.json')
_C.test_samples = os.path.join(_C.data_dir_anno, 'test_samples.json')
_C.dataset_coco = os.path.join(_C.data_dir_anno, 'dataset_coco.json')
_C.test_gts_dict = os.path.join(_C.data_dir_anno, 'test_gts_dict.json')
_C.karpathy_test_result_example = os.path.join(_C.data_dir_anno, 'karpathy_test_result_example.json')
_C.rprec_hard_samples_dir = os.path.join(_C.data_dir_anno, 'rprec_hard_samples')

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
_C.load_model_only = 0  # if 1, only load the model from model_path, otherwise will also load optimizer and scheduler
_C.resume_from_latest = 1
_C.restart_iteration = 0  # if 1, restart the iteration to 1; if 0, will load the iteration from model_path

_C.pretrained_model = os.path.join(root_path, 'models/pretrained/Epoch20_LXRT.pth')
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

_C.text_length = 17

_C.grid_feat_dim = 2048

# Number of updates steps to accumulate before performing a backward/update pass.
# I use it for run i2t and t2i in two iteration forwards and then optimizer.backward
_C.gradient_accumulation_steps = 1

_C.use_mix_feature = 1  # when set to 1: dense+discrete features, use with use_grid_cluster 1 use_grid_feat 1

# grid-level features: extracted with the butd faster rcnn detector, but with the grid boxes instead of region boxes
_C.use_grid_cluster = 1
_C.grid_cluster_num = 10000
_C.grid_size = 8

_C.use_grid_feat = 1

_C.iteration_tasks = ['xe_t2i','xe_i2t']  # 'xe_i2t', 'xe_t2i', 'rl_i2t', 'rl_t2i'

_C.mse_loss = 0
_C.use_clip_score = 1

_C.seed = 123

_C.test_num = 499

_C.xlxmert_zeroshot = 0  # zero-shot inference with x-xlxmert's pre-trained model

# 'train': only use train2014, the standard train split for text-to-image generation on MSCOCO
_C.test_val2014 = 1

_C.eval_fid = 1
_C.eval_is = 0
_C.eval_clipscore = 1
_C.eval_rprec = 0
_C.eval_rprec_hard = 0

_C.evaluate_images_dir = ''  # different from config.img_save_dir

