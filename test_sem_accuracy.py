import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from SSR.datasets.replica import replica_datasets
from SSR.datasets.scannet import scannet_datasets
from SSR.datasets.replica_nyu import replica_nyu_cnn_datasets
from SSR.datasets.scannet import scannet_datasets
import open3d as o3d

from SSR.training import trainer
from SSR.models.model_utils import run_network
from SSR.geometry.occupancy import grid_within_bound
from SSR.visualisation import open3d_utils
import numpy as np
import yaml
import json

import skimage.measure as ski_measure
import time
from imgviz import label_colormap
import trimesh

from sklearn.metrics import confusion_matrix

@torch.no_grad()
def render_fn(trainer, rays, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            trainer.render_rays(rays[i:i+chunk])

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="/home/shuaifeng/Documents/PhD_Research/CodeRelease/SemanticSceneRepresentations/SSR/configs/SSR_room0_config_test.yaml", help='config file name.')

    # parser.add_argument('--mesh_dir', type=str, required=True, help='Path to scene file, e.g., ROOT_PATH/Replica/mesh/room_0/')
    parser.add_argument('--save_dir', type=str, required=False, help='Path to the directory saving training logs and ckpts.')
    parser.add_argument('--ckpt', type=str, required=False, help='Path to rendered data.')

    # sparse-views
    parser.add_argument("--sparse_views", action='store_true',
                        help='Use labels from a sparse set of frames')
    parser.add_argument("--sparse_ratio", type=float, default=0,
                        help='The portion of dropped labelling frames during training, which can be used along with all working modes.')    
    parser.add_argument("--label_map_ids", nargs='*', type=int, default=[],
                        help='In sparse view mode, use selected frame ids from sequences as supervision.')
    parser.add_argument("--random_sample", action='store_true', help='Whether to randomly/evenly sample frames from the sequence.')

    # denoising---pixel-wsie
    parser.add_argument("--pixel_denoising", action='store_true',
                        help='Whether to work in pixel-denoising tasks.')
    parser.add_argument("--pixel_noise_ratio", type=float, default=0,
                        help='In sparse view mode, if pixel_noise_ratio > 0, the percentage of pixels to be perturbed in each sampled frame  for pixel-wise denoising task..')
                        
    # denoising---region-wsie
    parser.add_argument("--region_denoising", action='store_true',
                        help='Whether to work in region-denoising tasks by flipping class labels of chair instances in Replica Room_2')
    parser.add_argument("--region_noise_ratio", type=float, default=0,
                        help='In region-wise denoising task, region_noise_ratio is the percentage of chair instances to be perturbed in each sampled frame for region-wise denoising task.')
    parser.add_argument("--uniform_flip", action='store_true',
                        help='In region-wise denoising task, whether to change chair labels uniformly or not, i.e., by ascending area ratios. This corresponds to two set-ups mentioned in the paper.')
    parser.add_argument("--instance_id", nargs='*', type=int, default=[3, 6, 7, 9, 11, 12, 13, 48],
                        help='In region-wise denoising task, the chair instance ids in Replica Room_2 to be randomly perturbed. The ids of all 8 chairs are [3, 6, 7, 9, 11, 12, 13, 48]')
       
    # super-resolution
    parser.add_argument("--super_resolution", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument('--dense_sr',  action='store_true', help='Whether to use dense or sparse labels for SR instead of dense labels.')
    parser.add_argument('--sr_factor',  type=int, default=8, help='Scaling factor of super-resolution.')

    # label propagation
    parser.add_argument("--label_propagation", action='store_true',
                        help='Label propagation using partial seed regions.')
    parser.add_argument("--partial_perc", type=float, default=0,
                        help='0: single-click propagation; 1: using 1-percent sub-regions for label propagation, 5: using 5-percent sub-regions for label propagation')

    # misc.
    parser.add_argument('--visualise_save',  action='store_true', help='whether to save the noisy labels into harddrive for later usage')
    parser.add_argument('--load_saved',  action='store_true', help='use trained noisy labels for training to ensure consistency betwwen experiments')
    parser.add_argument('--gpu', type=str, default="", help='GPU IDs.')



    args = parser.parse_args()
    args.config_file = "/home/ps/hhy/semantic_nerf/SSR/configs/SSR_scene0_mydata_bfpose_config_10frame.yaml"
    args.save_dir = "/media/ps/passport1/hhy_data/NeRF/ScanNet/scans/scene0000_00_mydata_bfpose/logs_10frame/"
    args.ckpt = "200000.ckpt"
    args.dataset_type = "scannet"

    config_file_path = args.config_file

    # Read YAML file
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(args.gpu)>0:
        config["experiment"]["gpu"] = args.gpu
    print("Experiment GPU is {}.".format(config["experiment"]["gpu"]))
    trainer.select_gpus(config["experiment"]["gpu"])
        
    # Cast intrinsics to right types
    ssr_trainer = trainer.SSRTrainer(config)

    if args.dataset_type == "replica":
        total_num = 900
        step = 5

        # 10 frame
        if "complete" not in args.config_file: 
            train_ids = list(range(0, total_num, step))[:10]
            test_ids = [x+step//2 for x in list(range(0, total_num, step))]
        # 10 frame complete
        if "complete" in args.config_file: 
            train_ids = list(range(0, total_num, 90))
            test_ids = [x+step//2 for x in list(range(0, total_num, step))]
        # 180 frame
        # train_ids = list(range(0, total_num, step))
        # test_ids = [x+step//2 for x in list(range(0, total_num, step))]   

        #add ids to config for later saving.
        config["experiment"]["train_ids"] = train_ids
        config["experiment"]["test_ids"] = test_ids

        replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=config["experiment"]["dataset_dir"],
                                                                    train_ids=train_ids, test_ids=test_ids,
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"])

        ssr_trainer.set_params_replica()
        ssr_trainer.prepare_data_replica(replica_data_loader)
    
    elif args.dataset_type == "scannet":
        print("----- ScanNet Dataset with NYUv2-40 Conventions-----")

        print("processing ScanNet scene: ", os.path.basename(config["experiment"]["dataset_dir"]))
        # Todo: like nerf, creating sprial/test poses. Make training and test poses/ids interleaved
        scannet_data_loader = scannet_datasets.ScanNet_Dataset( scene_dir=config["experiment"]["dataset_dir"],
                                                                    img_h=config["experiment"]["height"],
                                                                    img_w=config["experiment"]["width"],
                                                                    sample_step=config["experiment"]["sample_step"],
                                                                    save_dir=config["experiment"]["dataset_dir"],
                                                                    args = args)

        ssr_trainer.set_params_scannet(scannet_data_loader)
        ssr_trainer.prepare_data_scannet(scannet_data_loader)

    ##########################

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr()
    # Create rays in world coordinates
    ssr_trainer.init_rays()

    # load_ckpt into NeRF
    ckpt_path = os.path.join(args.save_dir, "checkpoints", args.ckpt)
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)
    testsavedir = os.path.join(ssr_trainer.config["experiment"]["save_dir"], "test", 'step_{:s}'.format(args.ckpt.split(".")[0]))
    os.makedirs(testsavedir, exist_ok=True)

    start = ckpt['global_step']
    ssr_trainer.ssr_net_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    ssr_trainer.ssr_net_fine.load_state_dict(ckpt['network_fine_state_dict'])
    ssr_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ssr_trainer.training = False  # enable testing mode before rendering results, need to set back during training!
    ssr_trainer.ssr_net_coarse.eval()
    ssr_trainer.ssr_net_fine.eval()
    with torch.no_grad():
        rgbs, disps, deps, vis_deps, sems, vis_sems, sem_uncers, vis_sem_uncers = \
                                ssr_trainer.render_path(ssr_trainer.rays_test, save_dir=testsavedir)
        if ssr_trainer.enable_semantic:
            # mask out void regions for better visualisation
            for idx in range(vis_sems.shape[0]):
                vis_sems[idx][ssr_trainer.test_semantic_scaled[idx]==ssr_trainer.ignore_label, :] = 0
                
    depth_metrics_dic = calculate_depth_metrics(depth_trgt=ssr_trainer.test_depth_scaled, depth_pred=deps)  

    miou_test, miou_test_validclass, total_accuracy_test, class_average_accuracy_test, ious_test = \
        calculate_segmentation_metrics(true_labels=ssr_trainer.test_semantic_scaled, predicted_labels=sems, 
        number_classes=ssr_trainer.num_valid_semantic_class, ignore_label=ssr_trainer.ignore_label,
        semantic_uncers=sem_uncers, depth_trgt=ssr_trainer.test_depth_scaled, depth_pred=deps)

def calculate_depth_metrics(depth_trgt, depth_pred):
    """ Computes 2d metrics between two depth maps
    
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt<10) * (depth_trgt>0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred-depth_trgt)
    abs_rel = abs_diff/depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff/depth_trgt
    sq_log_diff = (np.log(depth_pred)-np.log(depth_trgt))**2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25**2).astype('float')
    r3 = (thresh < 1.25**3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, \
                                                ignore_label, semantic_uncers, depth_trgt, depth_pred):
    if (true_labels == ignore_label).all():
        return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    semantic_uncers = semantic_uncers.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    # uncertainty for error correction
    certain_correction_accuracy = {}
    certain_correction_miou = {}
    semantic_uncers = semantic_uncers[valid_pix_ids]
    for i in range(0, 10):
        certain_pix_ids = semantic_uncers < 10**(-i)
        r = 1 - np.mean(certain_pix_ids.astype('float'))
        predicted_cer = predicted_labels[certain_pix_ids]
        true_cer = true_labels[certain_pix_ids]
        conf_mat = confusion_matrix(true_cer, predicted_cer, labels=list(range(number_classes)))
        norm_conf_mat = np.transpose(
            np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))
        missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        exsiting_class_mask = ~ missing_class_mask
        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(number_classes)
        for class_id in range(number_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                    np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                    conf_mat[class_id, class_id]))
        miou = nanmean(ious)
        certain_correction_accuracy[r] = total_accuracy
        certain_correction_miou[r] = miou

    # # depth estimation for error correction
    # depth_correction_accuracy = {}
    # depth_correction_miou = {}
    # depth_trgt = depth_trgt.flatten()
    # depth_pred = depth_pred.flatten()
    # depth_trgt = depth_trgt[valid_pix_ids]
    # depth_pred = depth_pred[valid_pix_ids]
    # mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    # mask = (depth_trgt<10) * (depth_trgt>0) * mask1
    # depth_pred = depth_pred[mask]
    # depth_trgt = depth_trgt[mask]
    # predicted_labels = predicted_labels[mask]
    # true_labels = true_labels[mask]
    # thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    # for i in range(0, 10):
        # r_mask = thresh < 1.25**((i+1)/2)
        # r = 1-np.mean(r_mask.astype('float'))
        # predicted_cer = predicted_labels[r_mask]
        # true_cer = true_labels[r_mask]
        # conf_mat = confusion_matrix(true_cer, predicted_cer, labels=list(range(number_classes)))
        # norm_conf_mat = np.transpose(
        #     np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))
        # missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        # exsiting_class_mask = ~ missing_class_mask
        # class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        # total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        # ious = np.zeros(number_classes)
        # for class_id in range(number_classes):
        #     ious[class_id] = (conf_mat[class_id, class_id] / (
        #             np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
        #             conf_mat[class_id, class_id]))
        # miou = nanmean(ious)
        # depth_correction_accuracy[r] = total_accuracy
        # depth_correction_miou[r] = miou

    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious

def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)

if __name__=='__main__':
    train()



