# -*- coding: utf-8 -*-

from scripts.modelGraph import FullModel
from scripts.dataloaders import graphSitePPI
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score
from time import time
import numpy as np
import pandas as pd
import torch, sys
import torch.nn as nn
import argparse
import torch.nn.functional as F
import os, pickle
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.metrics import calculate_metric
import torch.optim as optim
from collections import deque
import math
from scripts.ClassCountEMA import ClassCountEMA
from scripts.CL import supcon_with_queue, MemoryQueue
from scripts.AdaptiveLambda import AdaptiveLambda


def setup_distributed(rank, world_size, port='12355'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def average_value_allGPU(value):
    """在所有GPU上平均一个标量"""
    world_size = dist.get_world_size() 
    if world_size < 2:
        return value
    with torch.no_grad():
        tensor = torch.tensor(value, device=torch.device('cuda', torch.distributed.get_rank()))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor.item() / dist.get_world_size()

def format_time(seconds):
    """格式化时间显示"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}小时：{int(minutes)}分钟：{int(seconds)}秒"

def to_var(x, device):
    """将张量移动到指定设备"""
    if torch.cuda.is_available():
        x = x.to(device, non_blocking=True)
    return x


def load_data(path):
    """加载数据"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



def train(model, train_loader, optimizer, device):
    """
    平衡训练函数
    """
    model.train()

    run_pep_loss = 0
    run_prot_loss = 0
    run_loss = 0

    prot_site_pred = []
    prot_site_gt = []
    pep_site_pred = []
    pep_site_gt = []

    ema_prot = ClassCountEMA(decay=0.9)
    ema_pep  = ClassCountEMA(decay=0.9)

    pep_queue  = MemoryQueue(dim=128, max_size=8000, device=device)
    prot_queue = MemoryQueue(dim=128, max_size=8000, device=device)

    tau = 0.07
    lambda_contrast_prot = 0.2
    lambda_contrast_pep  = 0.1

    adapt = AdaptiveLambda(
        base_prot=0.2,      # 你现在的起始值
        base_pep=0.1,
        warmup_steps=400,   # 5920 个 step/epoch 下，300~600 都行
        ema_decay=0.9,
        target_ratio=0.3,   # 让 contrast ≈ 0.3 * supervised（可按你期望调整）
        beta=0.5,
        min_lambda=0.02,
        max_lambda=2.0,
        use_cosine_warmup=False
    )

    for value, (pep_edge_attrs, pep_edges, pep_node_embed, pep_node_index, pep_sites, prot_edge_attrs, prot_edges, prot_sites, prot_emb) in enumerate(train_loader):

        prot_emb = to_var(prot_emb, device) 
        prot_edge_attrs = to_var(prot_edge_attrs, device)
        prot_edges = to_var(prot_edges, device)
        prot_sites = to_var(prot_sites, device)

        pep_edge_attrs = to_var(pep_edge_attrs, device)
        pep_edges = to_var(pep_edges, device)
        pep_node_embed = to_var(pep_node_embed, device)
        pep_node_index = to_var(pep_node_index, device)
        pep_sites = to_var(pep_sites, device)

        optimizer.zero_grad()
        
        # 前向传播
        prot_nodes, pep_nodes, prot_z, pep_z = model(pep_edge_attrs, pep_edges, pep_node_embed, pep_node_index, 
                                     prot_edge_attrs, prot_edges, prot_emb)

        # 处理target
        prot_target = prot_sites[prot_sites!=-1]
        pep_target = pep_sites[pep_sites!=-1]

        pep_nodes = pep_nodes.squeeze(0)
        pep_nodes = pep_nodes[pep_sites.view(-1)!=-1]

        prot_nodes = prot_nodes.squeeze(0)
        prot_nodes = prot_nodes[prot_sites.view(-1) !=1]
    
        prot_z = prot_z.squeeze(0)[prot_sites.view(-1) != -1]   # [Lp_valid, 128]
        pep_z  = pep_z.squeeze(0)[pep_sites.view(-1) != -1]   # [Lq_valid, 128]

        ema_prot.update(prot_target)
        ema_pep.update(pep_target)
        prot_w = ema_prot.weights("protein", device=device)
        pep_w  = ema_pep.weights("peptide", device=device)
    
        prot_loss = F.cross_entropy(prot_nodes, 
                            prot_target, 
                            ignore_index=-1, 
                            weight=prot_w)

        pep_loss = F.cross_entropy(pep_nodes, 
                            pep_target, 
                            ignore_index=-1, 
                            weight=pep_w)


        L_sup_prot = supcon_with_queue(
            z=prot_z, y=prot_target,
            queue=prot_queue, temperature=tau,
            pos_only=True, ignore_index=-1
        )
        # pep：正类较多，也可以 pos_only，或全监督都行
        L_sup_pep = supcon_with_queue(
            z=pep_z, y=pep_target,
            queue=pep_queue, temperature=tau,
            pos_only=True, ignore_index=-1
        )
        
        # 总损失

        lambda_contrast_prot, lambda_contrast_pep = adapt.step(
                                                                prot_sup=prot_loss.detach(),
                                                                pep_sup=pep_loss.detach(),
                                                                prot_con=L_sup_prot.detach(),
                                                                pep_con=L_sup_pep.detach()
                                                            )

        # 组合总损失
        loss = prot_loss + pep_loss + lambda_contrast_prot * L_sup_prot + lambda_contrast_pep * L_sup_pep


        run_prot_loss += prot_loss.item()
        run_pep_loss += pep_loss.item()
        run_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


        
        
        # 预测计算
        prot_probs = F.softmax(prot_nodes, dim=-1)[:, 1].detach().cpu().numpy()
        pep_probs = F.softmax(pep_nodes, dim=-1)[:, 1].detach().cpu().numpy()

        prot_site_pred.extend(prot_probs.tolist())
        prot_site_gt.extend(prot_target.cpu().detach().numpy().tolist())

        pep_site_pred.extend(pep_probs.tolist())
        pep_site_gt.extend(pep_target.cpu().detach().numpy().tolist())

    # 计算准确率和指标
    # 对多肽使用更高的阈值


    prot_binary_preds = (np.array(prot_site_pred) > 0.5).astype(int)
    pep_binary_preds = (np.array(pep_site_pred) > 0.5).astype(int)  # 更高阈值

    prot_correct = np.sum(prot_binary_preds == np.array(prot_site_gt))
    pep_correct = np.sum(pep_binary_preds == np.array(pep_site_gt))


    prot_site_auc, prot_site_prc, prot_site_mcc, prot_site_f1, prot_site_acc = calculate_metric(prot_binary_preds, prot_site_gt, prot_site_pred)
    pep_site_auc, pep_site_prc, pep_site_mcc, pep_site_f1, pep_site_acc = calculate_metric(pep_binary_preds, pep_site_gt, pep_site_pred)
    


    return (run_loss/len(train_loader), run_prot_loss/len(train_loader), run_pep_loss/len(train_loader), 
            prot_site_auc, prot_site_prc, prot_site_mcc, prot_site_f1, prot_site_acc,
            pep_site_auc, pep_site_prc, pep_site_mcc, pep_site_f1, pep_site_acc)


def test(model, test_loader, device):
    """
    测试函数
    """
    model.eval()

    run_pep_loss = 0
    run_prot_loss = 0
    run_loss = 0

    prot_site_pred = []
    prot_site_gt = []
    pep_site_pred = []
    pep_site_gt = []

    ema_prot = ClassCountEMA(decay=0.9)
    ema_pep  = ClassCountEMA(decay=0.9)
    
    with torch.no_grad():
        for value, (pep_edge_attrs, pep_edges, pep_node_embed, pep_node_index, pep_sites, prot_edge_attrs, prot_edges, prot_sites, prot_emb) in enumerate(test_loader):

            prot_emb = to_var(prot_emb, device) 
            prot_edge_attrs = to_var(prot_edge_attrs, device)
            prot_edges = to_var(prot_edges, device)
            prot_sites = to_var(prot_sites, device)

            pep_edge_attrs = to_var(pep_edge_attrs, device)
            pep_edges = to_var(pep_edges, device)
            pep_node_embed = to_var(pep_node_embed, device)
            pep_node_index = to_var(pep_node_index, device)
            pep_sites = to_var(pep_sites, device)

            # 前向传播
            prot_nodes, pep_nodes, prot_z, pep_z = model(pep_edge_attrs, pep_edges, pep_node_embed, pep_node_index, 
                                         prot_edge_attrs, prot_edges, prot_emb)

            # 处理target
            prot_target = prot_sites[prot_sites!=-1]
            pep_target = pep_sites[pep_sites!=-1]

            pep_nodes = pep_nodes.squeeze(0)
            pep_nodes = pep_nodes[pep_sites.view(-1)!=-1]

            prot_nodes = prot_nodes.squeeze(0)
            prot_nodes = prot_nodes[prot_sites.view(-1) != -1]

            prot_z = prot_z.squeeze(0)[prot_sites.view(-1) != -1]   # [Lp_valid, 128]
            pep_z  = pep_z.squeeze(0)[pep_sites.view(-1) != -1]   # [Lq_valid, 128]

        
            ema_prot.update(prot_target)
            ema_pep.update(pep_target)
            prot_w = ema_prot.weights("protein", device=device)
            pep_w  = ema_pep.weights("peptide", device=device)
              
            prot_loss = F.cross_entropy(prot_nodes, 
                                prot_target, 
                                ignore_index=-1, 
                                weight=prot_w)

            pep_loss = F.cross_entropy(pep_nodes, 
                                pep_target, 
                                ignore_index=-1, 
                                weight=pep_w)

            run_prot_loss += prot_loss.item()
            run_pep_loss += pep_loss.item()
            run_loss += (prot_loss.item() + pep_loss.item())
            
            prot_probs = F.softmax(prot_nodes, dim=-1)[:, 1].detach().cpu().numpy()
            pep_probs = F.softmax(pep_nodes, dim=-1)[:, 1].detach().cpu().numpy()

            prot_site_pred.extend(prot_probs.tolist())
            prot_site_gt.extend(prot_target.cpu().detach().numpy().tolist())

            pep_site_pred.extend(pep_probs.tolist())
            pep_site_gt.extend(pep_target.cpu().detach().numpy().tolist())

    

    # 计算准确率和指标
    prot_binary_preds = (np.array(prot_site_pred) > 0.5).astype(int)
    pep_binary_preds = (np.array(pep_site_pred) > 0.5).astype(int)  # 更高阈值

    prot_correct = np.sum(prot_binary_preds == np.array(prot_site_gt))
    pep_correct = np.sum(pep_binary_preds == np.array(pep_site_gt))

    prot_accuracy = prot_correct / len(prot_site_gt)
    pep_accuracy = pep_correct / len(pep_site_gt)

    prot_site_auc, prot_site_prc, prot_site_mcc, prot_site_f1, prot_site_acc = calculate_metric(prot_binary_preds, prot_site_gt, prot_site_pred)
    pep_site_auc, pep_site_prc, pep_site_mcc, pep_site_f1, pep_site_acc = calculate_metric(pep_binary_preds, pep_site_gt, pep_site_pred)


    return (run_loss/len(test_loader), run_prot_loss/len(  test_loader), run_pep_loss/len(test_loader), 
            prot_site_auc, prot_site_prc, prot_site_mcc, prot_site_f1, prot_site_acc,
            pep_site_auc, pep_site_prc, pep_site_mcc, pep_site_f1, pep_site_acc)


def train_pepnn_distributed(rank, world_size, epochs, output_file, use_focal_loss=True, port='12355'):
    
    # 设置分布式环境
    setup_distributed(rank, world_size, port)
    
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        logger.add(output_file)
    
    g = torch.Generator()
    g.manual_seed(2024)

    similarity = 5
    
    # 加载非标准肽数据
    file_path = '/work/data1/liutianyuan/wls/pepPI/data/non_canonical_peptide_chain.txt'
    df = pd.read_csv(file_path, sep='\t')
    pdbid = df["PDBID"].values
    chainId = df["PeptideChainId"].values

    non_canonical_peptide = []
    for i in range(len(pdbid)):
        non_canonical_peptide.append(pdbid[i].split('.')[0] + '_' + chainId[i])
    
    #'prot_5fold_5', 'pep_5fold_5', 
    folders = ['random_5fold_5','prot_5fold_5', 'pep_5fold_5','pair_9fold_5']

    for idx, folder in enumerate(folders):
        if folder == 'pair_9fold_5':
            fold_num = 9
        else:
            fold_num = 5

        for fold in range(1, fold_num + 1):
            
            if rank == 0:
                logger.info(f'======{folders[idx]}======')
                logger.info(f'======fold_{fold}======')

            dataFolder = os.path.join('/work/data1/liutianyuan/wls/split_list', folder)
            
            # 检查数据文件是否存在
            train_file = os.path.join(dataFolder, f'train_data_{similarity}_fold_{fold}.pkl')
            val_file = os.path.join(dataFolder, f'val_data_{similarity}_fold_{fold}.pkl')
            test_file = os.path.join(dataFolder, f'test_data_{similarity}_fold_{fold}.pkl')

            # 加载数据
            train_data = load_data(train_file)
            val_data = load_data(val_file)
            test_data = load_data(test_file)
       

            non_canonical_test_data = [
                                item for item in test_data
                                if (item[0].split('_')[0] + '_' + item[0].split('_')[1]) in non_canonical_peptide
                                ]
           
            
            # 创建数据集
            train_dataset = graphSitePPI(train_data)
            val_dataset = graphSitePPI(val_data)
            test_dataset = graphSitePPI(test_data)
            non_canonical_test_dataset = graphSitePPI(non_canonical_test_data)

            # 创建分布式采样器
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            non_canonical_test_sampler = DistributedSampler(non_canonical_test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

            # 创建DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                sampler=train_sampler,
                num_workers=5,
                pin_memory=True
            )
                
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                sampler=val_sampler,
                num_workers=5,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                sampler=test_sampler,
                num_workers=5,
                pin_memory=True
            )

            non_canonical_test_loader = DataLoader(
                non_canonical_test_dataset,
                batch_size=1,
                sampler=non_canonical_test_sampler,
                num_workers=5,
                pin_memory=True
            )

            edge_features = 39
            node_features = 32
            d_model = 300




                # 使用平衡模型
            model = FullModel(edge_features, node_features, 6, d_model, 2, 150, 300, 300)

            model = model.to(device)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

            # 初始化优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
            
            min_val_loss = 999999
            best_test_metrics = None
            
            for epoch in range(epochs):
            
                # 设置采样器的epoch
                train_sampler.set_epoch(epoch)
                
                start_time = time()

                # 训练
                train_results = train(model, train_loader, optimizer, device, use_focal_loss)
                
                # 解析训练结果
                train_loss, train_prot_loss, train_pep_loss, prot_auc, prot_prc, prot_mcc, prot_f1, prot_acc, pep_auc, pep_prc, pep_mcc, pep_f1, pep_acc = train_results

                scheduler.step()

                # 分布式平均化
                train_loss = average_value_allGPU(train_loss)
                train_prot_loss = average_value_allGPU(train_prot_loss)
                train_pep_loss = average_value_allGPU(train_pep_loss)
                prot_auc = average_value_allGPU(prot_auc)
                prot_prc = average_value_allGPU(prot_prc)
                prot_mcc = average_value_allGPU(prot_mcc)
                prot_acc = average_value_allGPU(prot_acc)
                prot_f1 = average_value_allGPU(prot_f1)
                pep_auc = average_value_allGPU(pep_auc)
                pep_prc = average_value_allGPU(pep_prc)
                pep_mcc = average_value_allGPU(pep_mcc)
                pep_f1 = average_value_allGPU(pep_f1)
                pep_acc = average_value_allGPU(pep_acc)

                elapsed = time() - start_time
                elapsed_time = format_time(elapsed)

                # 验证
                val_results = test(model, test_loader, device, use_focal_loss)
                
                # 解析验证结果
                val_loss, val_prot_loss, val_pep_loss, val_prot_auc, val_prot_prc, val_prot_mcc, val_prot_f1, val_prot_acc, val_pep_auc, val_pep_prc, val_pep_mcc, val_pep_f1, val_pep_acc = val_results

                # 分布式平均化验证指标
                val_loss = average_value_allGPU(val_loss)
                val_prot_loss = average_value_allGPU(val_prot_loss)
                val_pep_loss = average_value_allGPU(val_pep_loss)
                val_prot_auc = average_value_allGPU(val_prot_auc)
                val_prot_prc = average_value_allGPU(val_prot_prc)
                val_prot_mcc = average_value_allGPU(val_prot_mcc)
                val_prot_acc = average_value_allGPU(val_prot_acc)
                val_prot_f1 = average_value_allGPU(val_prot_f1)
                val_pep_auc = average_value_allGPU(val_pep_auc)
                val_pep_prc = average_value_allGPU(val_pep_prc)
                val_pep_mcc = average_value_allGPU(val_pep_mcc)
                val_pep_f1 = average_value_allGPU(val_pep_f1)
                val_pep_acc = average_value_allGPU(val_pep_acc)

                # 只在rank 0上打印和记录
                if rank == 0:   


                    logger.info(f"Epoch {epoch + 1}/{epochs}, elapsed_time: {elapsed_time}")
                    logger.info(f"Train - Loss: {train_loss:.4f}, Train prot_loss: {train_prot_loss:.4f}, train_pep_loss: {train_pep_loss:.4f}")
                    logger.info(f"Train - prot_AUC: {prot_auc:.4f}, prot_PRC: {prot_prc:.4f}, prot_MCC: {prot_mcc:.4f}, prot_F1: {prot_f1:.4f}, prot_Acc: {prot_acc:.4f}")
                    logger.info(f"Train - pep_AUC: {pep_auc:.4f}, pep_PRC: {pep_prc:.4f}, pep_MCC: {pep_mcc:.4f}, pep_F1: {pep_f1:.4f}, pep_Acc: {pep_acc:.4f}")

                    logger.info(f"Val - Loss: {val_loss:.4f}, Val prot_loss: {val_prot_loss:.4f}, val_pep_loss: {val_pep_loss:.4f}")
                    logger.info(f"Val - prot_AUC: {val_prot_auc:.4f}, prot_PRC: {val_prot_prc:.4f}, prot_MCC: {val_prot_mcc:.4f}, prot_F1: {val_prot_f1:.4f}, prot_Acc: {val_prot_acc:.4f}")
                    logger.info(f"Val - Pep_AUC: {val_pep_auc:.4f}, Pep_PRC: {val_pep_prc:.4f}, Pep_MCC: {val_pep_mcc:.4f}, Pep_F1: {val_pep_f1:.4f}, Pep_Acc: {val_pep_acc:.4f}")
                
                # 模型保存和测试
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    
                    # 测试
                    test_results = test(model, non_canonical_test_loader, device)
                    
                    # 解析测试结果
                    test_loss, test_prot_loss, test_pep_loss, test_prot_auc, test_prot_prc, test_prot_mcc, test_prot_f1, test_prot_acc, test_pep_auc, test_pep_prc, test_pep_mcc, test_pep_f1, test_pep_acc = test_results

                    # 平均化测试指标
                    test_loss = average_value_allGPU(test_loss)
                    test_prot_loss = average_value_allGPU(test_prot_loss)
                    test_pep_loss = average_value_allGPU(test_pep_loss)
                    test_prot_auc = average_value_allGPU(test_prot_auc)
                    test_prot_prc = average_value_allGPU(test_prot_prc)
                    test_prot_mcc = average_value_allGPU(test_prot_mcc)
                    test_prot_f1 = average_value_allGPU(test_prot_f1)
                    test_prot_acc = average_value_allGPU(test_prot_acc)
                    test_pep_auc = average_value_allGPU(test_pep_auc)
                    test_pep_prc = average_value_allGPU(test_pep_prc)
                    test_pep_mcc = average_value_allGPU(test_pep_mcc)
                    test_pep_f1 = average_value_allGPU(test_pep_f1)
                    test_pep_acc = average_value_allGPU(test_pep_acc)

                    best_test_metrics = [test_loss, test_prot_loss, test_pep_loss, test_prot_auc, test_prot_prc, test_prot_mcc, test_prot_f1, test_prot_acc, test_pep_auc, test_pep_prc, test_pep_mcc, test_pep_f1, test_pep_acc]

                    if rank == 0:

                        result_path = os.path.join('/work/data1/liutianyuan/wls/ncPPI-Bind/checkpoints', f'{folder}')
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)

                        torch.save(model.module.state_dict(), os.path.join(result_path, f'fold_{fold}_best.pth'))


                        print(f"Test - Loss: {test_loss:.4f}, AUC: {test_prot_auc:.4f}, PRC: {test_prot_prc:.4f}, MCC: {test_prot_mcc:.4f}, F1: {test_prot_f1:.4f}, Acc: {test_prot_acc:.4f}")
                        print(f"Test - Pep_AUC: {test_pep_auc:.4f}, Pep_PRC: {test_pep_prc:.4f}, Pep_MCC: {test_pep_mcc:.4f}, Pep_F1: {test_pep_f1:.4f}, Pep_Acc: {test_pep_acc:.4f}")

                        logger.info(f"Test - Loss: {test_loss:.4f}, AUC: {test_prot_auc:.4f}, PRC: {test_prot_prc:.4f}, MCC: {test_prot_mcc:.4f}, F1: {test_prot_f1:.4f}, Acc: {test_prot_acc:.4f}")
                        logger.info(f"Test - Pep_AUC: {test_pep_auc:.4f}, Pep_PRC: {test_pep_prc:.4f}, Pep_MCC: {test_pep_mcc:.4f}, Pep_F1: {test_pep_f1:.4f}, Pep_Acc: {test_pep_acc:.4f}")

            # 记录最佳结果
            if rank == 0 and best_test_metrics:
                test_loss, test_prot_loss, test_pep_loss, test_prot_auc, test_prot_prc, test_prot_mcc, test_prot_f1, test_prot_acc, test_pep_auc, test_pep_prc, test_pep_mcc, test_pep_f1, test_pep_acc = best_test_metrics
                
                logger.info(f'======BEST_RESULT======')
                best_log_msg = f"BEST Test - prot_AUC: {test_prot_auc:.4f}, prot_PRC: {test_prot_prc:.4f}, prot_MCC: {test_prot_mcc:.4f}, prot_F1: {test_prot_f1:.4f}, prot_Acc: {test_prot_acc:.4f}"
                best_log_msg += f", pep_AUC: {test_pep_auc:.4f}, pep_PRC: {test_pep_prc:.4f}, pep_MCC: {test_pep_mcc:.4f}, pep_F1: {test_pep_f1:.4f}, pep_Acc: {test_pep_acc:.4f}"
                logger.info(best_log_msg)

        cleanup_distributed()


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output_file", help="输出日志文件", required=True, type=str)
    parser.add_argument("--world_size", type=int, default=4, help="使用的GPU数量")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--port", type=str, default='12355', help="分布式训练端口")

    
    args = parser.parse_args()
    
    # 处理参数
    use_focal_loss = args.use_focal_loss and not args.no_focal_loss
    
    # 检查GPU数量
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    gpu_count = torch.cuda.device_count()
    if args.world_size > gpu_count:
        raise RuntimeError(f"Requested {args.world_size} GPUs but only {gpu_count} are available")

    print(f"Balanced Training for Peptide Imbalance:")
    print(f"  - Focal Loss: {use_focal_loss}")
    print(f"  - Reversed weights for peptide (neg > pos)")
    print(f"  - Higher threshold for peptide (0.7 vs 0.5)")
    print(f"  - GPUs: {args.world_size}")
    print(f"  - Epochs: {args.epochs}")

    # 启动多进程训练
    mp.spawn(
        train_pepnn_distributed,
        args=(args.world_size, args.epochs, args.output_file, use_focal_loss, args.port),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
