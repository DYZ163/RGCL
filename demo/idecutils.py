# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import division, print_function
import numpy as np
import torch
import hnswlib
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import normalized_mutual_info_score, f1_score, adjusted_rand_score, cluster, accuracy_score, \
    precision_score, recall_score
import sklearn.metrics as metrics
from munkres import Munkres
from typing import Dict, Optional
from omegaconf import DictConfig
from torch import nn, Tensor

pre = precision_score
rec = recall_score
Fscore = f1_score


def fit_hnsw_index(features, ef=100, M=40, save_index_file=False):
    # Convenience function to create HNSW graph
    # features : list of lists containing the embeddings
    # ef, M: parameters to tune the HNSW algorithm

    num_elements = len(features)
    labels_index = np.arange(num_elements)
    EMBEDDING_SIZE = len(features[0])

    # Declaring index
    # possible space options are l2, cosine or ip
    p = hnswlib.Index(space='l2', dim=EMBEDDING_SIZE)

    # Initing index - the maximum number of elements should be known
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)

    # Element insertion
    int_labels = p.add_items(features, labels_index)

    # Controlling the recall by setting ef
    # ef should always be > k
    p.set_ef(ef)

    # If you want to save the graph to a file
    if save_index_file:
        p.save_index(save_index_file)

    return p


def rebuild(data_matrix, drop_index, adj_array, k, y):
    # input: sample feature matrix of a view, index of dropped samples, feature of k-nn samples
    # output: rebuild sample feature matrix , feature of all k-nn samples, average feature of dropped samples
    rebuild_data_matrix = data_matrix.copy()
    feature_matrix = np.zeros((data_matrix.shape[0], k, data_matrix.shape[1]))
    adj = np.zeros((data_matrix.shape[0], k))
    feature_index = []

    for i in range(data_matrix.shape[0]):
        all = rebuild_data_matrix[adj_array[i]]
        adj[i] = adj_array[i]
        if i in drop_index:
            average = np.mean(all, axis=0)
            rebuild_data_matrix[i] = average
        feature_matrix[i] = all

    g_error = graph_error(adj.astype(int), y)

    return rebuild_data_matrix, feature_matrix, g_error


def indices2feature(feature, adj_indices, device):
    # feature -- size:(N * D)
    # adj_indices -- size:(N * knn)
    # reture feature_matrix -- size:(N * knn * D)
    feature_matrix = np.zeros((feature.shape[0], adj_indices.shape[1], feature.shape[1]))
    for i in range(feature.shape[0]):
        all = feature.data.cpu().numpy()[adj_indices[i]]
        feature_matrix[i] = all
    feature_matrix = torch.Tensor(feature_matrix).to(device)
    return feature_matrix


def graph_error(graph='', label=''):
    x_train, y_train = graph, label
    counter = 0
    for i, v in enumerate(x_train):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter += 1
    error = counter / (y_train.shape[0] * 10)
    return error


#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def acc_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] == c_x[:])
    accrate = err_x.astype(float) / (gt_s.shape[0])
    return accrate


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


#######################################################
# LOSS
#######################################################


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class SCELoss(nn.Module):
    """SCE loss.

    Args:
        trunk (DictConfig): Config tu build a trunk.
        optimizer (DictConfig): Config tu build optimizers and schedulers.
        projector (Optional[DictConfig]): Config to build a project. Defaults to None.
        predictor (Optional[DictConfig]): Config to build a predictor. Defaults to None.
        transform (Optional[DictConfig]): Config to perform transformation on input. Defaults to None.
        normalize_outputs (bool): If true, normalize outputs. Defaults to True.
        num_global_crops (int): Number of global crops which are the first elements of each batch. Defaults to 2.
        num_local_crops (int): Number of local crops which are the last elements of each batch. Defaults to 0.
        initial_momentum (float): initial value for the momentum update. Defaults to 0.999.
        scheduler_momentum (str): rule to update the momentum value. Defaults to constant.
        shuffle_bn (bool): If true, apply shuffle normalization trick from MoCo. Defaults to True.
        num_devices (int): Number of devices used to train the model in each node. Defaults to 1.
        simulate_n_devices (int): Number of devices to simulate to apply shuffle trick. Requires shuffle_bn to be True and num_devices to be 1. Defaults to 8.
        queue (Optional[DictConfig]): Config to build a queue. Defaults to None.
        sym (bool): If true, symmetrised the loss. Defaults to False.
        temp (float): Temperature parameter to scale the online similarities. Defaults to 0.1.
        temp_m (float): Temperature parameter to scale the target similarities. Initial value if warmup applied. Defaults to 0.05.
        coeff (float): Coeff parameter between InfoNCE and relational aspects. Defaults to 0.5.
    """

    def __init__(
            self,
            batch_size,
            temperature,
            device,
            temp: float = 0.1,
            temp_m: float = 0.05,
            coeff: float = 0.5
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.mask_l1 = self.mask_correlated_samples_l1(self.batch_size)
        self.mask_l2 = self.mask_correlated_samples_l2(self.batch_size)
        self.temp = temp
        self.temp_m = temp_m
        self.coeff = coeff

    def mask_correlated_samples(self, batch_size):
        ###########################################
        # 0 1 0 1
        # 1 0 1 0
        # 0 1 0 1
        # 1 0 1 0
        ###########################################
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def mask_correlated_samples_l1(self, batch_size):
        ###########################################
        # 0 1 0 0
        # 1 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        ###########################################
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        # Set the lower half to False
        mask[batch_size:N, :] = False
        mask[:batch_size, batch_size:N] = False
        return mask

    def mask_correlated_samples_l2(self, batch_size):
        ###########################################
        # 0 0 0 0
        # 0 0 0 0
        # 0 1 0 0
        # 1 0 0 0
        ###########################################
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        # Set the lower half to False
        mask[:batch_size, :] = False
        mask[batch_size:N, batch_size:N] = False
        return mask

    # 定义对数归一化函数
    def log_normalization_dim(self, X, dim=1):
        # 对每个元素加 1，计算 log
        X_log = torch.log(X + 1)  # 对数据取对数

        # 沿着指定的维度 dim 求每行的和
        row_sums = torch.sum(X_log, dim=dim, keepdim=True)

        # 将每一行的对数归一化值除以该行的和，使得每行的和为1
        X_log_norm = X_log / row_sums
        return X_log_norm

    def forward(self, k, q):
        """Compute the SCE loss.

        .. math::
            L_{SCE} = - \frac{1}{N} \sum_{i=1}^N\sum_{k=1}^N w^2_{ik} \log\left(p^1_{ik}\right)
            s^2_{ik} = \frac{{1}_{i \neq k} \cdot \exp(z^2_i \cdot z^2_k / \tau_m)}{\sum_{j=1}^{N}{1}_{i \neq j} \cdot \exp(z^2_i \cdot z^2_j / \tau_m)}
            w^2_{ik} = \lambda \cdot \mathbbm{1}_{i=k} + (1 - \lambda) \cdot s^2_{ik}
            p^1_{ik} = \frac{\exp(z^1_i \cdot z^2_k / \tau)}{\sum_{j=1}^{N}\exp(z^1_i \cdot z^2_j / \tau)}

        Args:
            q (Tensor): The representations of the queries.（强增强样本对，一致性关系）
            k (Tensor): The representations of the keys.（弱增强样本对，语义关系）
            queue : 表示与k同一个批次的除开k之外的样本
        Returns:
            Dict[str, Tensor]: The loss, logits, labels and similarities.
        """
        self.batch_size = q.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.mask_l1 = self.mask_correlated_samples_l1(self.batch_size)
        self.mask_l2 = self.mask_correlated_samples_l2(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((k, q), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        # 提前对角线向上偏移self.batch_size，然后提取对应元素
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device)
        # s1ik
        # sim_q_ktarget 计算 q 和 k 之间的相似度，q 和 k 是正样本对
        sim_q_ktarget = sim_i_j.unsqueeze(-1)

        sim_k_ktarget = torch.zeros(
            self.batch_size, device=self.device).unsqueeze(-1)

        # queue = self.queue.clone().detach()

        # sim_k_queue 计算 k 和 queue 之间的相似度，k 和 queue 中的键表示是负样本对
        sim_k_queue = sim[self.mask_l1].reshape(self.batch_size, -1)

        # sim_q_queue 计算 q 和 queue 之间的相似度，q 和 queue 中的键表示是负样本对
        sim_q_queue = sim[self.mask_l2].reshape(self.batch_size, -1)

        sim_k = torch.cat([sim_k_ktarget, sim_k_queue], dim=1)

        sim_q = torch.cat([sim_q_ktarget, sim_q_queue], dim=1)

        mask = nn.functional.one_hot(
            labels, self.batch_size)

        # p1ik
        logits_q = sim_q / self.temp

        # s2ik
        logits_k = sim_k / self.temp_m



        prob_k = nn.functional.softmax(logits_k, dim=1)
        # prob_k = self.log_normalization_dim(logits_k, dim=1)

        prob_q = nn.functional.normalize(
            self.coeff * mask + (1 - self.coeff) * prob_k, p=1, dim=1)

        loss = - \
            torch.sum(prob_q * nn.functional.log_softmax(logits_q,
                                                         dim=1), dim=1).mean(dim=0)
        # logits_q = self.log_normalization_dim(logits_q, dim=1)
        # print(logits_k[0])
        # print(prob_k[0])
        # print(logits_q[0])
        # loss = - \
        #     torch.sum(prob_q * torch.log(logits_q), dim=1).mean(dim=0)
        return loss
