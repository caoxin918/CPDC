import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

torch.set_default_tensor_type(torch.FloatTensor)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = torch.tensor(dist,dtype=torch.float32)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def patch_farthest_point_sample_batch(xyz, npoint,np):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.tensor(np).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = torch.tensor(dist,dtype=torch.float32)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #print('points size:', points.size(), 'idx size:', idx.size())
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape == [B, S, K]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # view_shape == [B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # repeat_shape == [1, S, K]
    #print('points:', points.size(), ', idx:', idx.size(), ', view_shape:', view_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # batch_indices == tensor[0, 1, ..., B-1]
    #print('batch_indices:', batch_indices.size())
    batch_indices = batch_indices.view(view_shape)
    # batch_indices size == [B, 1, 1]
    #print('after view batch_indices:', batch_indices.size())
    batch_indices = batch_indices.repeat(repeat_shape)
    # batch_indices size == [B, S, K]
    new_points = points[batch_indices, idx.long(), :]
    return new_points

# POINTNET
class PointNet(nn.Module):
    def __init__(self, in_channel, mlp, relu, bn):
        super(PointNet, self).__init__()

        mlp.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlp) - 1):
            if relu[i]:
                if bn:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.BatchNorm2d(mlp[i+1]),
                        nn.ReLU(),
                        )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.ReLU(),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlp[i], mlp[i+1], 1),
                    )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D]
        """
        
        points = points.unsqueeze(-1) # [B, C, N, 1]
        
        for m in self.mlp_Modules:
            points = m(points)
        # [B, D, N, 1]
        
        #points_np = points.detach().cpu().numpy()
        #np.save('./npys/ae_pn_feature.npy', points_np)

        points = torch.max(points, 2)[0]    # [B, D, 1]
        points = points.squeeze(-1) # [B, D] 

        return points



