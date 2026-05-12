# Simple 2-layer GAT for Biopoint fMRI autism classification.
# Architecture matches the STNAGNN-fMRI reference (GNN_att with local pooling, GAT backbone).
# https://github.com/Jiyao96/STNAGNN-fMRI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def _batch_indices(batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size, device=device).repeat_interleave(num_nodes).long()


class GATBiopoint(nn.Module):
    """2-layer GAT for fMRI classification on dynamic FC snapshots.

    Each snapshot is a graph over ROIs; node features = FC row; edges carry FC weight.
    Per-snapshot gmp+gap from both conv layers, concat across snapshots, then MLP.
    F.softmax on output (matching STNAGNN reference — implicit label smoothing with CE loss).
    """

    def __init__(
        self,
        roi_num: int,
        window_num: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.roi_num = roi_num
        self.window_num = window_num
        self.hidden_dim = hidden_dim
        self.pool_mode = "concat"

        self.conv1 = GATConv(roi_num, hidden_dim, heads=1, edge_dim=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, edge_dim=1)

        self.fc1 = nn.Linear(hidden_dim * window_num * 4, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dp = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(32)
        self.silu = nn.SiLU()

    def forward(self, data):
        batch_size = data.num_graphs
        nodes_per_graph = self.roi_num * self.window_num

        temp_x1 = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        temp_x2 = self.conv2(x=temp_x1, edge_index=data.edge_index, edge_attr=data.edge_attr)

        aggr_list1 = [[] for _ in range(self.window_num)]
        aggr_list2 = [[] for _ in range(self.window_num)]
        for i in range(self.window_num * batch_size):
            snap_num = i % self.window_num
            start = (i // self.window_num) * nodes_per_graph + snap_num * self.roi_num
            end = start + self.roi_num
            aggr_list1[snap_num].append(temp_x1[start:end, :])
            aggr_list2[snap_num].append(temp_x2[start:end, :])

        aggr_1 = [torch.stack(aggr_list1[j]).reshape(-1, self.hidden_dim) for j in range(self.window_num)]
        aggr_2 = [torch.stack(aggr_list2[j]).reshape(-1, self.hidden_dim) for j in range(self.window_num)]
        batch_idx = _batch_indices(batch_size, self.roi_num, data.x.device)
        aggr_all = []
        for k in range(self.window_num):
            seg = torch.cat([
                gmp(aggr_1[k], batch_idx), gap(aggr_1[k], batch_idx),
                gmp(aggr_2[k], batch_idx), gap(aggr_2[k], batch_idx),
            ], dim=1)
            aggr_all.append(seg)
        aggr_x = torch.stack(aggr_all).permute(1, 0, 2).reshape(batch_size, -1)

        out = self.silu(self.fc1(aggr_x))
        out = self.bn1(out)
        out = self.dp(out)
        out = self.silu(self.fc2(out))
        out = self.bn2(out)
        out = self.dp(out)
        out = F.softmax(self.fc3(out), dim=1)
        return out
