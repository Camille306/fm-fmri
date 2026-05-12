"""
GCN and GAT models for Biopoint fMRI autism classification.
Both use dynamic FC snapshots: node features = FC row, edges = FC weights.
Select model with --model_type gcn|gat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def _batch_indices(batch_size: int, num_nodes_per_graph: int, device: torch.device) -> torch.Tensor:
    """Return batch index for each node: [0,0,...,0, 1,1,...,1, ...] (num_nodes_per_graph each)."""
    return torch.arange(batch_size, device=device).repeat_interleave(num_nodes_per_graph).long()


class GATBiopoint(nn.Module):
    """
    Graph Attention Network for Biopoint fMRI classification.
    Uses dynamic FC snapshots: each snapshot is a graph over ROIs;
    node features = FC row (roi_num,); edges carry FC weight as edge_attr.
    Local pooling per snapshot, then global MLP.
    """

    def __init__(
        self,
        roi_num: int,
        window_num: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.roi_num = roi_num
        self.window_num = window_num
        self.hidden_dim = hidden_dim

        # GAT layers with edge attributes (FC weights)
        self.conv1 = GATConv(roi_num, hidden_dim, heads=1, edge_dim=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, edge_dim=1)

        # Local aggregation: per-window we get 4 * hidden_dim (gmp + gap for conv1 and conv2)
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

        temp_x1 = self.conv1(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )
        temp_x2 = self.conv2(
            x=temp_x1,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

        # Per-snapshot local pooling
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
            seg = torch.cat(
                [
                    gmp(aggr_1[k], batch_idx),
                    gap(aggr_1[k], batch_idx),
                    gmp(aggr_2[k], batch_idx),
                    gap(aggr_2[k], batch_idx),
                ],
                dim=1,
            )
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


class GCNBiopoint(nn.Module):
    """
    Graph Convolutional Network for Biopoint fMRI classification.
    Same dynamic-FC snapshot strategy as GATBiopoint; uses GCNConv instead of GATConv.
    GCNConv normalises by node degree; edge_attr is NOT used (GCN is unweighted by default),
    but we keep the edge structure identical to GAT for a fair comparison.
    Local pooling per snapshot, then global MLP.
    """

    def __init__(
        self,
        roi_num: int,
        window_num: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.roi_num = roi_num
        self.window_num = window_num
        self.hidden_dim = hidden_dim

        # GCN layers (no edge attributes)
        self.conv1 = GCNConv(roi_num, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Same MLP head as GAT: window_num snapshots × 4 pooling features each
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

        # GCNConv: edge_attr not used, only edge_index
        temp_x1 = F.relu(self.conv1(x=data.x, edge_index=data.edge_index))
        temp_x2 = F.relu(self.conv2(x=temp_x1, edge_index=data.edge_index))

        # Per-snapshot local pooling (identical layout to GATBiopoint)
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
            seg = torch.cat(
                [
                    gmp(aggr_1[k], batch_idx),
                    gap(aggr_1[k], batch_idx),
                    gmp(aggr_2[k], batch_idx),
                    gap(aggr_2[k], batch_idx),
                ],
                dim=1,
            )
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


def build_model(model_type: str, roi_num: int, window_num: int, hidden_dim: int = 128,
                num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    """Factory: model_type in {'gcn', 'gat'}."""
    model_type = model_type.lower()
    if model_type == "gat":
        return GATBiopoint(roi_num=roi_num, window_num=window_num, hidden_dim=hidden_dim,
                           num_classes=num_classes, dropout=dropout)
    elif model_type == "gcn":
        return GCNBiopoint(roi_num=roi_num, window_num=window_num, hidden_dim=hidden_dim,
                           num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'gcn' or 'gat'.")
