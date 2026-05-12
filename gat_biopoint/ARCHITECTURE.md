# GAT Biopoint: Architecture and Dimensions

## Data flow (input → graph → model → logits)

### 1. Input: ROI time series
- **Shape**: `(T, roi_num)` per subject, e.g. `(176, 268)` for Shen 268 atlas.
- **Dynamic FC**: For each of `window_num` time windows we compute a correlation matrix over `window_size` timepoints → `window_num` FC matrices of shape `(roi_num, roi_num)`.

### 2. Graph construction (per subject)

Each subject becomes **one PyG graph** with multiple “snapshots” (time windows) stacked as disjoint subgraphs:

| Quantity | Formula | Example (roi_num=268, window_num=12) |
|----------|---------|--------------------------------------|
| **Nodes per subject** | `roi_num × window_num` | 268 × 12 = **3,216** |
| **Node feature dim** | `roi_num` (each node = one row of FC) | 268 |
| **Edges per snapshot** | `roi_num × (roi_num − 1)` (all pairs, no self-loop) | 268 × 267 = **71,556** |
| **Edges per subject** | `window_num × edges_per_snapshot` | 12 × 71,556 = **858,672** |
| **`x`** | `(nodes_per_subject, roi_num)` | (3,216, 268) |
| **`edge_index`** | `(2, edges_per_subject)` | (2, 858,672) |
| **`edge_attr`** | `(edges_per_subject, 1)` | (858,672, 1) |

So **one subject** already has **~859K edges**. With **batch_size = 8**, a single batch has **~6.9M edges**.

### 3. Model dimensions (default: hidden_dim=128, window_num=12)

| Layer / step | Input shape | Output shape |
|--------------|-------------|--------------|
| **conv1** (GAT) | `x`: (N, roi_num), edge_index (2, E), edge_attr (E, 1) | (N, **128**) |
| **conv2** (GAT) | (N, 128), same edges | (N, **128**) |
| **Local aggregation** | For each of `window_num` snapshots, for each graph: take the `roi_num` nodes of that snapshot and apply global mean pool + global max pool on both conv1 and conv2 outputs → 4 × 128 = **512** dims per snapshot. | Per graph: (12 × 512) = **6,144** |
| **fc1** | (batch, 6144) | (batch, **512**) |
| **bn1** | (batch, 512) | (batch, 512) |
| **fc2** | (batch, 512) | (batch, **32**) |
| **bn2** | (batch, 32) | (batch, 32) |
| **fc3** | (batch, 32) | (batch, **2**) |
| **Output** | Softmax over 2 classes | (batch, 2) |

So the **MLP head** sees a **6,144-dimensional** vector per subject (12 windows × 4 pools × 128 hidden).

### 4. Batch sizes (example: batch_size=8, roi_num=268, window_num=12)

| Quantity | Value |
|----------|--------|
| Total nodes in batch | 8 × 3,216 = **25,728** |
| Total edges in batch | 8 × 858,672 = **6,869,376** |

---

## Why it is slow

1. **Very large edge count**
   - Edges scale as **O(roi_num²)** per snapshot. With Shen 268: ~71K edges per snapshot, ~859K per subject.
   - GAT does **attention over every edge** (and we have two GAT layers), so compute and memory are dominated by **E ≈ 6.9M** per batch.

2. **CPU-only**
   - All GAT and MLP ops run on CPU; no GPU acceleration.

3. **Data loading**
   - Each `__getitem__` builds **858K edges** with Python loops (`_fc_to_graph` double loop over roi_num × roi_num) and many small tensor creations, which is slow on CPU.

4. **Aggregation**
   - The per-snapshot aggregation uses Python loops over `window_num × batch_size` and repeated slicing; not a major bottleneck compared to the GAT layers, but not optimal.

---

## Ways to speed it up

- **Sparsify the graph**: Keep only top-k edges per node (e.g. by |FC| or correlation threshold) so E drops from O(roi_num²) to O(roi_num × k).
- **Use GPU**: Set `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` (and move data/model to it).
- **Smaller batches**: e.g. `minibatch_size=2` or 4 to reduce memory and edge count per step.
- **Fewer windows**: e.g. `window_num=6` halves both aggregation size and (if you only compute 6 FCs) part of the graph size.
- **Fewer ROIs**: Use a smaller atlas (e.g. AAL 116) so roi_num² is much smaller.
