"""
Train and test GAT on Biopoint for autism classification (real and/or synthetic fMRI).
"""

import os
import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from model import GATBiopoint
from dataset_biopoint import DatasetBiopointRest, DatasetBiopointRestWithSynthetic


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_all = 0.0
    correct = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        correct += (pred == data.y.view(-1)).sum().item()
        loss_all += loss.item() * data.num_graphs
        n += data.num_graphs
    return loss_all / max(n, 1), (correct / max(n, 1)) * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_all = 0.0
    correct = 0
    n = 0
    out_list = []
    pred_list = []
    y_list = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        pred = out.argmax(dim=1)
        correct += (pred == data.y.view(-1)).sum().item()
        loss_all += loss.item() * data.num_graphs
        n += data.num_graphs
        out_list.append(out.cpu())
        pred_list.append(pred.cpu())
        y_list.append(data.y.view(-1).cpu())
    if not out_list:
        return 0.0, 0.0, 0.0, 0.0
    out_all = torch.cat(out_list, dim=0)
    pred_all = torch.cat(pred_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    f1 = f1_score(y_all.numpy(), pred_all.numpy(), zero_division=1.0)
    try:
        auc = roc_auc_score(y_all.numpy(), out_all[:, 1].numpy())
    except Exception:
        auc = 0.0
    return loss_all / max(n, 1), (correct / max(n, 1)) * 100, f1, auc


def _build_dataset(argv):
    sparsity = getattr(argv, "sparsity", 0)
    noise_level = getattr(argv, "noise_level", 0.0)
    bootstrap_n = getattr(argv, "bootstrap_n", 0)
    common = dict(
        csv_path=argv.csv_path,
        k_fold=argv.k_fold,
        train_ratio=getattr(argv, "train_ratio", 0.8),
        window_size=argv.window_size,
        window_stride=argv.window_stride,
        window_num=argv.window_num,
        dynamic_length=argv.dynamic_length,
        ts_filename_suffix=argv.ts_filename_suffix,
        atlas_source=getattr(argv, "atlas_source", "shen268"),
        dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
        sparsity=sparsity,
        noise_level=noise_level,
        bootstrap_n=bootstrap_n,
        use_rest=(getattr(argv, "data_type", "rest") == "rest"),
    )
    if getattr(argv, "use_synthetic", False):
        return DatasetBiopointRestWithSynthetic(
            argv.sourcedir, synthetic_dir=argv.synthetic_dir, **common,
        )
    return DatasetBiopointRest(argv.sourcedir, **common)


def _build_model(roi_num, window_num, argv, num_classes, device):
    if getattr(argv, "model", "v1") == "v2":
        from model_v2 import GATBiopointV2
        model = GATBiopointV2(
            roi_num=roi_num,
            window_num=window_num,
            hidden_dim=argv.hidden_dim,
            num_classes=num_classes,
            dropout=argv.dropout,
            num_heads=argv.num_heads,
            use_gatv2=argv.use_gatv2,
            num_gat_layers=argv.num_gat_layers,
            residual=argv.residual,
            mlp_hidden=argv.mlp_hidden,
            mlp_bottleneck=argv.mlp_bottleneck,
            pool_mode=argv.pool_mode,
        )
    else:
        model = GATBiopoint(
            roi_num=roi_num,
            window_num=window_num,
            hidden_dim=argv.hidden_dim,
            num_classes=num_classes,
            dropout=argv.dropout,
        )
    return model.to(device)


def _compute_class_weights(dataset, device):
    """Inverse-frequency class weights from the full training set."""
    num_classes = dataset.num_classes
    label_counts = np.zeros(num_classes)
    if hasattr(dataset, '_train_real_ids') and dataset._train_real_ids:
        for sid in dataset._train_real_ids:
            label_counts[dataset.behavioral_dict[sid]] += 1
    else:
        for sid in dataset.subject_list:
            label_counts[dataset.behavioral_dict[sid]] += 1
    if hasattr(dataset, '_train_synthetic_items') and dataset._train_synthetic_items:
        for _, _, lab in dataset._train_synthetic_items:
            label_counts[lab] += 1
    if label_counts.sum() == 0:
        label_counts[:] = 1
    weights = torch.tensor(
        [label_counts.sum() / (num_classes * c) if c > 0 else 1.0 for c in label_counts],
        dtype=torch.float32,
    ).to(device)
    return weights, label_counts


def train(argv):
    os.makedirs(os.path.join(argv.targetdir, "model"), exist_ok=True)
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    device = torch.device("cuda" if argv.device == "cuda" and torch.cuda.is_available() else "cpu")
    if argv.device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but CUDA not available, using CPU")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(argv.seed)

    dataset = _build_dataset(argv)

    roi_num = dataset.num_nodes
    window_num = dataset.window_num
    num_classes = dataset.num_classes

    target_folds = dataset.folds
    if getattr(argv, "fold", None) is not None:
        target_folds = [argv.fold]
        print(f"Training only fold {argv.fold}")

    for k in target_folds:
        os.makedirs(os.path.join(argv.targetdir, "model", str(k)), exist_ok=True)
        dataset.set_fold(k, train=True)

        class_weights, label_counts = _compute_class_weights(dataset, device)
        if k == target_folds[0]:
            print(f"Train label counts: {dict(enumerate(label_counts.astype(int).tolist()))}, weights: {class_weights.tolist()}")

        train_loader = DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=True, num_workers=0)

        model = _build_model(roi_num, window_num, argv, num_classes, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if k == target_folds[0]:
            print(f"Model params: {n_params:,}")

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr, weight_decay=argv.weight_decay)
        sched_type = getattr(argv, "scheduler", "step")
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=argv.num_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(argv, "step_size", 10),
                gamma=getattr(argv, "step_gamma", 0.4),
            )

        model_path = os.path.join(argv.targetdir, "model", str(k), "model.pth")
        for epoch in range(argv.num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Fold {k} Epoch {epoch + 1}: "
                    f"train loss={train_loss:.4f} acc={train_acc:.2f}% "
                    f"(n_train={len(dataset)})"
                )

        torch.save(model.state_dict(), model_path)

    print("Training done.")


def test(argv):
    device = torch.device("cuda" if argv.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = _build_dataset(argv)

    roi_num = dataset.num_nodes
    window_num = dataset.window_num
    num_classes = dataset.num_classes
    criterion = torch.nn.CrossEntropyLoss()

    target_folds = dataset.folds
    if getattr(argv, "fold", None) is not None:
        target_folds = [argv.fold]

    all_metrics = []
    for k in target_folds:
        model = _build_model(roi_num, window_num, argv, num_classes, device)
        ckpt = os.path.join(argv.targetdir, "model", str(k), "model.pth")
        if not os.path.isfile(ckpt):
            print(f"Skip fold {k}: no checkpoint {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device))

        dataset.set_fold(k, train=False)
        test_loader = DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=0)
        test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device)
        all_metrics.append({"loss": test_loss, "acc": test_acc, "f1": test_f1, "auc": test_auc})
        print(f"Fold {k}: loss={test_loss:.4f} acc={test_acc:.2f}% f1={test_f1:.4f} auc={test_auc:.4f}")

    if all_metrics:
        mean_acc = np.mean([m["acc"] for m in all_metrics])
        mean_f1 = np.mean([m["f1"] for m in all_metrics])
        mean_auc = np.mean([m["auc"] for m in all_metrics])
        std_acc = np.std([m["acc"] for m in all_metrics])
        std_f1 = np.std([m["f1"] for m in all_metrics])
        std_auc = np.std([m["auc"] for m in all_metrics])
        print(f"Mean test: acc={mean_acc:.2f}±{std_acc:.2f}% f1={mean_f1:.4f}±{std_f1:.4f} auc={mean_auc:.4f}±{std_auc:.4f}")
        results_path = os.path.join(argv.targetdir, "test_results.txt")
        with open(results_path, "w") as f:
            f.write(f"acc\t{mean_acc}\n")
            f.write(f"f1\t{mean_f1}\n")
            f.write(f"auc\t{mean_auc}\n")
        print(f"Results written to {results_path}")
    return all_metrics


def compare(argv):
    """Head-to-head: real-only vs real+synthetic on same folds/test set."""
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    device = torch.device("cuda" if argv.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(argv.seed)

    common_kwargs = dict(
        csv_path=argv.csv_path,
        k_fold=argv.k_fold,
        train_ratio=getattr(argv, "train_ratio", 0.8),
        window_size=argv.window_size,
        window_stride=argv.window_stride,
        window_num=argv.window_num,
        dynamic_length=argv.dynamic_length,
        ts_filename_suffix=argv.ts_filename_suffix,
        atlas_source=getattr(argv, "atlas_source", "shen268"),
        dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
    )

    print("\n" + "=" * 70)
    print("Training: REAL-ONLY data")
    print("=" * 70)
    ds_real = DatasetBiopointRest(argv.sourcedir, **common_kwargs)
    real_metrics = _train_and_eval_on_folds(ds_real, argv, device, "Real-only", "compare_real")

    print("\n" + "=" * 70)
    print("Training: REAL + SYNTHETIC data")
    print("=" * 70)
    ds_syn = DatasetBiopointRestWithSynthetic(
        argv.sourcedir, synthetic_dir=argv.synthetic_dir, **common_kwargs,
    )
    syn_metrics = _train_and_eval_on_folds(ds_syn, argv, device, "Real+Syn", "compare_synthetic")

    def _summarize(metrics):
        acc = np.mean([m["acc"] for m in metrics])
        f1 = np.mean([m["f1"] for m in metrics])
        auc = np.mean([m["auc"] for m in metrics])
        acc_s = np.std([m["acc"] for m in metrics])
        f1_s = np.std([m["f1"] for m in metrics])
        auc_s = np.std([m["auc"] for m in metrics])
        return acc, acc_s, f1, f1_s, auc, auc_s

    r_acc, r_acc_s, r_f1, r_f1_s, r_auc, r_auc_s = _summarize(real_metrics)
    s_acc, s_acc_s, s_f1, s_f1_s, s_auc, s_auc_s = _summarize(syn_metrics)

    print("\n" + "=" * 70)
    print("COMPARISON: Real-only vs Real+Synthetic  (same folds, same test set)")
    print("=" * 70)
    print(f"{'':>16}  {'ACC':>16}  {'F1':>16}  {'AUC':>16}")
    print(f"{'Real-only':>16}  {r_acc:6.2f}±{r_acc_s:5.2f}%  {r_f1:6.4f}±{r_f1_s:.4f}  {r_auc:6.4f}±{r_auc_s:.4f}")
    print(f"{'Real+Synthetic':>16}  {s_acc:6.2f}±{s_acc_s:5.2f}%  {s_f1:6.4f}±{s_f1_s:.4f}  {s_auc:6.4f}±{s_auc_s:.4f}")
    print(f"{'Delta':>16}  {s_acc - r_acc:+6.2f}        {s_f1 - r_f1:+6.4f}        {s_auc - r_auc:+6.4f}")
    print("=" * 70)

    results_path = os.path.join(argv.targetdir, "comparison_results.txt")
    with open(results_path, "w") as f:
        f.write("condition\tacc\tacc_std\tf1\tf1_std\tauc\tauc_std\n")
        f.write(f"real_only\t{r_acc}\t{r_acc_s}\t{r_f1}\t{r_f1_s}\t{r_auc}\t{r_auc_s}\n")
        f.write(f"real_synthetic\t{s_acc}\t{s_acc_s}\t{s_f1}\t{s_f1_s}\t{s_auc}\t{s_auc_s}\n")
    print(f"Comparison results written to {results_path}")


def _train_and_eval_on_folds(dataset, argv, device, label, save_subdir):
    """Train on each fold (no val split) and return per-fold test metrics."""
    roi_num = dataset.num_nodes
    window_num = dataset.window_num
    num_classes = dataset.num_classes
    criterion = torch.nn.CrossEntropyLoss()

    model_dir = os.path.join(argv.targetdir, save_subdir, "model")
    os.makedirs(model_dir, exist_ok=True)

    all_metrics = []
    for k in dataset.folds:
        fold_dir = os.path.join(model_dir, str(k))
        os.makedirs(fold_dir, exist_ok=True)

        dataset.set_fold(k, train=True)
        train_loader = DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=True, num_workers=0)

        model = _build_model(roi_num, window_num, argv, num_classes, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr, weight_decay=argv.weight_decay)
        sched_type = getattr(argv, "scheduler", "step")
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=argv.num_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(argv, "step_size", 10),
                gamma=getattr(argv, "step_gamma", 0.4),
            )

        model_path = os.path.join(fold_dir, "model.pth")
        for epoch in range(argv.num_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
        torch.save(model.state_dict(), model_path)

        dataset.set_fold(k, train=False)
        test_loader = DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=0)
        test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device)
        all_metrics.append({"fold": k, "loss": test_loss, "acc": test_acc, "f1": test_f1, "auc": test_auc})
        print(f"  [{label}] Fold {k}: acc={test_acc:.2f}%  f1={test_f1:.4f}  auc={test_auc:.4f}")

    return all_metrics
