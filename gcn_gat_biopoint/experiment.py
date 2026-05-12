"""
Train and test GCN/GAT on Biopoint for autism classification (real and/or synthetic fMRI).
Select model with --model_type gcn|gat.
"""

import os
import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from model import build_model
from dataset_biopoint import DatasetBiopointRest, DatasetBiopointRestWithSynthetic


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_all = 0.0
    correct = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze(-1)).sum().item()
        loss_all += loss.item() * data.num_graphs
        n += data.num_graphs
    return loss_all / max(n, 1), (correct / max(n, 1)) * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_all = 0.0
    correct = 0
    n = 0
    out_list, pred_list, y_list = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.squeeze(-1))
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze(-1)).sum().item()
        loss_all += loss.item() * data.num_graphs
        n += data.num_graphs
        out_list.append(out.cpu())
        pred_list.append(pred.cpu())
        y_list.append(data.y.squeeze(-1).cpu())
    if not out_list:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    out_all = torch.cat(out_list, dim=0)
    pred_all = torch.cat(pred_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    acc = (correct / max(n, 1)) * 100
    f1 = f1_score(y_all.numpy(), pred_all.numpy(), zero_division=1.0)
    try:
        auc = roc_auc_score(y_all.numpy(), out_all[:, 1].numpy())
    except Exception:
        auc = 0.0
    # Sensitivity (recall for class 1) and Specificity (recall for class 0)
    try:
        cm = confusion_matrix(y_all.numpy(), pred_all.numpy(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        sensitivity = specificity = 0.0
    return loss_all / max(n, 1), acc, f1, auc, sensitivity, specificity


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _build_dataset(argv, use_synthetic: bool):
    kwargs = dict(
        sourcedir=argv.sourcedir,
        csv_path=argv.csv_path,
        k_fold=argv.k_fold,
        train_ratio=getattr(argv, "train_ratio", 0.8),
        window_size=argv.window_size,
        window_stride=argv.window_stride,
        window_num=argv.window_num,
        dynamic_length=argv.dynamic_length,
        ts_filename_suffix=argv.ts_filename_suffix,
        top_k_edges=getattr(argv, "top_k_edges", 50),
        use_rest=not getattr(argv, "use_task", False),
        atlas_source=getattr(argv, "atlas_source", "shen268"),
        dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
    )
    if use_synthetic:
        return DatasetBiopointRestWithSynthetic(
            synthetic_dir=argv.synthetic_dir,
            **kwargs,
        )
    return DatasetBiopointRest(**kwargs)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(argv):
    os.makedirs(os.path.join(argv.targetdir, "model"), exist_ok=True)
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if argv.device == "cuda" and torch.cuda.is_available() else "cpu")
    if argv.device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but CUDA not available, using CPU")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(argv.seed)

    use_synthetic = getattr(argv, "use_synthetic", False)
    dataset = _build_dataset(argv, use_synthetic)

    roi_num = dataset.num_nodes
    window_num = dataset.window_num
    num_classes = dataset.num_classes

    max_folds = getattr(argv, "max_folds", 1)
    folds_to_run = dataset.folds[:max_folds]
    print(f"Model: {argv.model_type.upper()} | synthetic={use_synthetic} | device={device} | folds={folds_to_run}")

    for k in folds_to_run:
        os.makedirs(os.path.join(argv.targetdir, "model", str(k)), exist_ok=True)
        dataset.set_fold(k, train=True)

        train_loader = DataLoader(
            dataset,
            batch_size=argv.minibatch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        model = build_model(
            model_type=argv.model_type,
            roi_num=roi_num,
            window_num=window_num,
            hidden_dim=argv.hidden_dim,
            num_classes=num_classes,
            dropout=argv.dropout,
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr, weight_decay=argv.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(argv, "scheduler_step", 10),
            gamma=getattr(argv, "scheduler_gamma", 0.4),
        )

        quality_frac = getattr(argv, "quality_frac", 1.0)
        curriculum = getattr(argv, "curriculum", False)

        # Check whether synthetic items actually exist for this fold
        n_syn_available = len(getattr(dataset, "_train_synthetic_items", None) or [])
        use_quality_filter = quality_frac < 1.0 and n_syn_available > 0
        if quality_frac < 1.0 and n_syn_available == 0:
            print(
                f"  [quality] WARNING: quality_frac={quality_frac} requested but no synthetic "
                f"samples were loaded (check --synthetic_dir path and manifest). "
                f"Training on real data only."
            )

        for epoch in range(argv.num_epochs):
            # --- Quality / curriculum scheduling (only when synthetic data exists) ---
            if use_quality_filter:
                if curriculum and argv.num_epochs > 1:
                    # Linear warm-up: start at quality_frac, reach 1.0 at last epoch
                    frac = quality_frac + (1.0 - quality_frac) * epoch / (argv.num_epochs - 1)
                else:
                    # Hard filter: same top fraction every epoch
                    frac = quality_frac
                dataset.set_quality_curriculum(frac)
                # Rebuild loader with (potentially changed) dataset length
                train_loader = DataLoader(
                    dataset,
                    batch_size=argv.minibatch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True,
                )

            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{argv.model_type.upper()}] Fold {k} Epoch {epoch + 1}: "
                    f"train loss={train_loss:.4f} acc={train_acc:.2f}% "
                    f"(n_train={len(dataset)})"
                )

        torch.save(model.state_dict(), os.path.join(argv.targetdir, "model", str(k), "model.pth"))

    print("Training done.")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(argv):
    device = torch.device("cuda" if argv.device == "cuda" and torch.cuda.is_available() else "cpu")

    use_synthetic = getattr(argv, "use_synthetic", False)
    dataset = _build_dataset(argv, use_synthetic)

    roi_num = dataset.num_nodes
    window_num = dataset.window_num
    num_classes = dataset.num_classes
    criterion = torch.nn.CrossEntropyLoss()

    max_folds = getattr(argv, "max_folds", 1)
    folds_to_run = dataset.folds[:max_folds]

    all_metrics = []
    for k in folds_to_run:
        model = build_model(
            model_type=argv.model_type,
            roi_num=roi_num,
            window_num=window_num,
            hidden_dim=argv.hidden_dim,
            num_classes=num_classes,
            dropout=argv.dropout,
        ).to(device)
        ckpt = os.path.join(argv.targetdir, "model", str(k), "model.pth")
        if not os.path.isfile(ckpt):
            print(f"Skip fold {k}: no checkpoint {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device))

        dataset.set_fold(k, train=False)
        test_loader = DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=0)
        test_loss, test_acc, test_f1, test_auc, test_sens, test_spec = evaluate(
            model, test_loader, criterion, device
        )
        all_metrics.append({
            "loss": test_loss,
            "acc": test_acc,
            "f1": test_f1,
            "auc": test_auc,
            "sensitivity": test_sens,
            "specificity": test_spec,
        })
        print(
            f"[{argv.model_type.upper()}] Fold {k}: "
            f"loss={test_loss:.4f} acc={test_acc:.2f}% "
            f"f1={test_f1:.4f} auc={test_auc:.4f} "
            f"sensitivity={test_sens:.4f} specificity={test_spec:.4f}"
        )

    if all_metrics:
        mean_acc = np.mean([m["acc"] for m in all_metrics])
        mean_f1 = np.mean([m["f1"] for m in all_metrics])
        mean_auc = np.mean([m["auc"] for m in all_metrics])
        mean_sens = np.mean([m["sensitivity"] for m in all_metrics])
        mean_spec = np.mean([m["specificity"] for m in all_metrics])

        std_acc = np.std([m["acc"] for m in all_metrics])
        std_f1 = np.std([m["f1"] for m in all_metrics])
        std_auc = np.std([m["auc"] for m in all_metrics])
        std_sens = np.std([m["sensitivity"] for m in all_metrics])
        std_spec = np.std([m["specificity"] for m in all_metrics])

        print(
            f"\n[{argv.model_type.upper()}] Mean test: "
            f"acc={mean_acc:.2f}±{std_acc:.2f}% "
            f"f1={mean_f1:.4f}±{std_f1:.4f} "
            f"auc={mean_auc:.4f}±{std_auc:.4f} "
            f"sensitivity={mean_sens:.4f}±{std_sens:.4f} "
            f"specificity={mean_spec:.4f}±{std_spec:.4f}"
        )

        results_path = os.path.join(argv.targetdir, "test_results.txt")
        with open(results_path, "w") as f:
            f.write(f"model\t{argv.model_type}\n")
            f.write(f"use_synthetic\t{use_synthetic}\n")
            f.write(f"k_fold\t{argv.k_fold}\n")
            f.write(f"num_epochs\t{argv.num_epochs}\n")
            f.write(f"hidden_dim\t{argv.hidden_dim}\n")
            f.write("\n")
            f.write(f"acc_mean\t{mean_acc:.4f}\n")
            f.write(f"acc_std\t{std_acc:.4f}\n")
            f.write(f"f1_mean\t{mean_f1:.4f}\n")
            f.write(f"f1_std\t{std_f1:.4f}\n")
            f.write(f"auc_mean\t{mean_auc:.4f}\n")
            f.write(f"auc_std\t{std_auc:.4f}\n")
            f.write(f"sensitivity_mean\t{mean_sens:.4f}\n")
            f.write(f"sensitivity_std\t{std_sens:.4f}\n")
            f.write(f"specificity_mean\t{mean_spec:.4f}\n")
            f.write(f"specificity_std\t{std_spec:.4f}\n")
            f.write("\n")
            for i, m in enumerate(all_metrics):
                f.write(
                    f"fold_{i}\tacc={m['acc']:.4f}\tf1={m['f1']:.4f}\t"
                    f"auc={m['auc']:.4f}\tsensitivity={m['sensitivity']:.4f}\t"
                    f"specificity={m['specificity']:.4f}\n"
                )
        print(f"Results written to {results_path}")
