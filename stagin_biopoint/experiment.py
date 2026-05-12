"""
Train and test STAGIN on Biopoint for autism classification.
"""

import os
import random
import torch
import numpy as np
from einops import repeat
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from model import ModelSTAGIN
from dataset_biopoint import DatasetBiopointRest, DatasetBiopointRestWithSynthetic
from util import bold
from util.logger import LoggerSTAGIN


def step(
    model,
    criterion,
    dyn_v,
    dyn_a,
    sampling_endpoints,
    t,
    label,
    reg_lambda,
    clip_grad=0.0,
    device="cpu",
    optimizer=None,
    scheduler=None,
):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    logit, attention, latent, reg_ortho = model(
        dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints
    )
    loss = criterion(logit, label.to(device))
    loss = loss + reg_lambda * reg_ortho

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return logit, loss, attention, latent, reg_ortho


def train(argv):
    os.makedirs(os.path.join(argv.targetdir, "model"), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, "summary"), exist_ok=True)

    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(argv.seed)

    if getattr(argv, "use_synthetic", False):
        dataset = DatasetBiopointRestWithSynthetic(
            argv.sourcedir,
            synthetic_dir=argv.synthetic_dir,
            csv_path=argv.csv_path,
            k_fold=argv.k_fold,
            train_ratio=getattr(argv, "train_ratio", 0.8),
            dynamic_length=argv.dynamic_length,
            ts_filename_suffix=argv.ts_filename_suffix,
            atlas_source=getattr(argv, "atlas_source", "shen268"),
            dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
        )
    else:
        dataset = DatasetBiopointRest(
            argv.sourcedir,
            csv_path=argv.csv_path,
            k_fold=argv.k_fold,
            train_ratio=getattr(argv, "train_ratio", 0.8),
            dynamic_length=argv.dynamic_length,
            ts_filename_suffix=argv.ts_filename_suffix,
            atlas_source=getattr(argv, "atlas_source", "shen268"),
            dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
        )
    dynamic_length = argv.dynamic_length or dataset.num_timepoints

    checkpoint_path = os.path.join(argv.targetdir, "checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        print("Resuming from checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = {"fold": 0, "epoch": 0, "model": None, "optimizer": None, "scheduler": None}

    single_fold = getattr(argv, "fold", None)
    folds_to_run = [single_fold] if single_fold is not None else dataset.folds
    if single_fold is not None:
        print(f"Training only fold {single_fold}")

    for k_index, k in enumerate(folds_to_run):
        if checkpoint["fold"] and k_index < folds_to_run.index(checkpoint["fold"]):
            continue
        os.makedirs(os.path.join(argv.targetdir, "model", str(k)), exist_ok=True)
        dataset.set_fold(k, train=True)

        # Apply quality filtering on synthetic data if requested
        quality_frac = getattr(argv, "quality_frac", 1.0)
        if quality_frac < 1.0 and hasattr(dataset, "set_quality_curriculum"):
            dataset.set_quality_curriculum(quality_frac)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=argv.minibatch_size,
            shuffle=False,
            num_workers=argv.num_workers,
            pin_memory=True,
        )

        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        )
        model.to(device)
        if checkpoint["model"] is not None:
            model.load_state_dict(checkpoint["model"])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=argv.max_lr,
            epochs=argv.num_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.2,
            div_factor=argv.max_lr / argv.lr,
            final_div_factor=1000,
        )
        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        summary_writer = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "train"))
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "val"))
        logger = LoggerSTAGIN(dataset.folds, dataset.num_classes)

        for epoch in range(checkpoint["epoch"], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0

            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f"k:{k} e:{epoch}")):
                dyn_a, sampling_points = bold.process_dynamic_fc(
                    x["timeseries"], argv.window_size, argv.window_stride, dynamic_length
                )
                sampling_endpoints = [p + argv.window_size for p in sampling_points]
                if i == 0:
                    dyn_v = repeat(
                        torch.eye(dataset.num_nodes),
                        "n1 n2 -> b t n1 n2",
                        t=len(sampling_points),
                        b=argv.minibatch_size,
                    )
                if dyn_a.shape[0] < argv.minibatch_size:
                    dyn_v = dyn_v[: dyn_a.shape[0]]
                t = x["timeseries"].permute(1, 0, 2)
                label = x["label"]

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                logger.add(
                    k=k,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )
                summary_writer.add_scalar("lr", scheduler.get_last_lr()[0], i + epoch * len(dataloader))

            samples = logger.get(k)
            metrics = logger.evaluate(k, print_metric=False)
            summary_writer.add_scalar("loss", loss_accumulate / len(dataloader), epoch)
            summary_writer.add_scalar("reg_ortho", reg_ortho_accumulate / len(dataloader), epoch)
            summary_writer.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1], epoch)
            for key, value in metrics.items():
                summary_writer.add_scalar(key, value, epoch)
            for key, value in attention.items():
                summary_writer.add_image(
                    key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch
                )
            summary_writer.flush()

            torch.save(
                {
                    "fold": k,
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_path,
            )

            if argv.validate:
                logger.initialize(k)
                dataset.set_fold(k, train=False)
                for i, x in enumerate(dataloader):
                    with torch.no_grad():
                        dyn_a, sampling_points = bold.process_dynamic_fc(
                            x["timeseries"], argv.window_size, argv.window_stride, dynamic_length
                        )
                        sampling_endpoints = [p + argv.window_size for p in sampling_points]
                        if i == 0:
                            dyn_v = repeat(
                                torch.eye(dataset.num_nodes),
                                "n1 n2 -> b t n1 n2",
                                t=len(sampling_points),
                                b=argv.minibatch_size,
                            )
                        if dyn_v.shape[1] != dyn_a.shape[1]:
                            dyn_v = repeat(
                                torch.eye(dataset.num_nodes),
                                "n1 n2 -> b t n1 n2",
                                t=len(sampling_points),
                                b=argv.minibatch_size,
                            )
                        if dyn_a.shape[0] < argv.minibatch_size:
                            dyn_v = dyn_v[: dyn_a.shape[0]]
                        t = x["timeseries"].permute(1, 0, 2)
                        label = x["label"]
                        logit, loss, attention, latent, reg_ortho = step(
                            model=model,
                            criterion=criterion,
                            dyn_v=dyn_v,
                            dyn_a=dyn_a,
                            sampling_endpoints=sampling_endpoints,
                            t=t,
                            label=label,
                            reg_lambda=argv.reg_lambda,
                            clip_grad=argv.clip_grad,
                            device=device,
                            optimizer=None,
                            scheduler=None,
                        )
                        pred = logit.argmax(1)
                        prob = logit.softmax(1)
                        logger.add(
                            k=k,
                            pred=pred.detach().cpu().numpy(),
                            true=label.detach().cpu().numpy(),
                            prob=prob.detach().cpu().numpy(),
                        )
                samples = logger.get(k)
                metrics = logger.evaluate(k, print_metric=False)
                summary_writer_val.add_scalar("loss", loss_accumulate / len(dataloader), epoch)
                summary_writer_val.add_scalar("reg_ortho", reg_ortho_accumulate / len(dataloader), epoch)
                summary_writer_val.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1], epoch)
                for key, value in metrics.items():
                    summary_writer_val.add_scalar(key, value, epoch)
                summary_writer_val.flush()

        torch.save(model.state_dict(), os.path.join(argv.targetdir, "model", str(k), "model.pth"))
        checkpoint.update({"epoch": 0, "model": None, "optimizer": None, "scheduler": None})
        summary_writer.close()
        summary_writer_val.close()
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)


def test(argv):
    os.makedirs(os.path.join(argv.targetdir, "attention"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(argv, "use_synthetic", False):
        dataset = DatasetBiopointRestWithSynthetic(
            argv.sourcedir,
            synthetic_dir=argv.synthetic_dir,
            csv_path=argv.csv_path,
            k_fold=argv.k_fold,
            train_ratio=getattr(argv, "train_ratio", 0.8),
            dynamic_length=argv.dynamic_length,
            ts_filename_suffix=argv.ts_filename_suffix,
            atlas_source=getattr(argv, "atlas_source", "shen268"),
            dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
        )
    else:
        dataset = DatasetBiopointRest(
            argv.sourcedir,
            csv_path=argv.csv_path,
            k_fold=argv.k_fold,
            train_ratio=getattr(argv, "train_ratio", 0.8),
            dynamic_length=argv.dynamic_length,
            ts_filename_suffix=argv.ts_filename_suffix,
            atlas_source=getattr(argv, "atlas_source", "shen268"),
            dk_atlas_ts_root=getattr(argv, "dk_atlas_ts_root", None),
        )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=argv.num_workers, pin_memory=True
    )
    logger = LoggerSTAGIN(dataset.folds, dataset.num_classes)
    dynamic_length = argv.dynamic_length or dataset.num_timepoints

    single_fold = getattr(argv, "fold", None)
    folds_to_run = [single_fold] if single_fold is not None else dataset.folds

    for k in folds_to_run:
        os.makedirs(os.path.join(argv.targetdir, "attention", str(k)), exist_ok=True)
        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        )
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, "model", str(k), "model.pth")))

        summary_writer = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "test"))
        logger.initialize(k)
        dataset.set_fold(k, train=False)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        fold_attention = {"node_attention": [], "time_attention": []}

        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f"k:{k}")):
            with torch.no_grad():
                dyn_a, sampling_points = bold.process_dynamic_fc(
                    x["timeseries"], argv.window_size, argv.window_stride, dynamic_length
                )
                sampling_endpoints = [p + argv.window_size for p in sampling_points]
                if i == 0:
                    dyn_v = repeat(
                        torch.eye(dataset.num_nodes),
                        "n1 n2 -> b t n1 n2",
                        t=len(sampling_points),
                        b=argv.minibatch_size,
                    )
                if dyn_v.shape[1] != dyn_a.shape[1]:
                    dyn_v = repeat(
                        torch.eye(dataset.num_nodes),
                        "n1 n2 -> b t n1 n2",
                        t=len(sampling_points),
                        b=argv.minibatch_size,
                    )
                if dyn_a.shape[0] < argv.minibatch_size:
                    dyn_v = dyn_v[: dyn_a.shape[0]]
                t = x["timeseries"].permute(1, 0, 2)
                label = x["label"]
                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=torch.nn.CrossEntropyLoss(),
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger.add(
                    k=k,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                latent_accumulate.append(latent.detach().cpu().numpy())
                fold_attention["node_attention"].append(attention["node-attention"].detach().cpu().numpy())
                fold_attention["time_attention"].append(attention["time-attention"].detach().cpu().numpy())

        samples = logger.get(k)
        metrics = logger.evaluate(k, print_metric=False)
        summary_writer.add_scalar("loss", loss_accumulate / len(dataloader))
        summary_writer.add_scalar("reg_ortho", reg_ortho_accumulate / len(dataloader))
        summary_writer.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1])
        for key, value in metrics.items():
            summary_writer.add_scalar(key, value)
        for key, value in attention.items():
            summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True))
        summary_writer.flush()

        logger.to_csv(argv.targetdir, k)
        for key, value in fold_attention.items():
            torch.save(value, os.path.join(argv.targetdir, "attention", str(k), f"{key}.pth"))
        np.save(os.path.join(argv.targetdir, "attention", str(k), "latent.npy"), np.concatenate(latent_accumulate))

    if single_fold is None:
        logger.to_csv(argv.targetdir)
        final_metrics = logger.evaluate(print_metric=True)
        torch.save(logger.get(), os.path.join(argv.targetdir, "samples.pkl"))
    else:
        # Print metrics for the single fold
        logger.evaluate(single_fold, print_metric=True)
