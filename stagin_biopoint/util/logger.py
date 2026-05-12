# From https://github.com/egyptdj/stagin

import os
import csv
import numpy as np
from sklearn import metrics


class LoggerSTAGIN(object):
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.initialize(k=None)

    def __call__(self, **kwargs):
        if len(kwargs) == 0:
            self.get()
        else:
            self.add(**kwargs)

    def _initialize_metric_dict(self):
        return {"pred": [], "true": [], "prob": []}

    def _print_metric(self, metric):
        assert isinstance(metric, dict)
        spacer = len(max(metric, key=len))
        for key, value in metric.items():
            print(f"> {key:{spacer + 1}}: {value}")

    def initialize(self, k=None):
        if self.k_fold is None:
            self.samples = self._initialize_metric_dict()
        else:
            if k is None:
                self.samples = {_k: self._initialize_metric_dict() for _k in self.k_fold}
            else:
                self.samples[k] = self._initialize_metric_dict()

    def add(self, k=None, **kwargs):
        if self.k_fold is None:
            for sample, value in kwargs.items():
                self.samples[sample].append(value)
        else:
            assert k in self.k_fold
            for sample, value in kwargs.items():
                self.samples[k][sample].append(value)

    def get(self, k=None, initialize=False):
        if self.k_fold is None:
            true = np.concatenate(self.samples["true"])
            pred = np.concatenate(self.samples["pred"])
            prob = np.concatenate(self.samples["prob"])
        else:
            if k is None:
                true = {kv: np.concatenate(self.samples[kv]["true"]) for kv in self.k_fold}
                pred = {kv: np.concatenate(self.samples[kv]["pred"]) for kv in self.k_fold}
                prob = {kv: np.concatenate(self.samples[kv]["prob"]) for kv in self.k_fold}
            else:
                true = np.concatenate(self.samples[k]["true"])
                pred = np.concatenate(self.samples[k]["pred"])
                prob = np.concatenate(self.samples[k]["prob"])

        if initialize:
            self.initialize(k)

        return dict(true=true, pred=pred, prob=prob)

    def evaluate(self, k=None, initialize=False, option="mean", print_metric=True):
        samples = self.get(k)
        if self.num_classes == 1:
            if self.k_fold is not None and k is None:
                aggregate = np.mean if option == "mean" else np.std
                explained_var = aggregate([metrics.explained_variance_score(samples["true"][kv], samples["pred"][kv]) for kv in self.k_fold])
                r2 = aggregate([metrics.r2_score(samples["true"][kv], samples["pred"][kv]) for kv in self.k_fold])
                mse = aggregate([metrics.mean_squared_error(samples["true"][kv], samples["pred"][kv]) for kv in self.k_fold])
            else:
                explained_var = metrics.explained_variance_score(samples["true"], samples["pred"])
                r2 = metrics.r2_score(samples["true"], samples["pred"])
                mse = metrics.mean_squared_error(samples["true"], samples["pred"])

            if initialize:
                self.initialize(k)
            metric = dict(explained_var=explained_var, r2=r2, mse=mse)
            if print_metric:
                self._print_metric(metric)
            return metric

        elif self.num_classes > 1:
            if self.k_fold is not None and k is None:
                aggregate = np.mean if option == "mean" else np.std
                accuracy = aggregate([metrics.accuracy_score(samples["true"][kv], samples["pred"][kv]) for kv in self.k_fold])
                precision = aggregate([metrics.precision_score(samples["true"][kv], samples["pred"][kv], average="binary" if self.num_classes == 2 else "micro") for kv in self.k_fold])
                recall = aggregate([metrics.recall_score(samples["true"][kv], samples["pred"][kv], average="binary" if self.num_classes == 2 else "micro") for kv in self.k_fold])
                if self.num_classes == 2:
                    roc_auc = aggregate([metrics.roc_auc_score(samples["true"][kv], samples["prob"][kv][:, 1]) for kv in self.k_fold])
                else:
                    roc_auc = np.mean([metrics.roc_auc_score(samples["true"][kv], samples["prob"][kv], average="macro", multi_class="ovr") for kv in self.k_fold])
            else:
                accuracy = metrics.accuracy_score(samples["true"], samples["pred"])
                precision = metrics.precision_score(samples["true"], samples["pred"], average="binary" if self.num_classes == 2 else "micro")
                recall = metrics.recall_score(samples["true"], samples["pred"], average="binary" if self.num_classes == 2 else "micro")
                if self.num_classes == 2:
                    roc_auc = metrics.roc_auc_score(samples["true"], samples["prob"][:, 1])
                else:
                    roc_auc = metrics.roc_auc_score(samples["true"], samples["prob"], average="macro", multi_class="ovr")

            if initialize:
                self.initialize(k)
            metric = dict(accuracy=accuracy, precision=precision, recall=recall, roc_auc=roc_auc)
            if print_metric:
                self._print_metric(metric)
            return metric
        else:
            raise ValueError("num_classes must be 1 or > 1")

    def to_csv(self, targetdir, k=None, initialize=False):
        metric_dict = self.evaluate(k, initialize, print_metric=False)
        append = os.path.isfile(os.path.join(targetdir, "metric.csv"))
        with open(os.path.join(targetdir, "metric.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(["fold"] + list(metric_dict.keys()))
            writer.writerow([str(k)] + list(metric_dict.values()))
        if k is None:
            with open(os.path.join(targetdir, "metric.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([str(k)] + list(self.evaluate(k, initialize, "std", print_metric=False).values()))
