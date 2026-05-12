# FC computation for building graph from time series

import torch


def corrcoef(x: torch.Tensor) -> torch.Tensor:
    """x: (num_roi, window_len). Returns (num_roi, num_roi) correlation matrix."""
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1 + 1e-9)
    d = torch.diag(c)
    stddev = torch.pow(d.clamp(min=0) + 1e-9, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def get_fc(timeseries: torch.Tensor, sampling_point: int, window_size: int, self_loop: bool = False) -> torch.Tensor:
    """timeseries (T, num_roi). Returns (num_roi, num_roi) FC."""
    segment = timeseries[sampling_point: sampling_point + window_size].T  # (num_roi, window_size)
    fc = corrcoef(segment)
    if not self_loop:
        fc = fc - torch.eye(fc.shape[0], device=fc.device, dtype=fc.dtype)
    return fc
