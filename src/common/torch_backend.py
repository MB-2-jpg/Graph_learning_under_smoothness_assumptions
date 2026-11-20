from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["available", "device_auto", "describe", "fbf_batch"]


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def device_auto() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe(device: Optional[str] = None) -> str:
    import torch
    device = device or device_auto()
    if device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return (f"{p.name}, {p.total_memory/2**30:.1f} GiB, "
                f"{torch.cuda.device_count()} device(s)")
    return f"cpu, {torch.get_num_threads()} threads"


class _Op:
    def __init__(self, m, device, dtype):
        import torch
        i, j = np.triu_indices(m, k=1)
        self.m, self.E = m, i.size
        self.I = torch.as_tensor(i, device=device, dtype=torch.long)
        self.J = torch.as_tensor(j, device=device, dtype=torch.long)
        self.device, self.dtype = device, dtype

    def S(self, v):                       # (B, E) -> (B, m), scatter-add
        import torch
        out = torch.zeros(v.shape[0], self.m, device=v.device, dtype=v.dtype)
        out.index_add_(1, self.I, v)
        out.index_add_(1, self.J, v)
        return out

    def St(self, d):                      # (B, m) -> (B, E), gather
        return d[:, self.I] + d[:, self.J]


def fbf_batch(z, first: Sequence[float], second: Sequence[float], m: int,
              model: str = "log", max_iter: int = 20_000, eps: float = 1e-4,
              device: Optional[str] = None, dtype: str = "float64",
              check_every: int = 25, progress: bool = False
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    device = device or device_auto()
    td = torch.float32 if str(dtype) in ("float32", "f32") else torch.float64
    dev = torch.device(device)

    a = torch.as_tensor(np.asarray(first, float), device=dev, dtype=td).view(-1, 1)
    b = torch.as_tensor(np.asarray(second, float), device=dev, dtype=td).view(-1, 1)
    B = a.shape[0]
    if b.shape[0] != B:
        raise ValueError("the two parameter lists must have the same length")

    zt = torch.as_tensor(np.ascontiguousarray(z), device=dev, dtype=td).view(1, -1)
    E = zt.shape[1]
    op = _Op(m, dev, td)

    if model == "log":
        g = 0.9 / (2.0 * b + float(np.sqrt(2.0 * (m - 1))))
    elif model == "dong":
        g = 0.9 / (4.0 * a * m + 2.0 * float(np.sqrt(E)))   # zeta = 4*alpha*m
    else:
        raise ValueError(f"unknown model {model!r}")

    w = torch.zeros(B, E, device=dev, dtype=td)
    d = torch.zeros(B, m if model == "log" else 1, device=dev, dtype=td)

    iters = torch.full((B,), max_iter, device=dev, dtype=torch.long)
    done = torch.zeros(B, device=dev, dtype=torch.bool)

    def grad_f3(v):                                    # model (14) only
        return a * (4.0 * v + 2.0 * op.St(op.S(v)))

    for it in range(max_iter):
        if model == "log":
            y   = w - g * (2.0 * b * w + op.St(d))
            y_b = d + g * op.S(w)
            p   = torch.clamp(y - 2.0 * g * zt, min=0.0)
            p_b = (y_b - torch.sqrt(y_b * y_b + 4.0 * a * g)) / 2.0
            q   = p - g * (2.0 * b * p + op.St(p_b))
            q_b = p_b + g * op.S(p)
        else:
            y   = w - g * (grad_f3(w) + 2.0 * d)
            y_b = d + g * (2.0 * w.sum(1, keepdim=True))
            p   = torch.clamp(y - 2.0 * g * zt, min=0.0)
            p_b = y_b - g * b
            q   = p - g * (grad_f3(p) + 2.0 * p_b)
            q_b = p_b + g * (2.0 * p.sum(1, keepdim=True))

        w_new, d_new = w - y + q, d - y_b + q_b

        if it > 1 and it % check_every == 0:
            dw = torch.linalg.norm(w_new - w, dim=1)
            dd = torch.linalg.norm(d_new - d, dim=1)
            nw = torch.clamp(torch.linalg.norm(w, dim=1), min=1e-12)
            nd = torch.clamp(torch.linalg.norm(d, dim=1), min=1e-12)
            hit = (dw <= eps * nw) & (dd <= eps * nd) & ~done
            if hit.any():
                iters[hit] = it + 1
                done |= hit
            if bool(done.all()):
                w, d = w_new, d_new
                break
            if progress:
                print(f"    iter {it+1}: {int(done.sum())}/{B} converged", flush=True)

        w, d = w_new, d_new

    w = torch.clamp(w, min=0.0)
    return (w.cpu().numpy().astype(np.float64),
            d.cpu().numpy().astype(np.float64),
            iters.cpu().numpy())
