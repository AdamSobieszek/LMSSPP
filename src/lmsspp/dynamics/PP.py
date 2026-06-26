"""Core Peszek-Poyato numerical dynamics backend.

This module contains the reusable PP interaction kernels, FFT/torch field
evaluators, grid interpolation helpers, and timestep/backend utilities. Higher
level simulation orchestration, widgets, dashboards, and experiment schemas
remain in ``pp_cs_equilibria.py`` for compatibility.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Sequence

import numpy as np

try:  # Optional accelerator backend; NumPy remains the required baseline.
    import torch
except Exception:  # pragma: no cover - torch is an optional extra
    torch = None  # type: ignore[assignment]

Array = np.ndarray
DTypeChoice = Literal["auto", "float32", "float64"]

AUTO_K_REFERENCE_ALPHA = 0.99
AUTO_K_REFERENCE_DELTA = 1.0 - AUTO_K_REFERENCE_ALPHA
AUTO_K_OPT_SAFETY_FACTOR = 1.003
AUTO_K_OPT_TRACTION_LIMIT = 1.5417376823742555
AUTO_K_OPT_EXP_AMPLITUDE = 1.7729528862175705
AUTO_K_OPT_EXP_AMPLITUDE = 1.7029528862175705
AUTO_K_OPT_EXP_RATE = 2.2877960408582667
AUTO_K_OPT_EXP_RATE = 2.301960408582667


def _resolve_auto_pp_K(alpha: float) -> float:
    """Resolve the fitted automatic PP coupling for ``alpha < 1``."""

    delta = 1.0 - float(alpha)
    if delta > 0.0:
        x = -float(np.log10(delta))
        correction = AUTO_K_OPT_TRACTION_LIMIT - AUTO_K_OPT_EXP_AMPLITUDE * np.exp(-AUTO_K_OPT_EXP_RATE * x)
        correction = max(0.0, float(correction))
        return float(delta * AUTO_K_OPT_SAFETY_FACTOR * correction)
    if abs(delta) < 1e-15:
        return float("inf")
    return 1.0


def resolve_pp_K(alpha: float, K: float | None) -> float:
    """Resolve the PP coupling strength.

    Passing ``K=None`` uses the empirical fixed-scale fit

        x = -log10(1 - alpha)
        K_auto = 1.003 * (1 - alpha) * (1.5417376823742555 - 1.7729528862175705 * exp(-2.2877960408582667 * x))

    for ``alpha < 1``. The fit is calibrated for high-alpha experiments; values
    outside that range are clipped to keep the automatic coupling non-negative.
    At the singular metadata point ``alpha=1`` this returns ``inf`` for
    ``K=None``. Kernel computations use
    :func:`resolve_pp_traction_scale` instead of this display/logging value.
    """

    alpha_f = float(alpha)
    if K is None:
        K_f = _resolve_auto_pp_K(alpha_f)
        if abs(1.0 - alpha_f) < 1e-15:
            return K_f
    else:
        K_f = float(K)
    if not np.isfinite(K_f):
        raise ValueError("K must be finite or None")
    return float(K_f)


def resolve_pp_traction_scale(alpha: float, K: float | None) -> float:
    """Resolve the scalar multiplying the PP kernel derivatives.

    The PP gradient and Hessian use the combined multiplier ``K / (1-alpha)``.
    Resolving this value once avoids unstable two-step kernel construction
    near singular metadata values while keeping ``K`` itself available for
    reporting and visualization.
    """

    alpha_f = float(alpha)
    denom = 1.0 - alpha_f
    if abs(denom) < 1e-15:
        if K is None:
            return float(AUTO_K_OPT_SAFETY_FACTOR * AUTO_K_OPT_TRACTION_LIMIT)
        return float(K)
    K_f = resolve_pp_K(alpha_f, K)
    traction = K_f / denom
    if not np.isfinite(traction):
        raise ValueError("PP traction scale K/(1-alpha) must be finite")
    return float(traction)


class FFTPeszekPoyato2D:
    """Grid/FFT evaluator for the PP interaction field and Hessian metric."""

    def __init__(self, alpha: float, K: float | None, grid_size: int, domain_radius: float):
        if not 0.0 <= alpha <= 2.0:
            raise ValueError("alpha must lie in [0, 2] for the PP kernel normalization")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if domain_radius <= 0:
            raise ValueError("domain_radius must be positive")

        self.alpha = float(alpha)
        self.K = resolve_pp_K(self.alpha, K)
        self.traction_scale = resolve_pp_traction_scale(self.alpha, K)
        self.G = int(grid_size)
        self.L = float(domain_radius)
        self.h = 2 * self.L / self.G
        self.P = 2 * self.G
        self.backend_name = "numpy"
        self.device_name = "cpu"
        self.dtype_name = "float64"

        kernels = self._build_kernels()
        self.fft_Kx = np.fft.rfft2(kernels[0])
        self.fft_Ky = np.fft.rfft2(kernels[1])
        self.fft_Hxx = np.fft.rfft2(kernels[2])
        self.fft_Hxy = np.fft.rfft2(kernels[3])
        self.fft_Hyy = np.fft.rfft2(kernels[4])
        self.fft_W = np.fft.rfft2(kernels[5])
        self._padded_work = np.zeros((self.P, self.P), dtype=np.float64)

    def clip_inside(self, x: Array) -> Array:
        margin = 2.1 * self.h
        return np.clip(x, -self.L + margin, self.L - margin)

    def clip_inside_with_count(self, x: Array) -> tuple[Array, int]:
        margin = 2.1 * self.h
        lo = -self.L + margin
        hi = self.L - margin
        clipped = np.any((x < lo) | (x > hi), axis=1)
        return np.clip(x, lo, hi), int(np.count_nonzero(clipped))

    def asarray(self, x: Array) -> Array:
        return np.asarray(x, dtype=np.float64)

    def copy_state(self, x: Array) -> Array:
        return np.array(x, dtype=np.float64, copy=True)

    def to_numpy(self, x: Any) -> Array:
        return np.asarray(x, dtype=np.float64)

    def center(self, x: Array) -> Array:
        return x - x.mean(axis=0, keepdims=True)

    def speed_stats(self, v: Array) -> tuple[float, float]:
        speed2 = np.sum(v * v, axis=1)
        return float(np.sqrt(np.mean(speed2))), float(np.sqrt(np.max(speed2)))

    def rms_delta(self, a: Array, b: Array) -> float:
        delta = a - b
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    def synchronize(self) -> None:
        return None

    def convolve(self, rho_grid: Array, fft_kernel: Array) -> Array:
        padded = self._zero_padded_grid(rho_grid)
        conv = np.fft.irfft2(np.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        return conv[: self.G, : self.G]

    def convolve_fields(self, rho_grid: Array, fft_kernels: Sequence[Array]) -> tuple[Array, ...]:
        padded = self._zero_padded_grid(rho_grid)
        rho_hat = np.fft.rfft2(padded)
        return tuple(np.fft.irfft2(rho_hat * fft_kernel, s=(self.P, self.P))[: self.G, : self.G] for fft_kernel in fft_kernels)

    def _zero_padded_grid(self, rho_grid: Array) -> Array:
        padded = self._padded_work
        padded.fill(0.0)
        padded[: self.G, : self.G] = rho_grid
        return padded

    def A_grid_from_particles(self, x: Array) -> tuple[Array, Array, Array]:
        rho_grid = deposit_mass(x, self.G, self.L)
        Ax_grid, Ay_grid = self.convolve_fields(rho_grid, (self.fft_Kx, self.fft_Ky))
        return rho_grid, Ax_grid, Ay_grid

    def A_at_particles(self, x: Array) -> tuple[Array, Array]:
        rho_grid, Ax_grid, Ay_grid = self.A_grid_from_particles(x)
        weights = cic_indices_weights(x, self.G, self.L)
        A = np.c_[interp_grid_with_weights(Ax_grid, weights), interp_grid_with_weights(Ay_grid, weights)]
        return A, rho_grid

    def velocity(self, x: Array, omega: Array) -> tuple[Array, Array]:
        A, _ = self.A_at_particles(x)
        return omega - A, A

    def projective_external(self, x: Array, omega: Array, eps: float) -> Array:
        """Projective external drift E_eps(omega, x) = eps^{-1} S[eps*omega, x]."""

        return _projective_external_numpy(np.asarray(x, dtype=np.float64), np.asarray(omega, dtype=np.float64), float(eps))

    def hessian_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array, Array]:
        return self.convolve_fields(rho_grid, (self.fft_Hxx, self.fft_Hxy, self.fft_Hyy))  # type: ignore[return-value]

    def W_grid_from_particles(self, x: Array) -> tuple[Array, Array]:
        rho_grid = deposit_mass(x, self.G, self.L)
        W_grid = self.convolve(rho_grid, self.fft_W)
        return rho_grid, W_grid

    def _build_kernels(self) -> tuple[Array, Array, Array, Array, Array, Array]:
        coords = make_lag_coords(self.P, self.h)
        Xlag, Ylag = np.meshgrid(coords, coords, indexing="ij")
        R = np.sqrt(Xlag**2 + Ylag**2)
        mask = R > 1e-14

        Kx = np.zeros((self.P, self.P), dtype=np.float64)
        Ky = np.zeros((self.P, self.P), dtype=np.float64)
        scale_grad = np.zeros_like(R)
        scale_grad[mask] = self.traction_scale * (R[mask] ** (-self.alpha))
        Kx[mask] = Xlag[mask] * scale_grad[mask]
        Ky[mask] = Ylag[mask] * scale_grad[mask]

        ex = np.zeros_like(R)
        ey = np.zeros_like(R)
        ex[mask] = Xlag[mask] / R[mask]
        ey[mask] = Ylag[mask] / R[mask]

        scale_hess = np.zeros_like(R)
        scale_hess[mask] = self.traction_scale * (R[mask] ** (-self.alpha))
        Hxx = np.zeros((self.P, self.P), dtype=np.float64)
        Hxy = np.zeros((self.P, self.P), dtype=np.float64)
        Hyy = np.zeros((self.P, self.P), dtype=np.float64)
        Hxx[mask] = scale_hess[mask] * (1 - self.alpha * ex[mask] * ex[mask])
        Hxy[mask] = scale_hess[mask] * (-self.alpha * ex[mask] * ey[mask])
        Hyy[mask] = scale_hess[mask] * (1 - self.alpha * ey[mask] * ey[mask])
        W = np.zeros((self.P, self.P), dtype=np.float64)
        if abs(self.alpha - 2.0) < 1e-9:
            W[mask] = self.traction_scale * np.log(R[mask])
        else:
            W[mask] = self.traction_scale * (R[mask] ** (2 - self.alpha) - 1.0) / (2 - self.alpha)
        return Kx, Ky, Hxx, Hxy, Hyy, W


class FFTPeszekPoyatoDensity2D:
    """Grid/FFT evaluator for continuous-density PP fields and diagnostics."""

    _BOUNDED_PAD_FACTOR = 4

    def __init__(self, alpha: float, K: float | None, grid_size: int, domain_radius: float):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("continuous density requires 0 <= alpha < 1")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if domain_radius <= 0:
            raise ValueError("domain_radius must be positive")

        self.alpha = float(alpha)
        self.K = resolve_pp_K(self.alpha, K)
        self.traction_scale = resolve_pp_traction_scale(self.alpha, K)
        self.G = int(grid_size)
        self.L = float(domain_radius)
        self.h = 2 * self.L / self.G
        self.P = self._BOUNDED_PAD_FACTOR * self.G
        self._embed_offset = (self.P - self.G) // 2
        self.cell_area = self.h * self.h

        w_kernel = self._build_w_kernel()
        self.fft_W = np.fft.rfft2(w_kernel)

    def marginal_from_fibers(self, r_fiber: Array, nu: Array) -> Array:
        return np.tensordot(nu, r_fiber, axes=(0, 0))

    def _mass_grid_from_density(self, rho_grid: Array) -> Array:
        return rho_grid * self.cell_area

    def _embed_mass(self, mass_grid: Array) -> Array:
        padded = np.zeros((self.P, self.P), dtype=np.float64)
        off = self._embed_offset
        padded[off : off + self.G, off : off + self.G] = mass_grid
        return padded

    def convolve(self, mass_grid: Array, fft_kernel: Array) -> Array:
        padded = self._embed_mass(mass_grid)
        conv = np.fft.irfft2(np.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        off = self._embed_offset
        return conv[off : off + self.G, off : off + self.G]

    def W_grid_from_rho(self, rho_grid: Array) -> Array:
        return self.convolve(self._mass_grid_from_density(rho_grid), self.fft_W)

    def A_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array]:
        """A_rho = grad(W^alpha * rho) on the grid, using one scalar bounded FFT pass."""
        w_field = self.W_grid_from_rho(rho_grid)
        ax_grid = np.gradient(w_field, self.h, axis=0)
        ay_grid = np.gradient(w_field, self.h, axis=1)
        return ax_grid, ay_grid

    def hessian_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array, Array]:
        """Hessian of the scalar interaction potential, consistent with A = grad W."""
        w_field = self.W_grid_from_rho(rho_grid)
        ax_grid = np.gradient(w_field, self.h, axis=0)
        ay_grid = np.gradient(w_field, self.h, axis=1)
        hxx = np.gradient(ax_grid, self.h, axis=0)
        hxy = np.gradient(ax_grid, self.h, axis=1)
        hyy = np.gradient(ay_grid, self.h, axis=1)
        return hxx, hxy, hyy

    def _build_w_kernel(self) -> Array:
        coords = make_lag_coords(self.P, self.h)
        xlag, ylag = np.meshgrid(coords, coords, indexing="ij")
        r = np.sqrt(xlag**2 + ylag**2)
        mask = r > 1e-14
        w = np.zeros((self.P, self.P), dtype=np.float64)
        if abs(self.alpha - 2.0) < 1e-9:
            w[mask] = self.traction_scale * np.log(r[mask])
        else:
            w[mask] = self.traction_scale * (r[mask] ** (2 - self.alpha) - 1.0) / (2 - self.alpha)
        return w


class TorchPeszekPoyato2D:
    """Torch-backed PP field evaluator for CPU, CUDA, and MPS devices."""

    def __init__(
        self,
        alpha: float,
        K: float | None,
        grid_size: int,
        domain_radius: float,
        *,
        device: Any,
        dtype: Any,
    ):
        if torch is None:  # pragma: no cover - guarded by backend selection
            raise RuntimeError("torch is not available")
        if not 0.0 <= alpha <= 2.0:
            raise ValueError("alpha must lie in [0, 2] for the PP kernel normalization")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if domain_radius <= 0:
            raise ValueError("domain_radius must be positive")

        self.alpha = float(alpha)
        self.K = resolve_pp_K(self.alpha, K)
        self.traction_scale = resolve_pp_traction_scale(self.alpha, K)
        self.G = int(grid_size)
        self.L = float(domain_radius)
        self.h = 2 * self.L / self.G
        self.P = 2 * self.G
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend_name = "torch"
        self.device_name = str(self.device)
        self.dtype_name = _torch_dtype_name(dtype)

        kernels = self._build_kernels()
        self.fft_Kx = torch.fft.rfft2(kernels[0])
        self.fft_Ky = torch.fft.rfft2(kernels[1])
        self.fft_Hxx = torch.fft.rfft2(kernels[2])
        self.fft_Hxy = torch.fft.rfft2(kernels[3])
        self.fft_Hyy = torch.fft.rfft2(kernels[4])
        self.fft_W = torch.fft.rfft2(kernels[5])

    def asarray(self, x: Array) -> Any:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def copy_state(self, x: Any) -> Any:
        return x.clone()

    def to_numpy(self, x: Any) -> Array:
        self.synchronize()
        return x.detach().cpu().numpy().astype(np.float64, copy=False)

    def clip_inside(self, x: Any) -> Any:
        margin = 2.1 * self.h
        return torch.clamp(x, -self.L + margin, self.L - margin)

    def clip_inside_with_count(self, x: Any) -> tuple[Any, int]:
        margin = 2.1 * self.h
        lo = -self.L + margin
        hi = self.L - margin
        clipped = torch.any((x < lo) | (x > hi), dim=1)
        return torch.clamp(x, lo, hi), int(torch.count_nonzero(clipped).detach().cpu().item())

    def center(self, x: Any) -> Any:
        return x - x.mean(dim=0, keepdim=True)

    def speed_stats(self, v: Any) -> tuple[float, float]:
        speed2 = torch.sum(v * v, dim=1)
        rms = torch.sqrt(torch.mean(speed2))
        maxv = torch.sqrt(torch.max(speed2))
        return float(rms.detach().cpu().item()), float(maxv.detach().cpu().item())

    def rms_delta(self, a: Any, b: Any) -> float:
        delta = a - b
        rms = torch.sqrt(torch.mean(torch.sum(delta * delta, dim=1)))
        return float(rms.detach().cpu().item())

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def convolve(self, rho_grid: Any, fft_kernel: Any) -> Any:
        padded = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        padded[: self.G, : self.G] = rho_grid
        conv = torch.fft.irfft2(torch.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        return conv[: self.G, : self.G]

    def convolve_fields(self, rho_grid: Any, fft_kernels: Sequence[Any]) -> tuple[Any, ...]:
        padded = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        padded[: self.G, : self.G] = rho_grid
        rho_hat = torch.fft.rfft2(padded)
        return tuple(torch.fft.irfft2(rho_hat * fft_kernel, s=(self.P, self.P))[: self.G, : self.G] for fft_kernel in fft_kernels)

    def A_grid_from_particles(self, x: Any) -> tuple[Any, Any, Any]:
        rho_grid = self.deposit_mass(x)
        Ax_grid, Ay_grid = self.convolve_fields(rho_grid, (self.fft_Kx, self.fft_Ky))
        return rho_grid, Ax_grid, Ay_grid

    def A_at_particles(self, x: Any) -> tuple[Any, Any]:
        rho_grid, Ax_grid, Ay_grid = self.A_grid_from_particles(x)
        weights = self.cic_indices_weights(x)
        A = torch.stack(
            [
                self.interp_grid_with_weights(Ax_grid, weights),
                self.interp_grid_with_weights(Ay_grid, weights),
            ],
            dim=1,
        )
        return A, rho_grid

    def velocity(self, x: Any, omega: Any) -> tuple[Any, Any]:
        A, _ = self.A_at_particles(x)
        return omega - A, A

    def projective_external(self, x: Any, omega: Any, eps: float) -> Any:
        """Projective external drift E_eps(omega, x) = eps^{-1} S[eps*omega, x]."""

        eps = float(eps)
        if eps == 0.0:
            return omega
        omega_sq = torch.sum(omega * omega, dim=1, keepdim=True)
        x_sq = torch.sum(x * x, dim=1, keepdim=True)
        dot = torch.sum(omega * x, dim=1, keepdim=True)
        kappa = 1.0 - 2.0 * eps * dot + (eps * eps) * omega_sq * x_sq
        return (omega - eps * omega_sq * x) / kappa

    def hessian_grid_from_rho(self, rho_grid: Any) -> tuple[Any, Any, Any]:
        return self.convolve_fields(rho_grid, (self.fft_Hxx, self.fft_Hxy, self.fft_Hyy))  # type: ignore[return-value]

    def W_grid_from_particles(self, x: Any) -> tuple[Any, Any]:
        rho_grid = self.deposit_mass(x)
        W_grid = self.convolve(rho_grid, self.fft_W)
        return rho_grid, W_grid

    def cic_indices_weights(self, x: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        h = 2 * self.L / self.G
        u = (x[:, 0] + self.L) / h
        v = (x[:, 1] + self.L) / h
        i = torch.floor(u).to(torch.int64).clamp(0, self.G - 2)
        j = torch.floor(v).to(torch.int64).clamp(0, self.G - 2)
        fu = u - i.to(dtype=x.dtype)
        fv = v - j.to(dtype=x.dtype)
        return i, j, (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv

    def deposit_mass(self, x: Any) -> Any:
        if int(x.shape[0]) == 0:
            raise ValueError("cannot deposit an empty particle set")
        i, j, w00, w10, w01, w11 = self.cic_indices_weights(x)
        mass = 1.0 / int(x.shape[0])
        grid_flat = torch.zeros((self.G * self.G,), dtype=self.dtype, device=self.device)
        grid_flat.index_add_(0, i * self.G + j, mass * w00)
        grid_flat.index_add_(0, (i + 1) * self.G + j, mass * w10)
        grid_flat.index_add_(0, i * self.G + j + 1, mass * w01)
        grid_flat.index_add_(0, (i + 1) * self.G + j + 1, mass * w11)
        return grid_flat.reshape(self.G, self.G)

    def interp_grid_with_weights(self, field: Any, weights: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        i, j, w00, w10, w01, w11 = weights
        return field[i, j] * w00 + field[i + 1, j] * w10 + field[i, j + 1] * w01 + field[i + 1, j + 1] * w11

    def _build_kernels(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        coords = _torch_lag_coords(self.P, self.h, self.device, self.dtype)
        Xlag, Ylag = torch.meshgrid(coords, coords, indexing="ij")
        R = torch.sqrt(Xlag * Xlag + Ylag * Ylag)
        mask = R > 1e-14

        Kx = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        Ky = torch.zeros_like(Kx)
        scale_grad = torch.zeros_like(R)
        scale_grad[mask] = self.traction_scale * torch.pow(R[mask], -self.alpha)
        Kx[mask] = Xlag[mask] * scale_grad[mask]
        Ky[mask] = Ylag[mask] * scale_grad[mask]

        ex = torch.zeros_like(R)
        ey = torch.zeros_like(R)
        ex[mask] = Xlag[mask] / R[mask]
        ey[mask] = Ylag[mask] / R[mask]

        scale_hess = torch.zeros_like(R)
        scale_hess[mask] = self.traction_scale * torch.pow(R[mask], -self.alpha)
        Hxx = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        Hxy = torch.zeros_like(Hxx)
        Hyy = torch.zeros_like(Hxx)
        Hxx[mask] = scale_hess[mask] * (1 - self.alpha * ex[mask] * ex[mask])
        Hxy[mask] = scale_hess[mask] * (-self.alpha * ex[mask] * ey[mask])
        Hyy[mask] = scale_hess[mask] * (1 - self.alpha * ey[mask] * ey[mask])
        W = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        if abs(self.alpha - 2.0) < 1e-9:
            W[mask] = self.traction_scale * torch.log(R[mask])
        else:
            W[mask] = self.traction_scale * (torch.pow(R[mask], 2 - self.alpha) - 1.0) / (2 - self.alpha)
        return Kx, Ky, Hxx, Hxy, Hyy, W


def make_lag_coords(P: int, h: float) -> Array:
    idx = np.arange(P)
    lag = np.where(idx <= P // 2, idx, idx - P)
    return lag * h


def _torch_lag_coords(P: int, h: float, device: Any, dtype: Any) -> Any:
    if torch is None:  # pragma: no cover - guarded by backend construction
        raise RuntimeError("torch is not available")
    idx = torch.arange(P, device=device)
    lag = torch.where(idx <= P // 2, idx, idx - P)
    return lag.to(dtype=dtype) * h


def _torch_dtype_name(dtype: Any) -> str:
    if torch is not None and dtype is torch.float32:
        return "float32"
    if torch is not None and dtype is torch.float64:
        return "float64"
    return str(dtype).replace("torch.", "")


def cic_indices_weights(x: Array, G: int, L: float) -> tuple[Array, Array, Array, Array, Array, Array]:
    h = 2 * L / G
    u = (x[:, 0] + L) / h
    v = (x[:, 1] + L) / h
    i = np.floor(u).astype(np.int64)
    j = np.floor(v).astype(np.int64)
    i = np.clip(i, 0, G - 2)
    j = np.clip(j, 0, G - 2)
    fu = u - i
    fv = v - j
    return i, j, (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv


def deposit_mass(x: Array, G: int, L: float) -> Array:
    if len(x) == 0:
        raise ValueError("cannot deposit an empty particle set")
    grid = np.zeros((G, G), dtype=np.float64)
    i, j, w00, w10, w01, w11 = cic_indices_weights(x, G, L)
    mass = 1.0 / len(x)
    np.add.at(grid, (i, j), mass * w00)
    np.add.at(grid, (i + 1, j), mass * w10)
    np.add.at(grid, (i, j + 1), mass * w01)
    np.add.at(grid, (i + 1, j + 1), mass * w11)
    return grid


def interp_grid(field: Array, x: Array, G: int, L: float) -> Array:
    return interp_grid_with_weights(field, cic_indices_weights(x, G, L))


def interp_grid_with_weights(field: Array, weights: tuple[Array, Array, Array, Array, Array, Array]) -> Array:
    i, j, w00, w10, w01, w11 = weights
    return (
        field[i, j] * w00
        + field[i + 1, j] * w10
        + field[i, j + 1] * w01
        + field[i + 1, j + 1] * w11
    )


def cs_kappa(a: Array, b: Array) -> Array:
    """Conformal denominator kappa[a, b] = 1 - 2<a,b> + |a|^2 |b|^2."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=-1)
    a_sq = np.sum(a * a, axis=-1)
    b_sq = np.sum(b * b, axis=-1)
    return 1.0 - 2.0 * dot + a_sq * b_sq


def cs_cost(a: Array, b: Array) -> Array:
    """Symmetric logarithmic projective cost c_S(a, b) = -1/2 log kappa[a, b]."""

    return -0.5 * np.log(cs_kappa(a, b))


def inversive_bracket(a: Array, b: Array) -> Array:
    """Inversive bracket S[a, b] = (a - |a|^2 b) / kappa[a, b]."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_sq = np.sum(a * a, axis=-1, keepdims=True)
    kappa = cs_kappa(a, b)[..., None]
    return (a - a_sq * b) / kappa


def _projective_external_numpy(x: Array, omega: Array, eps: float) -> Array:
    eps = float(eps)
    if eps == 0.0:
        return omega
    omega_sq = np.sum(omega * omega, axis=1, keepdims=True)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    dot = np.sum(omega * x, axis=1, keepdims=True)
    kappa = 1.0 - 2.0 * eps * dot + (eps * eps) * omega_sq * x_sq
    return (omega - eps * omega_sq * x) / kappa


def projective_external_field(omega: Array, x: Array, eps: float) -> Array:
    """External drift of the projective Peszek-Poyato gauge."""

    return _projective_external_numpy(np.asarray(x, dtype=np.float64), np.asarray(omega, dtype=np.float64), float(eps))


def direct_hessian_at(
    points: Array,
    sources: Array,
    alpha: float,
    K: float | None,
    chunk: int = 128,
) -> Array:
    """Direct finite-particle Hessian metric at query points."""

    traction_scale = resolve_pp_traction_scale(alpha, K)
    out = np.zeros((points.shape[0], 2, 2), dtype=np.float64)
    N = sources.shape[0]
    for a in range(0, points.shape[0], chunk):
        p = points[a : a + chunk]
        diff = p[:, None, :] - sources[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        mask = r > 1e-14

        ex = np.zeros_like(r)
        ey = np.zeros_like(r)
        ex[mask] = diff[..., 0][mask] / r[mask]
        ey[mask] = diff[..., 1][mask] / r[mask]

        scale = np.zeros_like(r)
        scale[mask] = traction_scale * (r[mask] ** (-alpha)) / N

        Hxx = scale * (1 - alpha * ex * ex)
        Hxy = scale * (-alpha * ex * ey)
        Hyy = scale * (1 - alpha * ey * ey)

        out[a : a + chunk, 0, 0] = Hxx.sum(axis=1)
        out[a : a + chunk, 0, 1] = Hxy.sum(axis=1)
        out[a : a + chunk, 1, 0] = Hxy.sum(axis=1)
        out[a : a + chunk, 1, 1] = Hyy.sum(axis=1)

    return out


def _make_pp_backend(config: Any) -> FFTPeszekPoyato2D | TorchPeszekPoyato2D:
    backend = str(config.backend)
    if backend == "auto":
        if torch is not None and _torch_accelerator_available():
            backend = "torch"
        else:
            if torch is None:
                warnings.warn("torch is unavailable; falling back to the NumPy PP backend.", RuntimeWarning, stacklevel=2)
            backend = "numpy"

    if backend == "numpy":
        if config.device is not None:
            warnings.warn("SimulationConfig.device is ignored by the NumPy PP backend.", RuntimeWarning, stacklevel=2)
        if config.dtype == "float32":
            warnings.warn("The NumPy PP backend preserves the legacy float64 path; dtype='float32' is ignored.", RuntimeWarning, stacklevel=2)
        return FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)

    if torch is None:
        raise RuntimeError("backend='torch' requires the optional torch dependency")
    device = _select_torch_device(config.device)
    dtype = _resolve_torch_dtype(config.dtype, device)
    return TorchPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius, device=device, dtype=dtype)


def _torch_accelerator_available() -> bool:
    if torch is None:
        return False
    return bool(torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))


def _select_torch_device(device: str | None) -> Any:
    if torch is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("torch is not available")
    if device:
        selected = torch.device(device)
    elif torch.cuda.is_available():
        selected = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        selected = torch.device("mps")
    else:
        selected = torch.device("cpu")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if selected.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return selected


def _resolve_torch_dtype(dtype: DTypeChoice, device: Any) -> Any:
    if torch is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("torch is not available")
    if dtype == "auto":
        return torch.float32 if device.type in ("cuda", "mps") else torch.float64
    if dtype == "float32":
        return torch.float32
    if device.type == "mps":
        raise ValueError("MPS does not support torch.float64; use dtype='auto' or dtype='float32'")
    return torch.float64


def _clamp_dt(dt: float, config: Any) -> float:
    return float(np.clip(float(dt), float(config.dt_min), float(config.dt_max)))


def _cfl_limited_dt(dt: float, max_speed: float, h: float, config: Any) -> float:
    limited = _clamp_dt(dt, config)
    if config.max_displacement_per_step > 0 and max_speed > 1e-14:
        cfl_dt = float(config.max_displacement_per_step) * float(h) / float(max_speed)
        limited = min(limited, max(float(config.dt_min), cfl_dt))
    return _clamp_dt(limited, config)


def _adaptive_step_factor(local_err: float, tol: float, *, grow: bool, order: int = 2) -> float:
    if not np.isfinite(local_err) or local_err <= 0:
        return 2.0 if grow else 0.25
    factor = 0.92 * float(tol / local_err) ** (1.0 / max(1, int(order)))
    if grow:
        return float(np.clip(factor, 0.5, 2.0))
    return float(np.clip(factor, 0.2, 0.8))


def _dt_history_summary(dt_history: Sequence[float]) -> tuple[float, float, float]:
    if not dt_history:
        return 0.0, 0.0, 0.0
    arr = np.asarray(dt_history, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(arr.mean())
