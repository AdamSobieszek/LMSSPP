"""Run bounded PP transient-artifact experiments and write a research report.

The full parameter grid proposed in the brief is intentionally large.  This
script runs a diagnostic subset that exercises the main falsifiable predictions:

* fixed timestep scaling,
* alpha and K-prefactor scaling,
* grid refinement,
* fixed/adaptive and FFT/direct comparisons,
* conical/saddle toy mechanisms.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lmsspp.dynamics.pp_cs_equilibria import (
    InitialCondition,
    fiber_colors,
    make_initial_condition,
)
from lmsspp.dynamics.pp_transient_research import (
    RESEARCH_DIAGNOSTIC_FIELDS,
    TransientResearchConfig,
    evaluate_A_at_particles,
    rk2_map_residual,
    run_research_simulation,
    run_toy_transient_suite,
    save_research_run_outputs,
)


def _base_config(out_dir: Path, *, n_per_fiber: int, grid_size: int, max_steps: int) -> TransientResearchConfig:
    return TransientResearchConfig(
        n_fibers=10,
        n_per_fiber=n_per_fiber,
        alpha=0.99,
        K=1.0,
        grid_size=grid_size,
        domain_radius=4.0,
        dt=0.055,
        dt_min=2.5e-4,
        dt_max=0.09,
        max_steps=max_steps,
        min_steps=max_steps + 10,
        tol_rms=0.0,
        max_displacement_per_step=0.75,
        backend="numpy",
        force_backend="fft",
        integrator="fixed_rk2",
        color_scheme="phase_color",
        seed=2026,
        make_dashboard=False,
        make_animation=False,
        record_research_diagnostics=True,
        record_every=20,
        research_diagnostics_every=20,
        research_diagnostic_sample_size=420,
        research_energy_sample_size=420,
        research_nn_chunk=256,
        max_plot_points_per_group=600,
        out_dir=out_dir,
    )


def _run_case(label: str, cfg: TransientResearchConfig, initial: InitialCondition, root: Path) -> dict[str, Any]:
    case_dir = root / label
    cfg = replace(cfg, out_dir=case_dir, record_research_diagnostics=True, make_dashboard=False, make_animation=False)
    print(f"[run] {label}: integrator={cfg.integrator}, backend={cfg.force_backend}, alpha={cfg.alpha}, K={cfg.K}, h={cfg.dt}, grid={cfg.grid_size}, N={len(initial.x)}")
    result = run_research_simulation(cfg, initial)
    metrics = save_research_run_outputs(result, cfg, case_dir, initial=initial)
    metrics["label"] = label
    metrics["case_dir"] = str(case_dir)
    metrics["x_final"] = result.x_final
    metrics["omega"] = result.initial.omega
    metrics["group_id"] = result.initial.group_id
    metrics["omega_atoms"] = result.initial.omega_atoms
    metrics["group_names"] = result.initial.group_names
    metrics["config"] = cfg
    return metrics


def _strip_for_json(case: dict[str, Any]) -> dict[str, Any]:
    blocked = {"x_final", "omega", "group_id", "omega_atoms", "group_names", "config"}
    return {k: v for k, v in case.items() if k not in blocked}


def _write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    clean_rows = [_strip_for_json(row) for row in rows]
    keys: list[str] = []
    for row in clean_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(clean_rows)


def _metric(case: dict[str, Any], key: str, default: float = np.nan) -> float:
    return float(case.get(key, default))


def _final_radius(case: dict[str, Any]) -> float:
    return _metric(case, "last_disk_radius", _metric(case, "final_disk_radius"))


def _plot_morphology_grid(cases: list[dict[str, Any]], path: Path, *, cols: int = 3, title: str = "") -> None:
    if not cases:
        return
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, case in enumerate(cases):
        ax = axes[idx // cols][idx % cols]
        ax.axis("on")
        cfg: TransientResearchConfig = case["config"]
        colors = fiber_colors(cfg, case["omega_atoms"], len(case["group_names"]))
        x = np.asarray(case["x_final"])
        group = np.asarray(case["group_id"])
        for k, color in enumerate(colors):
            mask = group == k
            ax.scatter(x[mask, 0], x[mask, 1], s=6, c=color, alpha=0.72, linewidths=0)
        ax.set_title(case["label"], fontsize=9)
        ax.set_xlim(-cfg.domain_radius, cfg.domain_radius)
        ax.set_ylim(-cfg.domain_radius, cfg.domain_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scaling(
    cases: list[dict[str, Any]],
    path: Path,
    *,
    x_key: str,
    x_label: str,
    title: str,
    invert_x: bool = False,
) -> None:
    x = np.array([_metric(case, x_key) for case in cases], dtype=float)
    order = np.argsort(x)
    x = x[order]
    ordered = [cases[i] for i in order]
    radius = np.array([_final_radius(case) for case in ordered])
    shell = np.array([_metric(case, "last_rk2_shell_radius") for case in ordered])
    q = np.array([_metric(case, "max_q_max") for case in ordered])
    rdisc = np.array([_metric(case, "last_R_disc_rms") for case in ordered])
    rcont = np.array([_metric(case, "last_R_cont_rms") for case in ordered])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, radius, "o-", label="observed radial peak")
    axes[0].plot(x, shell, "--", label="hK/[2(1-alpha)]")
    axes[0].set_ylabel("radius")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(x, q, "o-")
    axes[1].axhline(1.0, color="0.4", linestyle=":")
    axes[1].axhline(2.0, color="0.6", linestyle=":")
    axes[1].set_ylabel("max q")
    axes[2].semilogy(x, rcont, "o-", label="continuous RMS")
    axes[2].semilogy(x, rdisc, "o-", label="RK2 map RMS")
    axes[2].set_ylabel("residual")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(x_label)
        if invert_x:
            ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _read_diag(case: dict[str, Any]) -> np.ndarray:
    path = Path(case["case_dir"]) / "research_diagnostics.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def _plot_time_series_pair(cases: list[dict[str, Any]], path: Path, *, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for case in cases:
        diag = _read_diag(case)
        label = case["label"]
        t = diag["time"]
        axes[0, 0].semilogy(t, diag["r_min"], label=label)
        axes[0, 1].semilogy(t, diag["lambda_max_max"], label=label)
        axes[1, 0].semilogy(t, diag["q_max"], label=label)
        axes[1, 1].plot(t, diag["disk_radius"], label=label)
        axes[1, 1].plot(t, diag["axis_mass_fraction"], linestyle="--", label=f"{label} axis")
    axes[0, 0].set_ylabel("r_min")
    axes[0, 1].set_ylabel("Lambda_max")
    axes[1, 0].set_ylabel("q_max")
    axes[1, 1].set_ylabel("disk radius / axis fraction")
    for ax in axes.ravel():
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _conical_scale(case: dict[str, Any]) -> float:
    cfg: TransientResearchConfig = case["config"]
    return float(abs(cfg.dt) * abs(cfg.K) / max(abs(1.0 - float(cfg.alpha)), 1e-30))


def _plot_verdict_conical_collapse(cases_by_group: dict[str, list[dict[str, Any]]], path: Path) -> dict[str, float]:
    cases: list[tuple[str, dict[str, Any]]] = []
    for group in ("timestep", "alpha_fixed", "K"):
        cases.extend((group, case) for case in cases_by_group.get(group, []) if case["config"].integrator == "fixed_rk2")
    if not cases:
        return {}

    x = np.array([_conical_scale(case) for _, case in cases], dtype=float)
    y = np.array([_final_radius(case) for _, case in cases], dtype=float)
    q = np.array([_metric(case, "max_q_max") for _, case in cases], dtype=float)
    groups = [group for group, _ in cases]
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    fit_mask = valid & np.isfinite(q) & (q >= 1.0)
    if not np.any(fit_mask):
        return {}

    x_valid = x[fit_mask]
    y_valid = y[fit_mask]
    C = float(np.sum(x_valid * y_valid) / np.sum(x_valid * x_valid))
    y_hat = C * x_valid
    ss_res = float(np.sum((y_valid - y_hat) ** 2))
    ss_tot = float(np.sum((y_valid - float(np.mean(y_valid))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    log_slope = float(np.polyfit(np.log(x_valid), np.log(y_valid), 1)[0]) if len(x_valid) >= 2 else float("nan")

    colors = {"timestep": "#4c78a8", "alpha_fixed": "#f58518", "K": "#54a24b"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for group in ("timestep", "alpha_fixed", "K"):
        mask = np.array([g == group for g in groups]) & valid
        if np.any(mask):
            ax.scatter(x[mask], y[mask], s=54, label=group, color=colors[group], alpha=0.85)
        inactive = np.array([g == group for g in groups]) & valid & ~fit_mask
        if np.any(inactive):
            ax.scatter(x[inactive], y[inactive], s=76, facecolors="none", edgecolors=colors[group], linewidths=1.5, alpha=0.9, label=f"{group} (q<1 excluded)")
    xx = np.linspace(float(np.min(x_valid)) * 0.9, float(np.max(x_valid)) * 1.08, 200)
    ax.plot(xx, C * xx, color="0.15", linewidth=1.7, label=f"fit R = {C:.3g} hK/(1-alpha)")
    ax.set_xlabel("fixed-step conical scale  hK/(1-alpha)")
    ax.set_ylabel("observed disk/radial peak")
    ax.set_title(f"Verdict: active fixed-RK2 disk scale collapse (q>=1, R^2={r2:.3f}, log slope={log_slope:.2f})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {
        "conical_scale_fit_C": C,
        "conical_scale_fit_r2": r2,
        "conical_scale_log_slope": log_slope,
        "conical_scale_fit_q_threshold": 1.0,
        "conical_scale_fit_count": int(np.count_nonzero(fit_mask)),
        "conical_scale_excluded_count": int(np.count_nonzero(valid & ~fit_mask)),
    }


def _plot_spatial_residual_pair(case: dict[str, Any], path: Path) -> dict[str, float]:
    cfg: TransientResearchConfig = case["config"]
    x = np.asarray(case["x_final"], dtype=np.float64)
    omega = np.asarray(case["omega"], dtype=np.float64)
    group = np.asarray(case["group_id"])
    colors = np.array(fiber_colors(cfg, case["omega_atoms"], len(case["group_names"])))
    edge_colors = colors[group]

    A = evaluate_A_at_particles(cfg, x, force_backend=cfg.force_backend)
    continuous = np.linalg.norm(omega - A, axis=1)
    discrete = np.linalg.norm(rk2_map_residual(cfg, x, omega, cfg.dt, force_backend=cfg.force_backend), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), squeeze=False)
    values = (continuous, discrete)
    titles = (r"continuous residual $|\omega-A_\rho(x)|$", r"RK2 map residual $|\Phi_h(x)-x|$")
    for ax, val, title in zip(axes.ravel(), values, titles, strict=True):
        log_val = np.log10(val + 1e-30)
        sc = ax.scatter(x[:, 0], x[:, 1], s=10, c=log_val, cmap="magma", alpha=0.86, linewidths=0)
        ax.scatter(x[:, 0], x[:, 1], s=2, c=edge_colors, alpha=0.25, linewidths=0)
        ax.set_title(title)
        ax.set_xlim(-cfg.domain_radius, cfg.domain_radius)
        ax.set_ylim(-cfg.domain_radius, cfg.domain_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.16)
        cb = fig.colorbar(sc, ax=ax, shrink=0.82)
        cb.set_label("log10 residual")
    fig.suptitle(f"Fixed-RK2 disk: ODE residual vs one-step map residual ({case['label']})")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {
        "spatial_R_cont_rms": float(np.sqrt(np.mean(continuous * continuous))),
        "spatial_R_disc_rms": float(np.sqrt(np.mean(discrete * discrete))),
        "spatial_R_cont_max": float(np.max(continuous)),
        "spatial_R_disc_max": float(np.max(discrete)),
    }


def _interpret(cases_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    notes: dict[str, str] = {}
    timestep = cases_by_group.get("timestep", [])
    if len(timestep) >= 3:
        h = np.array([_metric(c, "dt") for c in timestep])
        r = np.array([_final_radius(c) for c in timestep])
        valid = np.isfinite(h) & np.isfinite(r) & (h > 0) & (r > 0)
        if np.count_nonzero(valid) >= 3:
            slope = float(np.polyfit(np.log(h[valid]), np.log(r[valid]), 1)[0])
            corr = float(np.corrcoef(h[valid], r[valid])[0, 1])
            notes["timestep"] = f"Fixed-RK2 radial peak scales with h with log-log slope {slope:.2f} and linear correlation {corr:.2f}."
    alpha = cases_by_group.get("alpha_fixed", [])
    if len(alpha) >= 3:
        inv_delta = np.array([_metric(c, "inv_delta") for c in alpha])
        r = np.array([_final_radius(c) for c in alpha])
        valid = np.isfinite(inv_delta) & np.isfinite(r) & (r > 0)
        if np.count_nonzero(valid) >= 3:
            slope = float(np.polyfit(np.log(inv_delta[valid]), np.log(r[valid]), 1)[0])
            notes["alpha"] = f"Fixed-RK2 radial peak scales against 1/(1-alpha) with log-log slope {slope:.2f}."
    direct = cases_by_group.get("direct_fft", [])
    if direct:
        fixed = [c for c in direct if c["config"].integrator == "fixed_rk2"]
        adaptive = [c for c in direct if c["config"].integrator == "adaptive_rk2"]
        notes["direct_fft"] = (
            "Direct small-N comparison final radii: "
            + ", ".join(f"{c['label']}={_final_radius(c):.3g}" for c in fixed + adaptive)
        )
    fixed = cases_by_group.get("regime", [None])[0]
    if fixed is not None:
        r_cont = _metric(fixed, "spatial_R_cont_rms", _metric(fixed, "last_R_cont_rms"))
        r_disc = _metric(fixed, "spatial_R_disc_rms", _metric(fixed, "last_R_disc_rms"))
        if np.isfinite(r_cont) and np.isfinite(r_disc) and r_disc > 0:
            notes["map_residual"] = f"Fixed disk residual split: continuous RMS / RK2-map RMS = {r_cont / r_disc:.2e}."
    return notes


def run_study(out_dir: Path, *, quick: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_per = 25 if quick else 50
    steps = 80 if quick else 180
    grid = 128 if quick else 256
    base = _base_config(out_dir, n_per_fiber=n_per, grid_size=grid, max_steps=steps)
    initial = make_initial_condition(base)
    all_cases: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}

    regime_cases = [
        _run_case("regime_fixed_rk2_fft", replace(base, integrator="fixed_rk2", dt=0.055), initial, out_dir),
        _run_case("regime_adaptive_rk2_fft", replace(base, integrator="adaptive_rk2", dt=0.055), initial, out_dir),
    ]
    groups["regime"] = regime_cases
    all_cases.extend(regime_cases)
    _plot_morphology_grid(regime_cases, out_dir / "fig_A_regime_pair_morphology.png", cols=2, title="Same initial condition, fixed vs adaptive RK2")
    _plot_time_series_pair(regime_cases, out_dir / "fig_D_regime_time_series.png", title="Same initial condition transient diagnostics")
    regime_cases[0].update(_plot_spatial_residual_pair(regime_cases[0], out_dir / "fig_G_fixed_disk_residual_spatial.png"))

    h_values = (0.09, 0.055, 0.03, 0.015, 0.0075) if quick else (0.09, 0.055, 0.03, 0.015, 0.0075, 0.00375)
    timestep_cases = [
        _run_case(f"timestep_h_{str(h).replace('.', 'p')}", replace(base, integrator="fixed_rk2", dt=h, dt_max=max(h, base.dt_max)), initial, out_dir)
        for h in h_values
    ]
    groups["timestep"] = timestep_cases
    all_cases.extend(timestep_cases)
    _plot_morphology_grid(timestep_cases, out_dir / "fig_C_timestep_morphology.png", cols=3, title="Fixed-RK2 timestep sweep")
    _plot_scaling(timestep_cases, out_dir / "fig_C_timestep_scaling.png", x_key="dt", x_label="fixed timestep h", title="Timestep-induced shell test")

    alpha_values = (0.95, 0.98, 0.99, 0.995) if quick else (0.95, 0.97, 0.98, 0.99, 0.995)
    alpha_fixed = [
        _run_case(f"alpha_fixed_{str(a).replace('.', 'p')}", replace(base, alpha=a, integrator="fixed_rk2"), initial, out_dir)
        for a in alpha_values
    ]
    alpha_adaptive = [
        _run_case(f"alpha_adaptive_{str(a).replace('.', 'p')}", replace(base, alpha=a, integrator="adaptive_rk2"), initial, out_dir)
        for a in alpha_values
    ]
    groups["alpha_fixed"] = alpha_fixed
    groups["alpha_adaptive"] = alpha_adaptive
    all_cases.extend(alpha_fixed + alpha_adaptive)
    _plot_morphology_grid(alpha_fixed + alpha_adaptive, out_dir / "fig_alpha_integrator_morphology.png", cols=len(alpha_values), title="Alpha sweep: fixed row then adaptive row")
    _plot_scaling(alpha_fixed, out_dir / "fig_alpha_fixed_scaling.png", x_key="inv_delta", x_label="1/(1-alpha)", title="Fixed-RK2 alpha scaling")

    delta = 1.0 - base.alpha
    k_cases = [
        _run_case("K_1", replace(base, K=1.0, integrator="fixed_rk2"), initial, out_dir),
        _run_case("K_delta", replace(base, K=delta, integrator="fixed_rk2"), initial, out_dir),
        _run_case("K_sqrt_delta", replace(base, K=float(np.sqrt(delta)), integrator="fixed_rk2"), initial, out_dir),
    ]
    groups["K"] = k_cases
    all_cases.extend(k_cases)
    _plot_morphology_grid(k_cases, out_dir / "fig_K_renormalization_morphology.png", cols=3, title="K renormalization at alpha=0.99")

    grid_values = (128, 256) if quick else (128, 256, 512)
    grid_cases: list[dict[str, Any]] = []
    for g in grid_values:
        grid_cases.append(_run_case(f"grid_fixed_{g}", replace(base, grid_size=g, integrator="fixed_rk2"), initial, out_dir))
        grid_cases.append(_run_case(f"grid_adaptive_{g}", replace(base, grid_size=g, integrator="adaptive_rk2"), initial, out_dir))
    groups["grid"] = grid_cases
    all_cases.extend(grid_cases)
    _plot_morphology_grid(grid_cases, out_dir / "fig_E_grid_morphology.png", cols=2, title="Grid refinement: fixed/adaptive by resolution")
    _plot_scaling([c for c in grid_cases if c["config"].integrator == "fixed_rk2"], out_dir / "fig_E_grid_fixed_scaling.png", x_key="dx", x_label="grid spacing dx", title="Fixed-RK2 grid dependence", invert_x=True)

    small = replace(base, n_per_fiber=20, grid_size=128, max_steps=80 if quick else 120, record_every=10, research_diagnostics_every=10)
    small_initial = make_initial_condition(small)
    direct_fft_cases = [
        _run_case("small_fft_fixed", replace(small, force_backend="fft", integrator="fixed_rk2"), small_initial, out_dir),
        _run_case("small_fft_adaptive", replace(small, force_backend="fft", integrator="adaptive_rk2"), small_initial, out_dir),
        _run_case("small_direct_fixed", replace(small, force_backend="direct", backend="numpy", integrator="fixed_rk2"), small_initial, out_dir),
        _run_case("small_direct_adaptive", replace(small, force_backend="direct", backend="numpy", integrator="adaptive_rk2"), small_initial, out_dir),
    ]
    groups["direct_fft"] = direct_fft_cases
    all_cases.extend(direct_fft_cases)
    _plot_morphology_grid(direct_fft_cases, out_dir / "fig_direct_fft_morphology.png", cols=2, title="Small-N direct vs FFT/CIC")

    toy_dir = out_dir / "toy_models"
    toy_metrics = run_toy_transient_suite(
        toy_dir,
        h_values=(0.09, 0.055, 0.03, 0.015, 0.0075),
        n_points=240 if quick else 700,
        max_steps=70 if quick else 180,
        seed=2026,
        c=1.0,
        eps=1e-4,
        delta=0.01,
    )

    verdict_metrics = _plot_verdict_conical_collapse(groups, out_dir / "fig_verdict_conical_scale_collapse.png")
    if verdict_metrics:
        for case in all_cases:
            case.update({f"verdict_{key}": value for key, value in verdict_metrics.items()})
    _write_summary_csv(out_dir / "all_case_metrics.csv", all_cases)
    (out_dir / "all_case_metrics.json").write_text(json.dumps([_strip_for_json(c) for c in all_cases], indent=2))
    notes = _interpret(groups)
    if verdict_metrics:
        notes["verdict"] = (
            "Combined timestep, alpha, and K-renormalization fixed-RK2 runs collapse against "
            f"hK/(1-alpha) for shell-active cases q >= {verdict_metrics['conical_scale_fit_q_threshold']:.1f}; "
            f"through-origin coefficient {verdict_metrics['conical_scale_fit_C']:.3g}, "
            f"R^2={verdict_metrics['conical_scale_fit_r2']:.3f}, log-log slope {verdict_metrics['conical_scale_log_slope']:.2f}, "
            f"n={int(verdict_metrics['conical_scale_fit_count'])}, excluded subthreshold n={int(verdict_metrics['conical_scale_excluded_count'])}."
        )
    _write_report(out_dir, groups, notes, toy_metrics)


def _write_report(out_dir: Path, groups: dict[str, list[dict[str, Any]]], notes: dict[str, str], toy_metrics: dict[str, Any]) -> None:
    regime = groups["regime"]
    fixed = regime[0]
    adaptive = regime[1]
    lines = [
        "# PP Hyperbolic-Cross vs Disk Transient Study",
        "",
        "This report is generated by `scripts/run_pp_transient_research.py`.",
        "",
        "## Main Saved Figures",
        "",
        "- [Figure A: same initial condition, fixed vs adaptive morphology](fig_A_regime_pair_morphology.png)",
        "- [Figure D: same initial condition time-series diagnostics](fig_D_regime_time_series.png)",
        "- [Figure G: fixed disk continuous-vs-RK2-map spatial residuals](fig_G_fixed_disk_residual_spatial.png)",
        "- [Verdict: conical scale collapse](fig_verdict_conical_scale_collapse.png)",
        "- [Figure C: fixed timestep morphology](fig_C_timestep_morphology.png)",
        "- [Figure C: fixed timestep scaling](fig_C_timestep_scaling.png)",
        "- [Alpha sweep morphology](fig_alpha_integrator_morphology.png)",
        "- [Alpha fixed-RK2 scaling](fig_alpha_fixed_scaling.png)",
        "- [K-renormalization morphology](fig_K_renormalization_morphology.png)",
        "- [Grid refinement morphology](fig_E_grid_morphology.png)",
        "- [Grid fixed-RK2 scaling](fig_E_grid_fixed_scaling.png)",
        "- [Direct-vs-FFT small-N morphology](fig_direct_fft_morphology.png)",
        "- [Toy conical/saddle transients](toy_models/toy_transients.html)",
        "- [Toy shell scaling](toy_models/toy_shell_scaling.html)",
        "",
        "## Same Initial Condition Regime Pair",
        "",
        f"- Fixed RK2 final radial peak: `{_final_radius(fixed):.6g}`; max q: `{_metric(fixed, 'max_q_max'):.6g}`; continuous RMS residual: `{_metric(fixed, 'last_R_cont_rms'):.6g}`; RK2-map RMS residual: `{_metric(fixed, 'last_R_disc_rms'):.6g}`.",
        f"- Adaptive RK2 final radial peak: `{_final_radius(adaptive):.6g}`; max q: `{_metric(adaptive, 'max_q_max'):.6g}`; continuous RMS residual: `{_metric(adaptive, 'last_R_cont_rms'):.6g}`; RK2-map RMS residual: `{_metric(adaptive, 'last_R_disc_rms'):.6g}`.",
        "",
        "## Quantitative Notes",
        "",
    ]
    for key, note in notes.items():
        lines.append(f"- `{key}`: {note}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The refined interpretation is that the disk scale is not primarily a `log r` effect. It is controlled by the dominant conical prefactor in `grad W^{1-delta} = (1/delta + log r + ...) zhat` together with unresolved Hessian stiffness. The decisive law is therefore `R_disk ~ C hK/(1-alpha)`. The `log r` correction is more plausibly visible after leading conical contributions cancel locally, where the resolved adaptive collapse can organize into saddle-like hyperbolic connectors with approximate `xy = C` sheets.",
            "",
            "The one-step residual figure is the visual fixed-map test: if the disk is a numerical equilibrium, the disk can carry a large continuous residual `|omega - A_rho(x)|` while the same points have a small RK2 map residual `|Phi_h(x)-x|`. That split is stronger evidence than a scalar RMS table because it shows the residual structure on the disk itself.",
            "",
            "The resulting hierarchy is: fixed RK2 creates the disk through conical overshoot at scale `hK/(1-alpha)`; disk persistence is a stable fixed point of the RK2 map rather than the ODE; FFT/CIC smoothing is not primary when direct fixed RK2 reproduces the disk and grid refinement does not move it; adaptive RK2 resolves the Hessian/focusing geometry of the continuous-time PP flow; logarithmic/hyperbolic sheet effects are secondary organizing features of the resolved collapse after local conical cancellation.",
            "",
            "## Toy Metrics",
            "",
            "```json",
            json.dumps(toy_metrics, indent=2),
            "```",
            "",
            "## Per-Case Metrics",
            "",
            "See [`all_case_metrics.csv`](all_case_metrics.csv) and [`all_case_metrics.json`](all_case_metrics.json). Each case directory also contains `research_diagnostics.csv`, `metrics.json`, `final_morphology.html`, and `pp_transient_research_diagnostics.html`.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("pp_transient_research_results"))
    parser.add_argument("--quick", action="store_true", help="run a smaller smoke-sized research grid")
    args = parser.parse_args()
    run_study(args.out_dir, quick=args.quick)
    print(f"Wrote study to {args.out_dir}")


if __name__ == "__main__":
    main()
