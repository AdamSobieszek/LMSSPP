"""Long max_steps=2000 PP reference figures for disk-vs-cross morphology."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lmsspp.dynamics.pp_cs_equilibria import (
    fiber_colors,
    make_initial_condition,
)
from lmsspp.dynamics.pp_transient_research import (
    RESEARCH_DIAGNOSTIC_FIELDS,
    TransientResearchConfig,
    run_research_simulation,
    save_research_run_outputs,
)


def _base_config(out_dir: Path) -> TransientResearchConfig:
    return TransientResearchConfig(
        n_fibers=10,
        n_per_fiber=100,
        alpha=0.99,
        K=1.0,
        grid_size=256,
        domain_radius=4.0,
        dt=0.055,
        dt_min=2.5e-4,
        dt_max=0.09,
        max_steps=2000,
        min_steps=2010,
        tol_rms=0.0,
        max_displacement_per_step=0.75,
        backend="numpy",
        force_backend="fft",
        color_scheme="phase_color",
        seed=2026,
        make_dashboard=False,
        make_animation=True,
        trajectory_frame_count=25,
        max_animation_points_per_group=1200,
        record_research_diagnostics=True,
        record_every=100,
        research_diagnostics_every=100,
        research_diagnostic_sample_size=900,
        research_energy_sample_size=900,
        research_nn_chunk=384,
        max_plot_points_per_group=1200,
        out_dir=out_dir,
    )


def _colors(config: TransientResearchConfig, result) -> np.ndarray:
    colors = np.array(fiber_colors(config, result.initial.omega_atoms, len(result.initial.group_names)))
    return colors[result.initial.group_id]


def _cloud_halfwidth(x: np.ndarray, q: float = 0.995, pad: float = 1.25, floor: float = 1e-8) -> float:
    rel = x - x.mean(axis=0, keepdims=True)
    r = np.linalg.norm(rel, axis=1)
    return max(float(floor), float(pad * np.quantile(r, q)))


def _plot_final(results: dict[str, tuple[TransientResearchConfig, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for col, name in enumerate(("fixed_rk2", "adaptive_rk2")):
        cfg, result = results[name]
        x = result.x_final
        pc = _colors(cfg, result)
        axes[0, col].scatter(x[:, 0], x[:, 1], s=4, c=pc, alpha=0.72, linewidths=0)
        axes[0, col].set_xlim(-cfg.domain_radius, cfg.domain_radius)
        axes[0, col].set_ylim(-cfg.domain_radius, cfg.domain_radius)
        axes[0, col].set_title(f"{name} full domain")
        zoom = cfg.domain_radius if name == "fixed_rk2" else _cloud_halfwidth(x, q=0.9975, pad=1.35, floor=1e-6)
        axes[1, col].scatter(x[:, 0], x[:, 1], s=4, c=pc, alpha=0.72, linewidths=0)
        axes[1, col].set_xlim(-zoom, zoom)
        axes[1, col].set_ylim(-zoom, zoom)
        axes[1, col].set_title(f"{name} zoom half-width={zoom:.3g}")
    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
    fig.suptitle("Long run, max_steps=2000: disk scale vs cross scale")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_long_2000_final_full_and_zoom.png", dpi=200)
    plt.close(fig)


def _plot_montage(config: TransientResearchConfig, result, out_dir: Path, *, name: str, dynamic_zoom: bool) -> None:
    if result.trajectory_x is None or result.trajectory_steps is None or result.trajectory_times is None:
        return
    traj = np.asarray(result.trajectory_x, dtype=float)
    steps = np.asarray(result.trajectory_steps)
    times = np.asarray(result.trajectory_times)
    pc = _colors(config, result)
    cols = 5
    rows = int(np.ceil(len(traj) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, x in enumerate(traj):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        ax.scatter(x[:, 0], x[:, 1], s=3, c=pc, alpha=0.70, linewidths=0)
        lim = _cloud_halfwidth(x, q=0.9975, pad=1.35, floor=1e-6) if dynamic_zoom else config.domain_radius
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.12)
        ax.set_title(f"step {int(steps[i])}, t={float(times[i]):.3g}", fontsize=8)
    suffix = "dynamic zoom" if dynamic_zoom else "full-domain"
    fig.suptitle(f"Long run montage: {name} ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_long_2000_montage_{name}.png", dpi=200)
    plt.close(fig)


def _plot_cross_details(config: TransientResearchConfig, result, out_dir: Path) -> None:
    x = result.x_final
    rel = x - x.mean(axis=0, keepdims=True)
    pc = _colors(config, result)
    xy = rel[:, 0] * rel[:, 1]
    log_abs_xy = np.log10(np.abs(xy) + 1e-40)
    radius = np.linalg.norm(rel, axis=1)
    zoom = _cloud_halfwidth(x, q=0.9975, pad=1.45, floor=1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(rel[:, 0], rel[:, 1], s=5, c=pc, alpha=0.75, linewidths=0)
    axes[0].set_title("adaptive final: fiber colors")
    im = axes[1].scatter(rel[:, 0], rel[:, 1], s=5, c=log_abs_xy, cmap="viridis", alpha=0.85, linewidths=0)
    axes[1].set_title(r"adaptive final colored by $\log_{10}|xy|$")
    fig.colorbar(im, ax=axes[1], shrink=0.82)
    axes[2].hist(radius, bins=80, color="#4c78a8", alpha=0.85)
    axes[2].set_title("adaptive final radial histogram")
    axes[2].set_xlabel("radius")
    for ax in axes[:2]:
        ax.axhline(0.0, color="0.25", linewidth=0.7, alpha=0.5)
        ax.axvline(0.0, color="0.25", linewidth=0.7, alpha=0.5)
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
    fig.suptitle(f"Adaptive cross-scale diagnostics, zoom half-width={zoom:.3g}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_long_2000_adaptive_cross_details.png", dpi=200)
    plt.close(fig)


def _write_report(results: dict[str, tuple[TransientResearchConfig, object]], out_dir: Path) -> None:
    lines = [
        "# Long max_steps=2000 PP Cross/Disk Reference",
        "",
        "Configuration: `n_fibers=10`, `n_per_fiber=100`, `alpha=0.99`, `K=1`, `grid_size=256`, `dt=0.055`, same seed/initial condition.",
        "",
        "## Figures",
        "",
        "- [Final full-domain and zoomed comparison](fig_long_2000_final_full_and_zoom.png)",
        "- [Fixed-RK2 full-domain montage](fig_long_2000_montage_fixed_rk2.png)",
        "- [Adaptive-RK2 cross montage with dynamic zoom](fig_long_2000_montage_adaptive_rk2.png)",
        "- [Adaptive cross-scale details](fig_long_2000_adaptive_cross_details.png)",
        "",
        "## Metrics",
        "",
    ]
    for name, (_, result) in results.items():
        diag = result.research_diagnostics
        if diag is None or len(diag) == 0:
            continue
        last = dict(zip(RESEARCH_DIAGNOSTIC_FIELDS, diag[-1], strict=True))
        max_q = float(np.max(diag[:, RESEARCH_DIAGNOSTIC_FIELDS.index("q_max")]))
        lines.append(
            f"- `{name}`: steps={result.steps}, final_time={result.final_time:.6g}, "
            f"disk_radius_metric={last['disk_radius']:.6g}, r_min={last['r_min']:.6g}, "
            f"q_max_max={max_q:.6g}, R_cont_rms={last['R_cont_rms']:.6g}, "
            f"R_disc_rms={last['R_disc_rms']:.6g}, clip_events={result.clip_events}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "At `max_steps=2000`, fixed RK2 remains on the large disk/shell scale, while adaptive RK2 evolves to a much smaller filamentary cross scale. The final adaptive figure must be viewed with dynamic zoom; on the disk-scale axes it is visually indistinguishable from a point.",
            "",
            "This long run reinforces the earlier diagnosis: the disk is a fixed-step map artifact, while the adaptive trajectory resolves the near-singular precision layer and continues toward cross-like collapse.",
            "",
        ]
    )
    (out_dir / "REPORT_LONG_2000.md").write_text("\n".join(lines))


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _base_config(out_dir)
    initial = make_initial_condition(base)
    results: dict[str, tuple[TransientResearchConfig, object]] = {}
    for integrator in ("fixed_rk2", "adaptive_rk2"):
        cfg = replace(base, integrator=integrator, out_dir=out_dir / integrator)
        print(f"[run] {integrator}: max_steps={cfg.max_steps}, N={len(initial.x)}")
        result = run_research_simulation(cfg, initial)
        save_research_run_outputs(result, cfg, cfg.out_dir, initial=initial)
        results[integrator] = (cfg, result)
    _plot_final(results, out_dir)
    _plot_montage(*results["fixed_rk2"], out_dir, name="fixed_rk2", dynamic_zoom=False)
    _plot_montage(*results["adaptive_rk2"], out_dir, name="adaptive_rk2", dynamic_zoom=True)
    _plot_cross_details(*results["adaptive_rk2"], out_dir)
    _write_report(results, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("pp_transient_research_results_2000"))
    args = parser.parse_args()
    run(args.out_dir)
    print(f"Wrote long reference to {args.out_dir}")


if __name__ == "__main__":
    main()
