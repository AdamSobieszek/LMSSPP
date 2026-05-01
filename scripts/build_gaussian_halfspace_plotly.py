#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained Plotly visualization of the normalized radial "
            "shell density of a high-dimensional Gaussian over a visible half-ball."
        )
    )
    parser.add_argument("--dimension", type=int, default=512, help="Gaussian dimension.")
    parser.add_argument(
        "--rmax",
        type=float,
        default=1.55,
        help="Maximum normalized radius s = r / sqrt(d) shown in the half-ball.",
    )
    parser.add_argument(
        "--shell-count",
        type=int,
        default=18,
        help="Number of chi-square quantile shell bands around the peak shell.",
    )
    parser.add_argument(
        "--polar-resolution",
        type=int,
        default=48,
        help="Samples in polar angle within the rendered first octant.",
    )
    parser.add_argument(
        "--azimuth-resolution",
        type=int,
        default=96,
        help="Samples in azimuth within the rendered first octant.",
    )
    parser.add_argument(
        "--disk-resolution",
        type=int,
        default=150,
        help="Resolution of the planar cut faces used to reveal shell layers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exports" / "plotly_local",
        help="Directory for the generated standalone HTML files.",
    )
    parser.add_argument(
        "--target-total-opacity",
        type=float,
        default=0.88,
        help=(
            "Approximate cumulative opacity after traversing all equal-mass shells. "
            "Used to derive a constant per-shell optical depth."
        ),
    )
    parser.add_argument(
        "--boundary-epsilon",
        type=float,
        default=0.015,
        help=(
            "Inset the rendered shell patches from the coordinate planes to avoid "
            "boundary halo artifacts."
        ),
    )
    parser.add_argument(
        "--display-mass-window",
        type=float,
        default=0.98,
        help=(
            "Central radial mass window shown around the peak shell. "
            "The shell bands are carved from chi-square quantiles inside this window."
        ),
    )
    return parser.parse_args()


def peak_normalized_radius(dimension: int) -> float:
    return math.sqrt((dimension - 1.0) / dimension)


def normalized_shell_density(radius: np.ndarray | float, dimension: int) -> np.ndarray:
    radius = np.asarray(radius, dtype=float)
    values = np.zeros_like(radius, dtype=float)
    positive = radius > 0.0
    if not np.any(positive):
        return values
    rp = radius[positive]
    peak = peak_normalized_radius(dimension)
    log_peak = (dimension - 1.0) * math.log(peak) - 0.5 * dimension * peak * peak
    log_density = (dimension - 1.0) * np.log(rp) - 0.5 * dimension * rp * rp
    values[positive] = np.exp(log_density - log_peak)
    return values


def axis_style(title: str, radius_min: float, radius_max: float) -> dict[str, object]:
    return dict(
        title=title,
        range=[radius_min, radius_max],
        showbackground=True,
        backgroundcolor="rgb(246, 248, 251)",
        gridcolor="rgb(212, 217, 225)",
        zerolinecolor="rgb(142, 150, 160)",
    )


def radial_mass_cdf(
    dimension: int,
    radius_max: float,
    *,
    sample_count: int = 20001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius_grid = np.linspace(0.0, radius_max, sample_count)
    density_grid = normalized_shell_density(radius_grid, dimension)
    cdf_grid = np.zeros_like(radius_grid)
    dr = np.diff(radius_grid)
    increments = 0.5 * (density_grid[:-1] + density_grid[1:]) * dr
    cdf_grid[1:] = np.cumsum(increments)
    total_mass = float(cdf_grid[-1])
    if total_mass <= 0.0:
        raise ValueError("Degenerate radial density integral.")
    cdf_grid /= total_mass
    return radius_grid, density_grid, cdf_grid


def equal_mass_shell_radii(
    dimension: int,
    radius_max: float,
    shell_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    radius_grid, _, cdf_grid = radial_mass_cdf(dimension, radius_max)
    quantiles = (np.arange(shell_count, dtype=float) + 0.5) / float(shell_count)
    shell_radii = np.interp(quantiles, cdf_grid, radius_grid)
    return shell_radii, quantiles


def quantile_shell_bands(
    dimension: int,
    radius_max: float,
    band_count: int,
    display_mass_window: float,
) -> list[dict[str, float]]:
    radius_grid, _, cdf_grid = radial_mass_cdf(dimension, radius_max)
    peak_radius = peak_normalized_radius(dimension)
    peak_quantile = float(np.interp(peak_radius, radius_grid, cdf_grid))
    max_symmetric_window = max(1e-4, 2.0 * min(peak_quantile, 1.0 - peak_quantile))
    display_mass_window = float(np.clip(display_mass_window, 1e-4, max_symmetric_window))
    half_window = 0.5 * display_mass_window
    q_low = peak_quantile - half_window
    q_high = peak_quantile + half_window
    q_edges = np.linspace(q_low, q_high, band_count + 1)
    r_edges = np.interp(q_edges, cdf_grid, radius_grid)

    bands: list[dict[str, float]] = []
    for index in range(band_count):
        q_inner = float(q_edges[index])
        q_outer = float(q_edges[index + 1])
        r_inner = float(r_edges[index])
        r_outer = float(r_edges[index + 1])
        q_mid = 0.5 * (q_inner + q_outer)
        r_mid = float(np.interp(q_mid, cdf_grid, radius_grid))
        density_mid = float(normalized_shell_density(r_mid, dimension))
        bands.append(
            dict(
                q_inner=q_inner,
                q_outer=q_outer,
                q_mid=q_mid,
                r_inner=r_inner,
                r_outer=r_outer,
                r_mid=r_mid,
                density_mid=density_mid,
            )
        )
    return bands


def build_halfspace_figure(
    *,
    dimension: int,
    radius_max: float,
    shell_count: int,
    polar_resolution: int,
    azimuth_resolution: int,
    disk_resolution: int,
    target_total_opacity: float,
    boundary_epsilon: float,
    display_mass_window: float,
) -> go.Figure:
    camera_eye = (-1.45, -1.35, 1.05)
    cut_face_resolution = max(40, int(disk_resolution))

    bands = quantile_shell_bands(dimension, radius_max, shell_count, display_mass_window)
    band_order = np.argsort([band["density_mid"] for band in bands])
    capped_total_opacity = float(np.clip(target_total_opacity, 0.05, 0.99))
    band_opacity = 1.0 - (1.0 - capped_total_opacity) ** (1.0 / float(shell_count))
    surface_opacity = 1.0 - math.sqrt(max(1e-9, 1.0 - band_opacity))
    cartesian_epsilon = float(np.clip(boundary_epsilon, 0.0, 0.20 * radius_max))
    outermost_radius = float(bands[-1]["r_outer"])
    simplex_threshold = outermost_radius + 2.0 * cartesian_epsilon
    halfspaces = [
        (np.array([1.0, 0.0, 0.0]), cartesian_epsilon),
        (np.array([0.0, 1.0, 0.0]), cartesian_epsilon),
        (np.array([0.0, 0.0, 1.0]), cartesian_epsilon),
        (np.array([1.0, 1.0, 1.0]), simplex_threshold),
    ]

    def clip_polygon_halfspace(
        polygon: list[np.ndarray],
        normal: np.ndarray,
        threshold: float,
    ) -> list[np.ndarray]:
        if not polygon:
            return []
        clipped: list[np.ndarray] = []
        previous = polygon[-1]
        previous_value = float(np.dot(normal, previous) - threshold)
        previous_inside = previous_value >= -1e-9
        for current in polygon:
            current_value = float(np.dot(normal, current) - threshold)
            current_inside = current_value >= -1e-9
            if previous_inside != current_inside:
                denom = previous_value - current_value
                t = 0.0 if abs(denom) < 1e-12 else previous_value / denom
                clipped.append(previous + t * (current - previous))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        return clipped

    def append_clipped_triangle(
        vertices: list[np.ndarray],
        intensities: list[float],
        faces: list[tuple[int, int, int]],
        triangle: tuple[np.ndarray, np.ndarray, np.ndarray],
        density: float,
    ) -> None:
        polygon = [triangle[0], triangle[1], triangle[2]]
        for normal, threshold in halfspaces:
            polygon = clip_polygon_halfspace(polygon, normal, threshold)
            if len(polygon) < 3:
                return
        base = len(vertices)
        vertices.extend(polygon)
        intensities.extend([density] * len(polygon))
        for idx in range(1, len(polygon) - 1):
            faces.append((base, base + idx, base + idx + 1))

    def append_grid_triangles(
        vertices: list[np.ndarray],
        intensities: list[float],
        faces: list[tuple[int, int, int]],
        grid_points: np.ndarray,
        density: float,
        *,
        wrap_second_axis: bool = False,
        reverse: bool = False,
    ) -> None:
        n0, n1, _ = grid_points.shape
        second_limit = n1 if wrap_second_axis else (n1 - 1)
        for i0 in range(n0 - 1):
            for i1 in range(second_limit):
                j1 = (i1 + 1) % n1
                p00 = grid_points[i0, i1]
                p10 = grid_points[i0 + 1, i1]
                p11 = grid_points[i0 + 1, j1]
                p01 = grid_points[i0, j1]
                if reverse:
                    tris = ((p00, p11, p10), (p00, p01, p11))
                else:
                    tris = ((p00, p10, p11), (p00, p11, p01))
                append_clipped_triangle(vertices, intensities, faces, tris[0], density)
                append_clipped_triangle(vertices, intensities, faces, tris[1], density)

    simplex_normal = np.array([1.0, 1.0, 1.0], dtype=float)
    simplex_normal /= np.linalg.norm(simplex_normal)
    simplex_center = simplex_normal * (simplex_threshold / np.linalg.norm(np.array([1.0, 1.0, 1.0])))
    simplex_u = np.array([1.0, -1.0, 0.0], dtype=float)
    simplex_u /= np.linalg.norm(simplex_u)
    simplex_v = np.cross(simplex_normal, simplex_u)
    simplex_v /= np.linalg.norm(simplex_v)

    figure = go.Figure()
    for draw_index, band_index in enumerate(band_order):
        band = bands[int(band_index)]
        density = band["density_mid"]
        quantile_mid = band["q_mid"]
        r_inner = band["r_inner"]
        r_outer = band["r_outer"]
        vertices: list[np.ndarray] = []
        intensities: list[float] = []
        faces: list[tuple[int, int, int]] = []

        alpha = np.linspace(0.0, 0.5 * np.pi, polar_resolution)
        beta = np.linspace(0.0, 0.5 * np.pi, azimuth_resolution)
        alpha_grid, beta_grid = np.meshgrid(alpha, beta, indexing="ij")

        for radius, reverse in ((r_outer, False), (r_inner, True)):
            sphere_points = np.stack(
                [
                    radius * np.cos(alpha_grid),
                    radius * np.sin(alpha_grid) * np.cos(beta_grid),
                    radius * np.sin(alpha_grid) * np.sin(beta_grid),
                ],
                axis=-1,
            )
            append_grid_triangles(
                vertices,
                intensities,
                faces,
                sphere_points,
                density,
                wrap_second_axis=False,
                reverse=reverse,
            )

        rho_theta = np.linspace(0.0, 0.5 * np.pi, cut_face_resolution)

        def add_axis_plane(which: str) -> None:
            rho_inner = math.sqrt(max(0.0, r_inner * r_inner - cartesian_epsilon * cartesian_epsilon))
            rho_outer = math.sqrt(max(0.0, r_outer * r_outer - cartesian_epsilon * cartesian_epsilon))
            if rho_outer <= 0.0:
                return
            rho = np.linspace(rho_inner, rho_outer, cut_face_resolution)
            rho_grid, theta_grid = np.meshgrid(rho, rho_theta, indexing="ij")
            a = rho_grid * np.cos(theta_grid)
            b = rho_grid * np.sin(theta_grid)
            if which == "x":
                points = np.stack([np.full_like(a, cartesian_epsilon), a, b], axis=-1)
            elif which == "y":
                points = np.stack([a, np.full_like(a, cartesian_epsilon), b], axis=-1)
            else:
                points = np.stack([a, b, np.full_like(a, cartesian_epsilon)], axis=-1)
            append_grid_triangles(vertices, intensities, faces, points, density, reverse=False)

        add_axis_plane("x")
        add_axis_plane("y")
        add_axis_plane("z")

        simplex_distance = simplex_threshold / np.linalg.norm(np.array([1.0, 1.0, 1.0]))
        if r_outer > simplex_distance:
            rho_inner = math.sqrt(max(0.0, r_inner * r_inner - simplex_distance * simplex_distance))
            rho_outer = math.sqrt(max(0.0, r_outer * r_outer - simplex_distance * simplex_distance))
            rho = np.linspace(rho_inner, rho_outer, cut_face_resolution)
            theta = np.linspace(0.0, 2.0 * np.pi, cut_face_resolution, endpoint=False)
            rho_grid, theta_grid = np.meshgrid(rho, theta, indexing="ij")
            simplex_points = (
                simplex_center[None, None, :]
                + rho_grid[..., None]
                * (
                    np.cos(theta_grid)[..., None] * simplex_u[None, None, :]
                    + np.sin(theta_grid)[..., None] * simplex_v[None, None, :]
                )
            )
            append_grid_triangles(
                vertices,
                intensities,
                faces,
                simplex_points,
                density,
                wrap_second_axis=True,
                reverse=False,
            )

        if not faces:
            continue
        vertex_array = np.asarray(vertices, dtype=float)
        intensity_array = np.asarray(intensities, dtype=float)
        face_array = np.asarray(faces, dtype=int)
        figure.add_trace(
            go.Mesh3d(
                x=vertex_array[:, 0],
                y=vertex_array[:, 1],
                z=vertex_array[:, 2],
                i=face_array[:, 0],
                j=face_array[:, 1],
                k=face_array[:, 2],
                intensity=intensity_array,
                intensitymode="vertex",
                cmin=0.0,
                cmax=1.0,
                colorscale="Turbo",
                opacity=surface_opacity,
                flatshading=True,
                showscale=draw_index == len(band_order) - 1,
                colorbar=dict(title="Normalized<br>shell density") if draw_index == len(band_order) - 1 else None,
                hovertemplate=(
                    f"shell band<br>"
                    f"s in [{r_inner:.3f}, {r_outer:.3f}]<br>"
                    f"center density={density:.3f}<br>"
                    f"center radial quantile={quantile_mid:.3f}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
                lighting=dict(ambient=0.85, diffuse=0.35, roughness=0.95, specular=0.02),
                lightposition=dict(x=-200, y=-120, z=150),
            )
        )

    peak = peak_normalized_radius(dimension)
    diag = np.array([1.0, 1.0, 1.0], dtype=float)
    diag /= np.linalg.norm(diag)
    radial_line = np.linspace(0.0, bands[-1]["r_outer"], 200)
    covered_quadrant_mass = float(bands[-1]["q_outer"] - bands[0]["q_inner"])
    figure.add_trace(
        go.Scatter3d(
            x=radial_line * diag[0],
            y=radial_line * diag[1],
            z=radial_line * diag[2],
            mode="lines",
            line=dict(width=4, color="rgba(255, 255, 255, 0.85)", dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    axis_label = f"(normalized by sqrt({dimension}))"
    figure.update_layout(
        template="plotly_white",
        title=(
            f"{dimension}D Gaussian radial shell density in the first octant<br>"
            "Positive-octant shell bands with a fixed simplex wedge removed to expose staggered layer depth"
        ),
        margin=dict(l=0, r=0, t=72, b=0),
        annotations=[
            dict(
                x=0.70,
                y=0.83,
                xref="paper",
                yref="paper",
                text=f"Shaded shell contains {100.0 * covered_quadrant_mass:.1f}% of first-octant probability mass",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.88)",
                bordercolor="rgba(90, 98, 110, 0.25)",
                borderwidth=1,
                font=dict(size=13, color="rgb(55, 63, 77)"),
                align="left",
            )
        ],
        scene=dict(
            xaxis=axis_style(f"x {axis_label}", -radius_max, radius_max),
            yaxis=axis_style(f"y {axis_label}", -radius_max, radius_max),
            zaxis=axis_style(f"z {axis_label}", -radius_max, radius_max),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                center=dict(x=0.28, y=0.14, z=-0.10),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
            dragmode="orbit",
        ),
    )
    return figure


def build_profile_figure(
    *,
    dimension: int,
    radius_max: float,
    shell_count: int,
    display_mass_window: float,
) -> go.Figure:
    grid = np.linspace(0.001, radius_max, 4000)
    profile = normalized_shell_density(grid, dimension)
    peak = peak_normalized_radius(dimension)
    bands = quantile_shell_bands(dimension, radius_max, shell_count, display_mass_window)
    band_mid_radii = np.array([band["r_mid"] for band in bands], dtype=float)
    band_mid_profile = normalized_shell_density(band_mid_radii, dimension)
    band_mid_quantiles = np.array([band["q_mid"] for band in bands], dtype=float)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=profile,
            mode="lines",
            line=dict(color="rgb(13, 110, 253)", width=3),
            fill="tozeroy",
            fillcolor="rgba(13, 110, 253, 0.12)",
            hovertemplate="s=%{x:.4f}<br>density=%{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=band_mid_radii,
            y=band_mid_profile,
            mode="markers",
            marker=dict(
                size=7,
                color=band_mid_quantiles,
                colorscale="Turbo",
                cmin=0.0,
                cmax=1.0,
                line=dict(width=0.5, color="rgba(30, 30, 30, 0.40)"),
            ),
            name="Quantile shell centers",
            hovertemplate=(
                "s=%{x:.4f}<br>density=%{y:.4f}<br>"
                "radial mass quantile=%{marker.color:.3f}<extra></extra>"
            ),
        )
    )
    for band in bands:
        figure.add_vrect(
            x0=band["r_inner"],
            x1=band["r_outer"],
            fillcolor="rgba(13, 110, 253, 0.05)",
            line_width=0,
            layer="below",
        )
    figure.add_vline(x=peak, line_dash="dash", line_color="rgb(220, 53, 69)", line_width=2)
    figure.add_annotation(
        x=peak,
        y=1.0,
        text=f"peak at s ~= {peak:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=70,
        ay=-35,
        bgcolor="rgba(255, 255, 255, 0.90)",
    )
    figure.update_layout(
        template="plotly_white",
        title=f"{dimension}D Gaussian normalized radial shell density",
        xaxis_title=f"s = r / sqrt({dimension})",
        yaxis_title="density / max density",
        margin=dict(l=70, r=30, t=60, b=60),
    )
    return figure


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_figure = build_halfspace_figure(
        dimension=args.dimension,
        radius_max=args.rmax,
        shell_count=max(8, args.shell_count),
        polar_resolution=max(12, args.polar_resolution),
        azimuth_resolution=max(24, args.azimuth_resolution),
        disk_resolution=max(40, args.disk_resolution),
        target_total_opacity=args.target_total_opacity,
        boundary_epsilon=args.boundary_epsilon,
        display_mass_window=args.display_mass_window,
    )
    profile_figure = build_profile_figure(
        dimension=args.dimension,
        radius_max=args.rmax,
        shell_count=max(8, args.shell_count),
        display_mass_window=args.display_mass_window,
    )

    main_path = output_dir / f"gaussian_{args.dimension}d_halfspace_density_plotly.html"
    profile_path = output_dir / f"gaussian_{args.dimension}d_radial_profile_plotly.html"

    main_figure.write_html(str(main_path), include_plotlyjs=True, full_html=True)
    profile_figure.write_html(str(profile_path), include_plotlyjs=True, full_html=True)

    print(f"Saved main figure: {main_path}")
    print(f"Saved profile figure: {profile_path}")
    print(f"Peak shell radius s ~= {peak_normalized_radius(args.dimension):.6f}")


if __name__ == "__main__":
    main()
