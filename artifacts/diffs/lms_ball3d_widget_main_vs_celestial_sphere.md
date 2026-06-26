# `lms_ball3d_widget.py` diff: `main` → `celestial_sphere`

Generated: 2026-06-12

## Comparison

| | `main` | `celestial_sphere` |
|---|---|---|
| **Branch tip** | `2caff58` | `c8a2d6d` |
| **Last commit touching file** | `33e5da2` — *cannonical inversion* | `dbb9047` — *experiment yaml orchestration* (bulk: `a812b2e` — *celestial sphere, website*) |
| **Blob** | `6403080` | `5761166` |
| **Lines** | 6,813 | 7,068 (+255 net) |
| **Diff size** | +433 / −178 lines across 37 hunks | |

**Companion artifact:** [`lms_ball3d_widget_main_vs_celestial_sphere.patch`](./lms_ball3d_widget_main_vs_celestial_sphere.patch) (938 lines, unified diff)

## Regenerate

```bash
git diff main..celestial_sphere -- src/lmsspp/lms_ball3d_widget.py \
  > artifacts/diffs/lms_ball3d_widget_main_vs_celestial_sphere.patch
```

View in terminal:

```bash
git diff main..celestial_sphere -- src/lmsspp/lms_ball3d_widget.py | less
```

Apply onto a `main` checkout (creates a dirty working tree):

```bash
git checkout main
git apply artifacts/diffs/lms_ball3d_widget_main_vs_celestial_sphere.patch
```

## Thematic summary

### 1. Canonical gauge API rename

- Import `canonical_center` → `canonical_cloud` (both relative and fallback import paths).

### 2. Center estimation simplified to exact Busemann only

- `CenterEstimationMode` was `Literal["poisson_shrink", "busemann_exact"]`; now only `"busemann_exact"`.
- Center-estimation dropdown is disabled and fixed to **Exact Busemann**.
- `_on_center_estimation_change` in `LMSBall3DHydrodynamicEnsembleWidget` no longer reacts to UI changes; it always forces `"busemann_exact"`.
- Default job center mode: `"poisson_shrink"` → `"busemann_exact"`.

### 3. New 2D inversive / spherical-inversion chart (largest UI addition)

New tabbed visual layout on `LMSBall3DWidget`:

- `visual_tabs` — **3D Sphere** + **2D Inversive Chart**
- `projection_fig`, `projection_box`, `projection_status_html`
- `btn_projection_chart` toggles rescaled vs unscaled chart
- Gauge diagnostics: `_last_gauge_residual_norm`, `_last_gauge_center_error`, `_last_gauge_converged`

**New methods (~175 lines in one hunk):**

| Method | Role |
|---|---|
| `_apply_inversive_projection_to_widget` | Push projected traces into the Plotly widget |
| `_sync_projection_chart_button_label` | Update button label for chart mode |
| `_projection_basis_from_pole` | Orthonormal chart basis from pole |
| `_spherical_inversion_project_2d` | Map S² points to 2D chart coords |
| `_spherical_inversion_grid_2d` | Chart grid curves |
| `_line_segments_from_projected_series` | Build disconnected path segments for Plotly |
| `_projection_pole_from_state` | Pole from reduced state |
| `_projection_chart_omega` | Chart dilation parameter |
| `_projection_chart_scale` | Effective chart scale |
| `_projected_point_paths_2d` | Time series of projected boundary paths |
| `_on_projection_chart_clicked` | Toggle rescaled/unscaled |
| `_on_visual_tab_change` | Re-render on tab switch |

### 4. Boundary → reduced-state pipeline refactored

**Removed:**

- `_estimate_w_from_boundary_points_poisson` (+ inner `q_of_r`)
- `_estimate_w_from_boundary_points_busemann`
- `_estimate_w_from_boundary_points` (dispatcher)

**Added:**

- `_exact_reduced_state_from_observed_cloud` — single exact canonical path from observed boundary cloud

Downstream callers (`_reduced_state_from_boundary_points`, entropy-shell init, hydro ensemble init) now use the exact path and often return `(w0, base_points)` directly instead of `(x_points, w0)` followed by `_recover_base_points_from_state`.

### 5. Hydrodynamic ensemble widget

- Drops `_target_w_from_radius` / `target_w=` when building initial reduced state from boundary points.
- Center-estimation handler simplified (see §2).

### 6. Entropy-shell mixin & widgets

- `_make_energy_shell_boundary_points` return signature changes: `(x0_points, w0)` → `(w0, base_points)`.
- All branches that called `_estimate_w_from_boundary_points` now call `_exact_reduced_state_from_observed_cloud`.
- Poisson reference mode also runs through exact reduced-state recovery.
- `LMSBall3DEntropyShellEnsembleWidget` and `LMSBall3DEntropyShellTwoSheetWidget` no longer call `_recover_base_points_from_state` after shell init.

## Hunk index (by `celestial_sphere` line)

Approximate locations of each change block in the new file:

| Hunk | ~Line (new) | Area |
|---:|---:|---|
| 1–2 | 25, 43 | `canonical_cloud` import |
| 3 | 148 | `CenterEstimationMode` |
| 4–7 | 301–615 | `LMSBall3DWidget.__init__`, gauge fields, visual tabs, projection UI |
| 8 | 801 | Projection geometry helpers (large block) |
| 9–11 | 1226–1292 | Rendering / state hooks |
| 12 | 1649 | More projection / chart logic |
| 13–16 | 2230–2333 | Integration with frame render |
| 17 | 2633 | `_exact_reduced_state_from_observed_cloud` (+ removal of estimators) |
| 18–21 | 3223–3754 | `_reduced_state_from_boundary_points` refactor, chart handlers |
| 22–24 | 3857–3935 | Layout / export tweaks |
| 25–27 | 5194–5463 | `LMSBall3DHydrodynamicEnsembleWidget` |
| 28–32 | 6700–6771 | `_LMSEntropyShellMixin._make_energy_shell_boundary_points` |
| 33–34 | 6875–6986 | Entropy-shell ensemble & two-sheet widgets |

## Commits on `celestial_sphere` not in `main`

```
dbb9047  Me making reproducible code for once in my life (experiment yaml orchestration)  [+1 line]
a812b2e  celestial sphere, website, "original_work" folder with md                     [+432 / −178]
```

## Unchanged top-level structure

Same classes and module-level helpers on both branches:

`LMS3DControlSpec`, `_HydroRecomputeCancelled`, `_EntropyShellTable`, `_monotone_decreasing`, `_poisson_density_s2`, `_entropy_shell_table`, `_sphere_wireframe_traces`, `_angles_to_unit`, `LMSBall3DWidget`, `LMSBall3DBackwardTwoSheetWidget`, `LMSBall3DHydrodynamicEnsembleWidget`, `_LMSEntropyShellMixin`, `LMSBall3DEntropyShellEnsembleWidget`, `LMSBall3DEntropyShellTwoSheetWidget`
