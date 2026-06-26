
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Setup intrinsic coordinate grids
res_r = 60
res_t = 120
r_grid = np.linspace(0.05, 1.0, res_r)  # Avoid absolute 0 for singularity plotting
theta_grid = np.linspace(0, 2 * np.pi, res_t)
R_COORD, T_COORD = np.meshgrid(r_grid, theta_grid)

# Logarithmic grid for the Affine Throat (u = log(r))
u_grid = np.linspace(-4, 0, res_r)
U_COORD, T_U_COORD = np.meshgrid(u_grid, theta_grid)

# Parameter configuration for the (1, 2) Lorentzian dual regime
alpha_vals = np.linspace(1.20, 1.95, 30)
current_alpha = 1.50

# 2. Computation Functions for the 4 Dual Visualizations

def compute_deficit_cone(r_mat, theta_mat, alpha):
    """Visual 1: Absolute Spatial Geometry (Deficit Cone)"""
    gamma_tilde = (2 - alpha) / (2 * np.sqrt(alpha - 1))
    gamma_tilde = min(gamma_tilde, 1.0) 
    s_mat = (2 / (2 - alpha)) * (r_mat ** ((2 - alpha) / 2))
    X = s_mat * gamma_tilde * np.cos(theta_mat)
    Y = s_mat * gamma_tilde * np.sin(theta_mat)
    Z = s_mat * np.sqrt(1 - gamma_tilde**2)
    return X, Y, Z, s_mat

def compute_gravity_well(r_mat, theta_mat, alpha):
    """Visual 2: Lorentzian Gradient Collapse (Gravity Well)"""
    X = r_mat * np.cos(theta_mat)
    Y = r_mat * np.sin(theta_mat)
    Z = - (1 / (alpha - 1)) * (r_mat ** (- (alpha - 1)))
    return X, Y, Z

def compute_conformal_crush(r_mat, theta_mat, alpha):
    """Visual 3: Conformal Inversion (Holographic Boundary pushed to 0)"""
    R_dual = 1 / r_mat
    X = r_mat * np.cos(theta_mat)
    Y = r_mat * np.sin(theta_mat)
    Z = - (R_dual ** (alpha - 1)) 
    return X, Y, Z, R_dual

def compute_logarithmic_throat(u_mat, theta_mat, alpha):
    """Visual 4: Affine Logarithmic Throat (u = log(r))"""
    X = np.cos(theta_mat)
    Y = np.sin(theta_mat)
    Z = u_mat
    volume_metric = (1 / (alpha - 1)) * np.exp(-alpha * u_mat)
    return X, Y, Z, volume_metric

# 3. Compute initial frame data
X1, Y1, Z1, C1 = compute_deficit_cone(R_COORD, T_COORD, current_alpha)
X2, Y2, Z2 = compute_gravity_well(R_COORD, T_COORD, current_alpha)
X3, Y3, Z3, C3 = compute_conformal_crush(R_COORD, T_COORD, current_alpha)
X4, Y4, Z4, C4 = compute_logarithmic_throat(U_COORD, T_U_COORD, current_alpha)

# Create 2x2 Subplots
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}],
           [{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=(
        "<b>1. Spatial Deficit Cone</b><br>Absolute Geometry (Volume Loss)",
        "<b>2. Lorentzian Gravity Well</b><br>Inward Gradient Collapse",
        "<b>3. Conformal Inversion</b><br>Holographic Boundary Crushed to x₀",
        "<b>4. Affine Logarithmic Throat</b><br>u-Coordinate Wormhole"
    ),
    horizontal_spacing=0.05,
    vertical_spacing=0.08
)

# Add Traces
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, surfacecolor=C1, colorscale="Plasma", colorbar=dict(x=-0.05, len=0.45, y=0.8, title="Proper s")), row=1, col=1)
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, colorscale="Inferno", colorbar=dict(x=1.05, len=0.45, y=0.8, title="Gradient Mag")), row=1, col=2)
fig.add_trace(go.Surface(x=X3, y=Y3, z=Z3, surfacecolor=C3, colorscale="Viridis", colorbar=dict(x=-0.05, len=0.45, y=0.2, title="Dual R")), row=2, col=1)
fig.add_trace(go.Surface(x=X4, y=Y4, z=Z4, surfacecolor=C4, colorscale="Magma", colorbar=dict(x=1.05, len=0.45, y=0.2, title="Metric Vol")), row=2, col=2)

# 4. Generate dynamic animation frames
frames = []
slider_steps = []

for alpha in alpha_vals:
    X1_f, Y1_f, Z1_f, C1_f = compute_deficit_cone(R_COORD, T_COORD, alpha)
    X2_f, Y2_f, Z2_f = compute_gravity_well(R_COORD, T_COORD, alpha)
    X3_f, Y3_f, Z3_f, C3_f = compute_conformal_crush(R_COORD, T_COORD, alpha)
    X4_f, Y4_f, Z4_f, C4_f = compute_logarithmic_throat(U_COORD, T_U_COORD, alpha)
    
    frame_name = f"alpha_{alpha:.2f}"
    frames.append(go.Frame(
        data=[
            dict(type="surface", x=X1_f, y=Y1_f, z=Z1_f, surfacecolor=C1_f),
            dict(type="surface", x=X2_f, y=Y2_f, z=Z2_f),
            dict(type="surface", x=X3_f, y=Y3_f, z=Z3_f, surfacecolor=C3_f),
            dict(type="surface", x=X4_f, y=Y4_f, z=Z4_f, surfacecolor=C4_f)
        ],
        name=frame_name
    ))
    
    step = dict(
        method="animate",
        args=[[frame_name], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
        label=f"{alpha:.2f}"
    )
    slider_steps.append(step)

sliders = [dict(
    active=np.argmin(np.abs(alpha_vals - current_alpha)),
    currentvalue={"prefix": "Lorentzian Kernel Alpha (α > 1): ", "font": {"size": 18}},
    pad={"t": 40, "b": 10},
    steps=slider_steps
)]

# 5. Global Layout Operations
scene_dict = dict(
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    zaxis=dict(showgrid=False, zeroline=False, visible=False),
    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    aspectmode="data"
)

fig.update_layout(
    title={
        "text": "<b>The Lorentzian Dual Manifold Visualizations (α ∈ 1 to 2)</b><br>Black Hole Analogs and Conformal Inversions",
        "y": 0.98, "x": 0.5, "xanchor": "center"
    },
    scene=scene_dict, scene2=scene_dict, scene3=scene_dict, scene4=scene_dict,
    sliders=sliders,
    width=1400, height=1100,
    margin=dict(l=50, r=50, t=100, b=50),
    plot_bgcolor="rgb(15, 15, 20)",
    paper_bgcolor="rgb(15, 15, 20)",
    font=dict(color="white")
)

# Enforce uniform aspect ratios for logarithmic and conformal charts
fig.layout.scene3.aspectmode = "cube"
fig.layout.scene4.aspectmode = "manual"
fig.layout.scene4.aspectratio = dict(x=1, y=1, z=2)

fig.frames = frames
fig.show()

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Setup the intrinsic coordinate grid (Geodesic distance 's' and Angle 'theta')
s_res, t_res = 60, 250
s_grid = np.linspace(0, 1, s_res)
theta_grid = np.linspace(0, 2 * np.pi, t_res)  
S_COORD, T_COORD = np.meshgrid(s_grid, theta_grid)

# Parameter configuration
N_RUFFLES_DRAMATIC = 10  
alpha_vals = np.linspace(0.0, 0.90, 40)  
current_alpha = 0.50  # Initial active slide position

# 2. Computation Functions for each visualization
def compute_dramatic_folds(s_mat, theta_mat, alpha, n_ruffles):
    if alpha == 0:
        gamma = 1.0
        A_f = 0.0
    else:
        gamma = (2 - alpha) / (2 * np.sqrt(1 - alpha))
        A_f = np.sqrt(2 * (gamma**2 - 1)) / n_ruffles

    X = s_mat * np.cos(theta_mat)
    Y = s_mat * np.sin(theta_mat)
    Z = A_f * s_mat * np.sin(n_ruffles * theta_mat)
    return X, Y, Z

def compute_intrinsic_spiral(s_mat, theta_mat, alpha):
    if alpha == 0:
        gamma = 1.0
    else:
        gamma = (2 - alpha) / (2 * np.sqrt(1 - alpha))

    Theta_Intrinsic = theta_mat * gamma
    X_spir = s_mat * np.cos(Theta_Intrinsic)
    Y_spir = s_mat * np.sin(Theta_Intrinsic)
    Z_spir = 0.2 * (Theta_Intrinsic / (2 * np.pi))
    Layer_Count = np.floor(Theta_Intrinsic / (2 * np.pi))
    return X_spir, Y_spir, Z_spir, Layer_Count

# 3. Compute initial frame data (Alpha = 0.50)
X_df, Y_df, Z_df = compute_dramatic_folds(S_COORD, T_COORD, current_alpha, N_RUFFLES_DRAMATIC)
X_is, Y_is, Z_is, Layers_is = compute_intrinsic_spiral(S_COORD, T_COORD, current_alpha)

# Create Subplots: Left = Dramatic Folds, Right = Spiral Flattening
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=(
        "<b>Visual 1: Dramatic Ruffled Folds</b><br>Severe Embedding (High α)",
        "<b>Visual 2: Intrinsic Spiral (Conceptual Flattening)</b><br>Visualizing Excess Sheets"
    ),
    horizontal_spacing=0.08,
)

# Add Dramatic Folds Trace to Column 1
surface_1 = go.Surface(
    x=X_df, y=Y_df, z=Z_df,
    colorscale="Viridis",
    colorbar=dict(title="Intrinsic Distance s", len=0.8, x=-0.05),
    showscale=True,
    opacity=0.9,
    lighting=dict(ambient=0.3, diffuse=0.9, roughness=0.1, specular=0.8)
)
fig.add_trace(surface_1, row=1, col=1)

# Add Intrinsic Spiral Trace directly to Column 2 (Fixed Bug Here)
surface_2 = go.Surface(
    x=X_is, y=Y_is, z=Z_is,
    colorscale="RdYlBu",
    surfacecolor=Layers_is,
    cmin=0, cmax=2,
    colorbar=dict(
        title="Excess Angle Layer Count",
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Sheet 1 (0-2π)", "Sheet 2 (2π-4π)", "Sheet 3 (4π+)"],
        x=1.05
    ),
    showscale=True,
    opacity=0.8
)
fig.add_trace(surface_2, row=1, col=2)

# 4. Generate the dynamic animation frames for the slider
frames = []
slider_steps = []

for alpha in alpha_vals:
    X_df_f, Y_df_f, Z_df_f = compute_dramatic_folds(S_COORD, T_COORD, alpha, N_RUFFLES_DRAMATIC)
    X_is_f, Y_is_f, Z_is_f, Layers_is_f = compute_intrinsic_spiral(S_COORD, T_COORD, alpha)

    frame_name = f"alpha_{alpha:.2f}"
    frames.append(
        go.Frame(
            data=[
                dict(type="surface", x=X_df_f, y=Y_df_f, z=Z_df_f),  # Maps to Trace 0 (Col 1)
                dict(type="surface", x=X_is_f, y=Y_is_f, z=Z_is_f, surfacecolor=Layers_is_f)  # Maps to Trace 1 (Col 2)
            ],
            name=frame_name
        )
    )

    step = dict(
        method="animate",
        args=[[frame_name], dict(mode="immediate", frame=dict(duration=120, redraw=True), transition=dict(duration=0))],
        label=f"{alpha:.2f}"
    )
    slider_steps.append(step)

sliders = [
    dict(
        active=len(alpha_vals) // 2,  # Start mid-way through values to match alpha=0.50 setup
        currentvalue={"prefix": "Kernel Alpha (α): ", "font": {"size": 18}},
        pad={"t": 70, "b": 10},
        steps=slider_steps
    )
]

# 5. Global Layout Options
axis_props = dict(range=[-1.1, 1.1], gridcolor="white", zerolinecolor="white")
scene_layout = dict(
    xaxis=axis_props, yaxis=axis_props,
    zaxis=dict(range=[-0.9, 0.9], gridcolor="white", zerolinecolor="white"),
    camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.3, y=1.3, z=0.9)),
    aspectmode="manual",
    aspectratio=dict(x=1, y=1, z=0.8),
    bgcolor="rgb(229, 236, 246)"
)

fig.update_layout(
    title={
        "text": "<b>Hyperbolic Cone Fiber Manifold Visualizations</b><br>Imposed by Kernel W(x - x₀)",
        "y": 0.97, "x": 0.5, "xanchor": "center"
    },
    scene=scene_layout,
    scene2=scene_layout,
    sliders=sliders,
    width=1400, height=800
)

fig.update_layout(scene_zaxis_title="Z (Fold Amplitude)", scene2_zaxis_title="Z (Stacked Offset)")
fig.frames = frames

fig.show()