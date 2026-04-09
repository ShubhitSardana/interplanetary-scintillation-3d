import os
# Force EGL before any VTK/PyVista import — unset DISPLAY so VTK never
# tries X11 at all (avoids "bad X server connection" warnings)
os.environ.pop("DISPLAY", None)
os.environ["PYVISTA_OFF_SCREEN"]                   = "true"
os.environ["PYVISTA_OFF_SCREEN_RENDERING_BACKEND"] = "egl"
os.environ["VTK_DEFAULT_OPENGL_WINDOW"]            = "vtkEGLRenderWindow"

import json, glob, tempfile, shutil, subprocess
import multiprocessing
import numpy as np
from tqdm import tqdm
import imageio, imageio_ffmpeg
from astropy.time import Time
from astropy.coordinates import get_sun
import astropy.units as u
import pyvista as pv

pv.OFF_SCREEN = True
print("Backend: EGL (NVIDIA GPU offscreen)")

# ==============================================================================
# THEME  —  set THEME = "dark" or THEME = "light"
# ==============================================================================

THEME = "light"

_THEMES = {
    "dark": dict(
        BACKGROUND_COLOR             = (5,   5,   8),
        GLOBE_COLOR                  = (13,  13,  20),
        GLOBE_OPACITY                = 0.35,
        GLOBE_WIRE_COLOR             = (255, 255, 255),
        GLOBE_WIRE_OPACITY           = 0.06,
        EARTH_OCEAN_COLOR            = (13,  46,  97),
        EARTH_LAND_COLOR             = (33,  95,  33),
        EARTH_NIGHT_OCEAN_COLOR      = (3,   10,  26),
        EARTH_NIGHT_LAND_COLOR       = (10,  23,  10),
        EARTH_ICE_COLOR              = (230, 237, 247),
        EARTH_ATMOSPHERE_COLOR       = (51,  128, 255),
        EARTH_ATMOSPHERE_OPACITY     = 0.08,
        SUN_HALO_COLOR               = (255, 140,   0),
        SUN_CORE_COLOR               = (255, 220, 100),
        SUN_HALO_INTENSITY           = 2.5,
        SUN_CORE_INTENSITY           = 7.0,
        SUN_GLOW_OPACITY             = 0.85,
        GREAT_CIRCLE_COLOR           = (77,  179, 255),
        GREAT_CIRCLE_OPACITY         = 0.20,
        POINTS_CMAP                  = "plasma",
        POINTS_OPACITY               = 0.92,
        FLASH_COLOR                  = (255, 255, 255),
        AMBIENT_LIGHT                = 0.35,
        DIFFUSE_LIGHT                = 0.65,
        SPECULAR_LIGHT               = 0.15,
        TEXT_COLOR                   = "white",
    ),
    "light": dict(
        BACKGROUND_COLOR             = (255, 255, 255),
        GLOBE_COLOR                  = (255, 255, 255),
        GLOBE_OPACITY                = 0.08,
        GLOBE_WIRE_COLOR             = (30,  30,  60),
        GLOBE_WIRE_OPACITY           = 0.25,
        EARTH_OCEAN_COLOR            = (70,  140, 210),
        EARTH_LAND_COLOR             = (80,  160,  60),
        EARTH_NIGHT_OCEAN_COLOR      = (50,  110, 170),
        EARTH_NIGHT_LAND_COLOR       = (60,  130,  50),
        EARTH_ICE_COLOR              = (245, 250, 255),
        EARTH_ATMOSPHERE_COLOR       = (100, 170, 255),
        EARTH_ATMOSPHERE_OPACITY     = 0.12,
        SUN_HALO_COLOR               = (255, 140,   0),
        SUN_CORE_COLOR               = (255, 220, 100),
        SUN_GLOW_OPACITY             = 0.85,
        SUN_HALO_INTENSITY           = 2.5,
        SUN_CORE_INTENSITY           = 7.0,
        SUN_ALPHA_BOOST              = 0.8,
        GREAT_CIRCLE_COLOR           = (20,  100, 200),
        GREAT_CIRCLE_OPACITY         = 0.30,
        POINTS_CMAP                  = "plasma",
        POINTS_OPACITY               = 0.92,
        FLASH_COLOR                  = (255, 100,   0),
        AMBIENT_LIGHT                = 0.45,
        DIFFUSE_LIGHT                = 0.65,
        SPECULAR_LIGHT               = 0.15,
        TEXT_COLOR                   = "black",
    ),
}

_t = _THEMES.get(THEME, _THEMES["dark"])
globals().update(_t)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OUTPUT_FILENAME  = f"IPS-video-{THEME}.mp4"
RENDER_WORKERS   = 0          # 0 = auto; GPU saturates ~32, so auto will cap there
RENDER_WIDTH     = 1920
RENDER_HEIGHT    = 1080
ANIMATION_FPS    = 60

ANIMATION_CHUNK_SIZE         = 25 // 4
TOTAL_ANIMATION_DURATION_SEC = 0
SETTLE_DURATION_SEC          = 30

EXPORT_FRAME     = -1
EXPORT_FILENAME  = "poster_frame.png"

# --- Globe ---
GLOBE_RADIUS         = 1.0
GLOBE_RESOLUTION     = 200
GLOBE_WIRE_ENABLED   = True
GLOBE_WIRE_LAT_LINES = 18
GLOBE_WIRE_LON_LINES = 36

# --- Earth ---
EARTH_RADIUS                   = 0.12
EARTH_RESOLUTION               = 60
EARTH_ATMOSPHERE_RADIUS_FACTOR = 1.06

# --- Sun glow ---
SUN_ENABLED         = True
SUN_GLOW_HALO_POWER = 100
SUN_GLOW_CORE_POWER = 600
SUN_ALPHA_BOOST     = 0.8

# --- IPS Points ---
POINTS_RADIUS_FACTOR = 1.001
POINTS_BASE_SIZE     = 10
POINTS_MAX_SIZE      = 22

# --- Flash ---
FLASH_FRAMES    = 2
FLASH_SIZE_MULT = 1.5

# --- Great circles ---
GREAT_CIRCLE_ENABLED   = True
GREAT_CIRCLE_MAX_LINES = 40
GREAT_CIRCLE_LINEWIDTH = 3
GREAT_CIRCLE_STEPS     = 60

# --- Camera ---
INITIAL_CAMERA_AZIMUTH      = 170.0
CAMERA_ELEVATION_DEG        = 23.5
CAMERA_DISTANCE             = 5
FLY_IN_FRAMES               = 30
FLY_IN_START_DIST           = 10
ROTATION_SPEED_DEGS_PER_SEC = 12.0

HUD_ENABLED = True

SOLAR_CYCLE_DATA_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
TELESCOPE_LABELS = {
    "LOFAR": "LOFAR", "MWA": "MWA",
    "Ooty": "Ooty RT", "ORT": "Ooty RT", "GMRT": "uGMRT",
}

# ==============================================================================

_LAND_PATCHES = [
    (-168, -52,  10,  72), (-82, -34, -56,  13), ( -9,  40,  36,  71),
    ( -18,  52, -35,  38), ( 26, 145,  -5,  75), (100, 145, -10,  50),
    ( 113, 154, -44,  -9), (-57, -17,  59,  84), (-180, 180, -90, -65),
]


def _c(rgb255):
    return tuple(v / 255.0 for v in rgb255)


# ── Astropy helpers ───────────────────────────────────────────────────────────

def get_sun_vec(time_str):
    try:
        t = Time(time_str, scale="utc")
        s = get_sun(t)
        ra  = float(s.ra.to(u.rad).value)
        dec = float(s.dec.to(u.rad).value)
        return np.array([np.cos(dec) * np.cos(ra),
                         np.cos(dec) * np.sin(ra),
                         np.sin(dec)])
    except Exception:
        return np.array([1.0, 0.0, 0.0])


# ── Earth texture ─────────────────────────────────────────────────────────────

def _earth_texture(res_u, res_v):
    LON, LAT = np.meshgrid(np.linspace(-180, 180, res_u),
                            np.linspace( -90,  90, res_v))
    img = np.full((res_v, res_u, 3), EARTH_OCEAN_COLOR, dtype=np.uint8)
    for lo0, lo1, la0, la1 in _LAND_PATCHES:
        m = (LON >= lo0) & (LON <= lo1) & (LAT >= la0) & (LAT <= la1)
        img[m] = EARTH_LAND_COLOR
    img[np.abs(LAT) > 75] = EARTH_ICE_COLOR
    return img

# Precompute earth texture array once (shared across workers via args)
_EARTH_TEX_ARRAY = _earth_texture(256, 128)

def build_earth(sun_vec):
    sphere = pv.Sphere(radius=EARTH_RADIUS,
                       theta_resolution=EARTH_RESOLUTION,
                       phi_resolution=EARTH_RESOLUTION // 2)
    sphere.texture_map_to_sphere(inplace=True)
    tex = pv.Texture(_EARTH_TEX_ARRAY)
    return sphere, tex


# ── Globe ─────────────────────────────────────────────────────────────────────

def build_globe(sun_vec):
    mesh = pv.Sphere(radius=GLOBE_RADIUS,
                     theta_resolution=GLOBE_RESOLUTION,
                     phi_resolution=GLOBE_RESOLUTION // 2)
    pts = mesh.points
    n   = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
    su  = sun_vec / (np.linalg.norm(sun_vec) + 1e-9)
    dot = n @ su

    bg   = np.array(_c(BACKGROUND_COLOR), dtype=np.float32)
    rgba = np.zeros((len(pts), 4), dtype=np.float32)
    rgba[:, :3] = bg
    rgba[:, 3]  = GLOBE_OPACITY

    if SUN_ENABLED:
        gh = np.power(np.clip(dot, 0, 1), SUN_GLOW_HALO_POWER)
        gc = np.power(np.clip(dot, 0, 1), SUN_GLOW_CORE_POWER)
        hc = np.array(_c(SUN_HALO_COLOR), dtype=np.float32)
        cc = np.array(_c(SUN_CORE_COLOR),  dtype=np.float32)
        blend   = np.clip(gh * SUN_HALO_INTENSITY + gc * SUN_CORE_INTENSITY, 0, 1)
        total_w = gh * SUN_HALO_INTENSITY + gc * SUN_CORE_INTENSITY + 1e-9
        sun_rgb = (gh[:, None] * hc * SUN_HALO_INTENSITY +
                   gc[:, None] * cc * SUN_CORE_INTENSITY) / total_w[:, None]
        rgba[:, :3] = bg * (1 - blend[:, None]) + sun_rgb * blend[:, None]
        rgba[:, 3]  = np.maximum(GLOBE_OPACITY, blend * SUN_GLOW_OPACITY)

    mesh["colors"] = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    return mesh


# ── Wireframe — precomputed as plain numpy point arrays ───────────────────────

def precompute_wireframe():
    """Returns list of (N,3) numpy arrays — one per lat/lon line."""
    lines = []
    for lat in np.linspace(-90, 90, GLOBE_WIRE_LAT_LINES + 2)[1:-1]:
        lr  = np.radians(lat)
        rxy = GLOBE_RADIUS * np.cos(lr)
        z   = GLOBE_RADIUS * np.sin(lr)
        t   = np.linspace(0, 2 * np.pi, 120)
        lines.append(np.c_[rxy * np.cos(t), rxy * np.sin(t), np.full_like(t, z)])
    for lon in np.linspace(0, 360, GLOBE_WIRE_LON_LINES, endpoint=False):
        lr = np.radians(lon)
        p  = np.linspace(0, 2 * np.pi, 120)
        lines.append(np.c_[GLOBE_RADIUS * np.cos(p) * np.cos(lr),
                            GLOBE_RADIUS * np.cos(p) * np.sin(lr),
                            GLOBE_RADIUS * np.sin(p)])
    return lines   # list of (120,3) arrays


def wire_pts_to_mesh(pts_array):
    """Reconstruct a pv.PolyData line from a (N,3) numpy array."""
    n    = len(pts_array)
    cell = np.r_[n, np.arange(n)].astype(np.int_)
    pd   = pv.PolyData()
    pd.points = pts_array
    pd.lines  = cell
    return pd


# ── Great circles — precomputed as plain numpy point arrays ──────────────────

def precompute_great_circles(xs, ys, zs):
    """Returns list of (GREAT_CIRCLE_STEPS, 3) arrays, one per point."""
    ts  = np.linspace(0, 1, GREAT_CIRCLE_STEPS)
    out = []
    for i in range(len(xs)):
        ep = np.array([xs[i], ys[i], zs[i]])
        if np.linalg.norm(ep) > 1e-9:
            out.append(np.outer(ts, ep))   # (steps, 3)
        else:
            out.append(None)
    return out


def gc_pts_to_mesh(pts_array):
    n    = len(pts_array)
    cell = np.r_[n, np.arange(n)].astype(np.int_)
    pd   = pv.PolyData()
    pd.points = pts_array
    pd.lines  = cell
    return pd


# ── IPS point clouds ──────────────────────────────────────────────────────────

def build_ips_cloud(xs, ys, zs, sis, end_idx):
    if end_idx == 0:
        return None
    cloud       = pv.PolyData(np.c_[xs[:end_idx], ys[:end_idx], zs[:end_idx]])
    cloud["SI"] = np.array(sis[:end_idx])
    return cloud


def build_flash_cloud(xs, ys, zs, end_idx, chunk_size, alpha):
    if alpha <= 0.01 or end_idx == 0:
        return None
    fs = max(0, end_idx - FLASH_FRAMES * chunk_size)
    if fs >= end_idx:
        return None
    return pv.PolyData(np.c_[xs[fs:end_idx], ys[fs:end_idx], zs[fs:end_idx]])


# ── Camera ────────────────────────────────────────────────────────────────────

def camera_for_frame(frame_idx):
    az  = np.radians(INITIAL_CAMERA_AZIMUTH +
                     frame_idx * (ROTATION_SPEED_DEGS_PER_SEC / ANIMATION_FPS))
    el  = np.radians(CAMERA_ELEVATION_DEG)
    if frame_idx < FLY_IN_FRAMES:
        t    = 1 - (1 - frame_idx / FLY_IN_FRAMES) ** 3
        dist = FLY_IN_START_DIST + (CAMERA_DISTANCE - FLY_IN_START_DIST) * t
    else:
        dist = CAMERA_DISTANCE
    return (dist * np.cos(el) * np.sin(az),
            -dist * np.cos(el) * np.cos(az),
             dist * np.sin(el))


# ── Solar cycle (date-based, monotonic — no jumps) ────────────────────────────

def get_solar_cycle_data():
    import urllib.request
    print("Fetching solar cycle data...")
    try:
        with urllib.request.urlopen(SOLAR_CYCLE_DATA_URL) as r:
            data = json.loads(r.read().decode())
        out = {e["time-tag"]: float(e["ssn"])
               for e in data if e.get("time-tag") and e.get("ssn") is not None}
        print(f"Loaded {len(out)} months.")
        return out
    except Exception as e:
        print(f"Solar cycle fetch error: {e}")
        return {}


def compute_cycle_limits(sd, s="2008-12", e="2019-12"):
    if not sd:
        return np.nan, np.nan
    v = [x for k, x in sd.items() if s <= k <= e]
    return (float(np.min(v)), float(np.max(v))) if v else (np.nan, np.nan)


def get_cycle_pct(date_iso, sd, mn, mx):
    """
    Date-based monotonic phase within Solar Cycle 24.
    Bar rises from 0% (Dec 2008 minimum) to 50% (Apr 2014 peak)
    then rises from 50% to 100% (Dec 2019 minimum).
    No dependency on noisy month-to-month SSN values.
    """
    try:
        from datetime import date
        d     = date.fromisoformat(date_iso[:10])
        start = date(2008, 12, 1)   # cycle 24 minimum
        peak  = date(2014,  4, 1)   # cycle 24 maximum
        end   = date(2019, 12, 1)   # cycle 25 minimum

        if d <= peak:
            pct = (d - start).days / (peak - start).days * 50.0
            lbl = "Solar Minimum" if pct < 10 else "Rising Phase"
        else:
            pct = 50.0 + (d - peak).days / (end - peak).days * 50.0
            lbl = "Declining Phase" if pct < 90 else "Solar Minimum"

        return float(np.clip(pct, 0, 100)), lbl
    except Exception:
        return np.nan, "Error"


# ── HUD ───────────────────────────────────────────────────────────────────────

def add_hud(frame_arr, time_str, source, pct, label,
            end_idx, si_max, frame_idx, total_frames, settle_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as mpe
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    H, W = frame_arr.shape[:2]
    fig  = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax   = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.patch.set_alpha(0.0)

    stroke_bg = "white" if TEXT_COLOR == "black" else "black"
    sk        = [mpe.withStroke(linewidth=3, foreground=stroke_bg)]
    nice      = time_str.replace("T", " ")[:16]

    ax.text(W / 2, H - 30, "Interplanetary Scintillation MAP",
            color=TEXT_COLOR, fontsize=22, fontweight="bold",
            ha="center", va="top", path_effects=sk, transform=ax.transData)
    ax.text(W / 2, H - 65, f"Solar Cycle Phase: {label}  |  {nice}",
            color=TEXT_COLOR, fontsize=14, ha="center", va="top",
            path_effects=sk, transform=ax.transData)

    tel        = TELESCOPE_LABELS.get(source, source)
    badge_face = "#f0f4ff" if TEXT_COLOR == "black" else "#0a0a1a"
    badge_edge = "#334466"

    ax.text(W - 20, H - 30, f"Source Name: {tel}",
            color=TEXT_COLOR, fontsize=13, ha="right", va="top", fontweight="bold",
            bbox=dict(facecolor=badge_face, edgecolor=badge_edge,
                      alpha=0.85, boxstyle="round,pad=0.4"),
            transform=ax.transData)
    ax.text(W - 20, H - 75, f"Observations: {end_idx:,}",
            color=TEXT_COLOR, fontsize=11, ha="right", va="top",
            bbox=dict(facecolor=badge_face, edgecolor=badge_edge,
                      alpha=0.85, boxstyle="round,pad=0.3"),
            transform=ax.transData)
    ax.text(W - 1900, H - 30, "Telescope: ORT",
            color=TEXT_COLOR, fontsize=13, ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor=badge_face, edgecolor=badge_edge,
                      alpha=0.85, boxstyle="round,pad=0.4"),
            transform=ax.transData)

    bx, by0, bw, bh = 70, H * 0.28, 25, H * 0.50
    bar_bg = "#ddddee" if TEXT_COLOR == "black" else "#111111"
    ax.add_patch(plt.Rectangle((bx - bw / 2, by0), bw, bh, color=bar_bg, zorder=5))
    if not np.isnan(pct):
        fh = bh * pct / 100.0
        ax.add_patch(plt.Rectangle((bx - bw / 2, by0), bw, fh,
                                    color=plt.cm.plasma(pct / 100.0), zorder=6))
        ax.text(bx + bw + 8, by0 + fh, f"{pct:.1f}%",
                color=TEXT_COLOR, fontsize=10, ha="left", va="center",
                path_effects=sk, transform=ax.transData)
    ax.text(bx, by0 - 20, "SOLAR\nCYCLE",
            color=TEXT_COLOR, fontsize=12, ha="center", va="top",
            path_effects=sk, transform=ax.transData)

    cb_left   = (W - 122) / W
    cb_bottom = 0.28
    cb_width  = 0.014
    cb_height = 0.50
    cax  = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
    cmap = plt.get_cmap(POINTS_CMAP)
    norm = Normalize(vmin=0, vmax=si_max)
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.linspace(0, si_max, 256))
    cb   = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR, labelsize=9)
    cb.outline.set_edgecolor(badge_edge)
    tick_vals = np.linspace(0, si_max, 5)
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([f"{v:.4f}" for v in tick_vals])
    for tl in cb.ax.get_yticklabels():
        tl.set_path_effects(sk)
    fig.text(cb_left - 0.012 + cb_width / 2 + 0.012,
             cb_bottom + cb_height + 0.01,
             "Scintillation\nIndex",
             color=TEXT_COLOR, fontsize=12, ha="center", va="bottom")

    fig.canvas.draw()
    hud = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(H, W, 4)
    plt.close(fig)

    hr         = np.zeros_like(hud)
    hr[:, :, 0] = hud[:, :, 1]; hr[:, :, 1] = hud[:, :, 2]
    hr[:, :, 2] = hud[:, :, 3]; hr[:, :, 3] = hud[:, :, 0]
    base  = frame_arr.astype(np.float32)
    alpha = hr[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(base * (1 - alpha) + hr[:, :, :3].astype(np.float32) * alpha,
                   0, 255).astype(np.uint8)


# ── HUD-only composite task (runs in separate CPU pool) ──────────────────────

def _hud_task(args):
    (path, time_str, source, pct, label,
     end_idx, si_max, frame_idx, total_frames, settle_idx) = args
    from PIL import Image as PILImage
    frame_arr = np.array(PILImage.open(path))
    frame_arr = add_hud(frame_arr, time_str, source, pct, label,
                        end_idx, si_max, frame_idx, total_frames, settle_idx)
    PILImage.fromarray(frame_arr).save(path, format="JPEG", quality=92, subsampling=0)
    return path


# ── 3-D render task (GPU, no HUD) ────────────────────────────────────────────

def render_frame(args):
    (frame_idx, chunk_size,
     xs, ys, zs, sis, times, sources,
     solar_data, min_ssn, max_ssn,
     total_frames, settle_idx, settle_total,
     wire_pts_list, gc_pts_list,
     output_path) = args

    end_idx = (len(xs) if settle_idx >= 0
               else min((frame_idx + 1) * chunk_size, len(xs)))
    if end_idx == 0:
        return output_path

    cur_time    = times[end_idx - 1]
    cur_src     = sources[end_idx - 1]
    sun_vec     = get_sun_vec(cur_time)
    flash_alpha = (1.0 - min(settle_idx / max(settle_total * 0.5, 1), 1.0)
                   if settle_idx >= 0 else 1.0)

    si_arr = np.array(sis[:end_idx])
    si_max = float(np.percentile(si_arr, 95)) if len(si_arr) > 1 else max(float(si_arr[0]), 1e-3)
    si_max = max(si_max, 1e-3)

    pl = pv.Plotter(off_screen=True, window_size=[RENDER_WIDTH, RENDER_HEIGHT])
    pl.set_background(_c(BACKGROUND_COLOR))
    pl.enable_anti_aliasing("ssaa")

    # Globe
    globe = build_globe(sun_vec)
    pl.add_mesh(globe, scalars="colors", rgba=True, opacity=1.0,
                smooth_shading=True, show_edges=False, lighting=False)

    # Wireframe — rebuild from precomputed numpy arrays (no spline recompute)
    if GLOBE_WIRE_ENABLED:
        for pts in wire_pts_list:
            pl.add_mesh(wire_pts_to_mesh(pts),
                        color=_c(GLOBE_WIRE_COLOR),
                        opacity=GLOBE_WIRE_OPACITY,
                        line_width=0.6, lighting=False)

    # Great circles — slice precomputed list, no recompute
    if GREAT_CIRCLE_ENABLED and end_idx > 0:
        start_i = max(0, end_idx - GREAT_CIRCLE_MAX_LINES)
        for i in range(start_i, end_idx):
            pts = gc_pts_list[i]
            if pts is not None:
                pl.add_mesh(gc_pts_to_mesh(pts),
                            color=_c(GREAT_CIRCLE_COLOR),
                            opacity=GREAT_CIRCLE_OPACITY,
                            line_width=GREAT_CIRCLE_LINEWIDTH,
                            lighting=False)

    # IPS points
    ips = build_ips_cloud(xs, ys, zs, sis, end_idx)
    if ips is not None:
        pl.add_mesh(ips, scalars="SI", cmap=POINTS_CMAP, clim=[0, si_max],
                    point_size=POINTS_BASE_SIZE,
                    render_points_as_spheres=True,
                    opacity=POINTS_OPACITY, lighting=False,
                    show_scalar_bar=False)

    # Flash
    flash = build_flash_cloud(xs, ys, zs, end_idx, chunk_size, flash_alpha)
    if flash is not None:
        pl.add_mesh(flash, color=_c(FLASH_COLOR),
                    point_size=int(POINTS_BASE_SIZE * FLASH_SIZE_MULT),
                    render_points_as_spheres=True,
                    opacity=float(flash_alpha) * 0.85, lighting=False)

    # Earth
    earth, earth_tex = build_earth(sun_vec)
    pl.add_mesh(earth, texture=earth_tex, smooth_shading=True,
                lighting=True, ambient=AMBIENT_LIGHT,
                diffuse=DIFFUSE_LIGHT, specular=SPECULAR_LIGHT,
                show_edges=False)

    # Atmosphere
    atmo = pv.Sphere(radius=EARTH_RADIUS * EARTH_ATMOSPHERE_RADIUS_FACTOR,
                     theta_resolution=30, phi_resolution=15)
    pl.add_mesh(atmo, color=_c(EARTH_ATMOSPHERE_COLOR),
                opacity=EARTH_ATMOSPHERE_OPACITY,
                smooth_shading=True, lighting=False)

    # Camera
    pos = camera_for_frame(frame_idx)
    pl.camera.position    = pos
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up          = (0.0, 0.0, 1.0)

    # Lights
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=tuple(sun_vec * 10),
                           focal_point=(0, 0, 0), intensity=1.0))
    pl.add_light(pv.Light(light_type="headlight", intensity=AMBIENT_LIGHT))

    frame_arr = pl.screenshot(None, return_img=True)
    pl.close()

    # Save as JPEG (3-4x faster than PNG for 1080p frames)
    from PIL import Image as PILImage
    PILImage.fromarray(frame_arr).save(
        output_path, format="JPEG", quality=92, subsampling=0)
    return output_path


# ── Worker init ───────────────────────────────────────────────────────────────

def _worker_init(counter, lock):
    os.environ.pop("DISPLAY", None)
    os.environ["PYVISTA_OFF_SCREEN"]                   = "true"
    os.environ["PYVISTA_OFF_SCREEN_RENDERING_BACKEND"] = "egl"
    os.environ["VTK_DEFAULT_OPENGL_WINDOW"]            = "vtkEGLRenderWindow"
    import pyvista as pv
    pv.OFF_SCREEN = True
    with lock:
        idx           = counter.value
        counter.value += 1


def _render_task(args):
    return render_frame(args)


# ── Main animation function ───────────────────────────────────────────────────

def animate_3d_globe(json_file):
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found."); return

    with open(json_file) as f:
        data = json.load(f)

    valid = [d for d in data if d.get("si") is not None and d.get("type") == "source"]
    valid.sort(key=lambda x: x["start_time"])
    print(f"Loaded {len(valid)} valid points.")

    xs, ys, zs, sis, times, sources = [], [], [], [], [], []
    for e in valid:
        ra  = np.deg2rad(e["ra"])
        dec = np.deg2rad(e["dec"])
        R   = GLOBE_RADIUS * POINTS_RADIUS_FACTOR
        xs.append(R * np.cos(dec) * np.cos(ra))
        ys.append(R * np.cos(dec) * np.sin(ra))
        zs.append(R * np.sin(dec))
        sis.append(e["si"])
        times.append(e["start_time"])
        sources.append(e.get("source", "?"))

    chunk_size = ANIMATION_CHUNK_SIZE
    if TOTAL_ANIMATION_DURATION_SEC > 0:
        chunk_size = max(1, int(len(valid) / (TOTAL_ANIMATION_DURATION_SEC * ANIMATION_FPS)))

    total_frames  = (len(valid) // chunk_size) + 1
    settle_frames = int(SETTLE_DURATION_SEC * ANIMATION_FPS)
    grand_total   = total_frames + settle_frames
    print(f"  {total_frames} data + {settle_frames} settle = {grand_total} frames "
          f"({grand_total / ANIMATION_FPS:.1f}s)")

    solar_data       = get_solar_cycle_data()
    min_ssn, max_ssn = compute_cycle_limits(solar_data)

    # ── Precompute static geometry ONCE ──────────────────────────────────────
    print("Precomputing wireframe geometry...")
    wire_pts_list = precompute_wireframe()          # list of (120,3) arrays

    print("Precomputing great circle geometry...")
    gc_pts_list = precompute_great_circles(xs, ys, zs)   # list of (steps,3)|None

    temp_dir = tempfile.mkdtemp(prefix="ips_pv_")
    print(f"Temp dir: {temp_dir}")

    try:
        # Frame paths use .jpg for faster writes
        def fpath(i):
            return os.path.join(temp_dir, f"frame_{i:05d}.jpg")

        tasks = []
        for i in range(total_frames):
            tasks.append((i, chunk_size,
                          xs, ys, zs, sis, times, sources,
                          solar_data, min_ssn, max_ssn,
                          grand_total, -1, settle_frames,
                          wire_pts_list, gc_pts_list,
                          fpath(i)))
        for s in range(settle_frames):
            gf = total_frames + s
            tasks.append((gf, chunk_size,
                          xs, ys, zs, sis, times, sources,
                          solar_data, min_ssn, max_ssn,
                          grand_total, s, settle_frames,
                          wire_pts_list, gc_pts_list,
                          fpath(gf)))

        # GPU saturates around 32 workers; more just causes context-switch overhead
        n_workers = (RENDER_WORKERS if RENDER_WORKERS > 0
                     else min(multiprocessing.cpu_count(), 32))
        print(f"CPU cores available: {multiprocessing.cpu_count()}")
        print(f"Using {n_workers} EGL/GPU workers...")

        ctx     = multiprocessing.get_context("spawn")
        counter = ctx.Value('i', 0)
        lock    = ctx.Lock()
        cs      = max(2, len(tasks) // (n_workers * 2))

        # ── Pass 1: GPU render (no HUD) ───────────────────────────────────
        with ctx.Pool(processes=n_workers,
                      initializer=_worker_init,
                      initargs=(counter, lock)) as pool:
            list(tqdm(
                pool.imap(_render_task, tasks, chunksize=cs),
                total=len(tasks), unit="frame", desc="Rendering (GPU)"
            ))

        # ── Pass 2: HUD composite on all 128 CPU cores ───────────────────
        if HUD_ENABLED:
            print("Compositing HUDs on all CPU cores...")
            hud_tasks = []
            for i in range(total_frames):
                end_idx  = min((i + 1) * chunk_size, len(xs))
                if end_idx == 0: end_idx = 1
                cur_time = times[end_idx - 1]
                cur_src  = sources[end_idx - 1]
                si_arr   = np.array(sis[:end_idx])
                si_max   = float(np.percentile(si_arr, 95)) if len(si_arr) > 1 \
                           else max(float(si_arr[0]), 1e-3)
                si_max   = max(si_max, 1e-3)
                pct, lbl = get_cycle_pct(cur_time, solar_data, min_ssn, max_ssn)
                hud_tasks.append((fpath(i), cur_time, cur_src, pct, lbl,
                                   end_idx, si_max, i, grand_total, -1))

            for s in range(settle_frames):
                gf       = total_frames + s
                end_idx  = len(xs)
                cur_time = times[end_idx - 1]
                cur_src  = sources[end_idx - 1]
                si_arr   = np.array(sis)
                si_max   = float(np.percentile(si_arr, 95)) if len(si_arr) > 1 \
                           else max(float(si_arr[0]), 1e-3)
                si_max   = max(si_max, 1e-3)
                pct, lbl = get_cycle_pct(cur_time, solar_data, min_ssn, max_ssn)
                hud_tasks.append((fpath(gf), cur_time, cur_src, pct, lbl,
                                   end_idx, si_max, gf, grand_total, s))

            hud_cs = max(2, len(hud_tasks) // multiprocessing.cpu_count())
            with ctx.Pool(processes=multiprocessing.cpu_count()) as pool:
                list(tqdm(
                    pool.imap(_hud_task, hud_tasks, chunksize=hud_cs),
                    total=len(hud_tasks), unit="frame", desc="HUD composite"
                ))

        # ── Encode with ffmpeg directly ───────────────────────────────────
        frames = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
        if not frames:
            print("Error: no frames generated."); return

        ff  = imageio_ffmpeg.get_ffmpeg_exe()
        print("Encoding video with ffmpeg...")
        cmd = [
            ff, "-y",
            "-framerate", str(ANIMATION_FPS),
            "-i", os.path.join(temp_dir, "frame_%05d.jpg"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            OUTPUT_FILENAME,
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("ffmpeg error:", result.stderr.decode()[-500:])
        else:
            print(f"Done! → {OUTPUT_FILENAME}")

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Cleanup warning: {e}")


# ── Single-frame poster export ────────────────────────────────────────────────

def export_single_frame(json_file, frame_idx="last", output_path=None):
    if output_path is None:
        output_path = EXPORT_FILENAME
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found."); return

    with open(json_file) as f:
        data = json.load(f)

    valid = [d for d in data if d.get("si") is not None and d.get("type") == "source"]
    valid.sort(key=lambda x: x["start_time"])
    print(f"Loaded {len(valid)} valid points.")

    xs, ys, zs, sis, times, sources = [], [], [], [], [], []
    for e in valid:
        ra  = np.deg2rad(e["ra"])
        dec = np.deg2rad(e["dec"])
        R   = GLOBE_RADIUS * POINTS_RADIUS_FACTOR
        xs.append(R * np.cos(dec) * np.cos(ra))
        ys.append(R * np.cos(dec) * np.sin(ra))
        zs.append(R * np.sin(dec))
        sis.append(e["si"])
        times.append(e["start_time"])
        sources.append(e.get("source", "?"))

    chunk_size   = ANIMATION_CHUNK_SIZE
    total_frames = (len(valid) // chunk_size) + 1
    fi           = (total_frames - 1 if (frame_idx == "last" or frame_idx < 0)
                    else min(frame_idx, total_frames - 1))

    print(f"Exporting frame {fi} of {total_frames} → {output_path}")

    solar_data       = get_solar_cycle_data()
    min_ssn, max_ssn = compute_cycle_limits(solar_data)
    wire_pts_list    = precompute_wireframe()
    gc_pts_list      = precompute_great_circles(xs, ys, zs)

    pv.OFF_SCREEN = True
    args = (fi, chunk_size,
            xs, ys, zs, sis, times, sources,
            solar_data, min_ssn, max_ssn,
            total_frames, -1, 0,
            wire_pts_list, gc_pts_list,
            output_path)
    render_frame(args)

    if HUD_ENABLED:
        end_idx  = min((fi + 1) * chunk_size, len(xs))
        cur_time = times[end_idx - 1]
        cur_src  = sources[end_idx - 1]
        si_arr   = np.array(sis[:end_idx])
        si_max   = float(np.percentile(si_arr, 95)) if len(si_arr) > 1 \
                   else max(float(si_arr[0]), 1e-3)
        pct, lbl = get_cycle_pct(cur_time, solar_data, min_ssn, max_ssn)
        from PIL import Image as PILImage
        frame_arr = np.array(PILImage.open(output_path))
        frame_arr = add_hud(frame_arr, cur_time, cur_src, pct, lbl,
                             end_idx, si_max, fi, total_frames, -1)
        PILImage.fromarray(frame_arr).save(output_path)

    print(f"Done! Poster saved → {output_path} ({RENDER_WIDTH}×{RENDER_HEIGHT})")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if EXPORT_FRAME != -1:
        export_single_frame("ips_continuous_results.json",
                            frame_idx=EXPORT_FRAME,
                            output_path=EXPORT_FILENAME)
    else:
        animate_3d_globe("ips_continuous_results.json")