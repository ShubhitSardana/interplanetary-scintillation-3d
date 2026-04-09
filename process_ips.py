import os
import glob
import gzip
import numpy as np
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import importlib.util
import warnings
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Absolute paths — required for child worker processes
# ------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_GETOUT_PATH = os.path.join(_SCRIPT_DIR, "getout.py")

DATA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "ips-dat-par-May-Dec2011/"))

_month_dirs = [
    ("May",       "May2011-dat",    "May2011-par"),
    # ("June",      "june2011-dat",   "june2011-par"),
    # ("July",      "Jul-2011-dat",   "Jul-2011-par"),
    # ("August",    "Aug-2011-dat",   "Aug-2011-par"),
    # ("September", "Sep-2011-dat",   "Sep-2011-par"),
    # ("October",   "oct-2011-dat",   "oct-2011-par"),
    # ("November",  "Nov2011-dat",    "Nov2011-par"),
    # ("December",  "Dec-2011-dat",   "Dec-2011-par"),
]

DAT_DIRS = [os.path.join(DATA_DIR, m, d) for m, d, _ in _month_dirs if os.path.isdir(os.path.join(DATA_DIR, m, d))]
PAR_DIRS = [os.path.join(DATA_DIR, m, p) for m, _, p in _month_dirs if os.path.isdir(os.path.join(DATA_DIR, m, p))]

# Filter to only existing directories
DAT_DIRS = [d for d in DAT_DIRS if os.path.isdir(d)]
PAR_DIRS = [d for d in PAR_DIRS if os.path.isdir(d)]
print(DAT_DIRS)
print(PAR_DIRS)

SAMPLE_RATE_MS = 20.0
SAMPLING_FREQ  = 1000.0 / SAMPLE_RATE_MS   # 50 Hz

AU_TO_PC = 4.848e-6   # AU → pc

# Number of worker processes.  Leave one core free for the main process.
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)
PLOT_WORKERS = min(64, N_WORKERS)   



# ------------------------------------------------------------
# Light thesis-friendly plot style
# ------------------------------------------------------------
BG        = "white"
PANEL_BG  = "#f7f8fc"
GRID_CLR  = "#d0d4e0"
TEXT_CLR  = "#1a1d2e"
SPINE_CLR = "#8a8fa8"

TS_CLR    = "#1a6faf"
FFT_CLR   = "#c0392b"
PSD_CLR   = "#117a65"
NOISE_CLR = "#e67e22"
FILL_A    = 0.12
PLOT_DPI  = 180

plt.rcParams.update({
    # Font — consistent with publication plot
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":     "stix",

    # Axes
    "axes.facecolor":       PANEL_BG,
    "figure.facecolor":     BG,
    "axes.edgecolor":       SPINE_CLR,
    "axes.labelcolor":      TEXT_CLR,
    "axes.labelsize":       10,
    "axes.titlesize":       11,
    "axes.titlecolor":      TEXT_CLR,
    "axes.titleweight":     "bold",
    "axes.linewidth":       0.8,

    # Ticks — inward on all 4 sides with minors
    "xtick.color":          TEXT_CLR,
    "ytick.color":          TEXT_CLR,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.minor.size":     2,
    "ytick.minor.size":     2,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,

    # Grid
    "grid.color":           GRID_CLR,
    "grid.linewidth":       0.5,
    "grid.linestyle":       "--",
    "grid.alpha":           0.6,

    # Legend
    "legend.facecolor":     "white",
    "legend.edgecolor":     SPINE_CLR,
    "legend.labelcolor":    TEXT_CLR,
    "legend.fontsize":      8.5,
    "legend.framealpha":    0.9,

    # Misc
    "text.color":           TEXT_CLR,
    "figure.dpi":           PLOT_DPI,
    "savefig.facecolor":    BG,
    "savefig.edgecolor":    "none",
    "savefig.dpi":          300,       # high-res PNG output
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
})

# ------------------------------------------------------------
# getout loader — called once per worker process
# ------------------------------------------------------------
def _load_getout():
    spec = importlib.util.spec_from_file_location("getout", _GETOUT_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Worker-process module cache (avoids reloading on every call)
_getout_cache = None

def _getout():
    global _getout_cache
    if _getout_cache is None:
        _getout_cache = _load_getout()
    return _getout_cache


# ------------------------------------------------------------
# PAR file parser
# ------------------------------------------------------------
def parse_par_file(par_path):
    observations = []
    try:
        with gzip.open(par_path, 'rt', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        basename  = os.path.basename(par_path)
        day_str   = basename[2:4]
        month_str = basename[4:7]
        year      = 2011
        months    = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month    = months.get(month_str.lower(), 5)
        day      = int(day_str)
        date_obj = datetime.date(year, month, day)

        for line in lines:
            parts = line.split()
            if not parts or len(parts) < 10:
                continue
            src_name = parts[0]
            if src_name == "01":
                continue
            obs_type = 'cal' if src_name == 'offsour' else 'source'
            try:
                st_h, st_m, st_s = int(parts[7]), int(parts[8]), int(parts[9])
                start_dt = datetime.datetime.combine(
                    date_obj, datetime.time(st_h, st_m, st_s)
                )
                ra_deg, dec_deg = 0.0, 0.0
                if obs_type == 'source':
                    ra_h, ra_m, ra_s    = int(parts[1]), int(parts[2]), int(parts[3])
                    dec_d, dec_m, dec_s = parts[4], int(parts[5]), int(parts[6])
                    ra_deg  = (ra_h + ra_m / 60 + ra_s / 3600) * 15
                    dec_deg = abs(int(dec_d)) + dec_m / 60 + dec_s / 3600
                    if dec_d.startswith('-'):
                        dec_deg = -dec_deg
                duration_min  = float(parts[10])
                end_dt_approx = start_dt + datetime.timedelta(minutes=duration_min)
                observations.append({
                    "source":     src_name,
                    "type":       obs_type,
                    "ra":         ra_deg,
                    "dec":        dec_deg,
                    "start":      start_dt,
                    "end_approx": end_dt_approx,
                })
            except ValueError:
                continue
    except Exception as e:
        print(f"Error parsing {par_path}: {e}")
    return observations


# ------------------------------------------------------------
# DAT loader
# ------------------------------------------------------------
def load_dat_file(dat_path):
    try:
        g = _getout()

        # If gzipped, decompress to a temp file first
        if dat_path.endswith(".gz"):
            import tempfile
            with gzip.open(dat_path, 'rb') as f_in:
                with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f_out:
                    tmp_path = f_out.name
                    f_out.write(f_in.read())
            try:
                hdr, arr = g.decode_dat(tmp_path)
                start_dt = g.try_extract_datetime(hdr)
            finally:
                os.unlink(tmp_path)
        else:
            hdr, arr = g.decode_dat(dat_path)
            start_dt = g.try_extract_datetime(hdr)

        return start_dt, arr
    except Exception as e:
        print(f"Error decoding {dat_path}: {e}")
        return None, None


# ------------------------------------------------------------
# SI computation
# ------------------------------------------------------------
def process_segment(data_chunk):
    if len(data_chunk) < 50:
        return None, None, None

    flux      = np.mean(data_chunk, axis=1) 
    mean_flux = np.mean(flux)                  
    if mean_flux == 0:
        return None, None, None

    rms = np.std(flux)
    if mean_flux == 0 or rms / abs(mean_flux) < 1e-6:
        return None, None, None

    ac = flux - mean_flux
    N  = len(ac)

    window = np.hanning(N)
    W_eff  = np.mean(window ** 2)

    fft_vals  = np.fft.rfft(ac * window)
    raw_power = np.abs(fft_vals) ** 2
    psd       = raw_power / (SAMPLING_FREQ * N * W_eff)
    psd[1:-1] *= 2.0
    freqs     = np.fft.rfftfreq(N, d=1.0 / SAMPLING_FREQ)

    tau = 0.045
    psd *= 1 + (2 * np.pi * freqs * tau) ** 2

    noise_mask  = (freqs >= 8.0) & (freqs <= 10.0)
    noise_level = np.mean(psd[noise_mask]) if np.any(noise_mask) else 0.0

    psd_signal = np.maximum(psd - noise_level, 0.0)

    mask = (freqs >= 0.1) & (freqs <= 6.0)
    if not np.any(mask):
        return None, freqs, psd

    df       = freqs[1] - freqs[0]
    variance = np.sum(psd_signal[mask]) * df
    si2      = variance / mean_flux ** 2
    si       = np.sqrt(si2) if si2 > 0 else 0.0

    return si, freqs, psd


# ------------------------------------------------------------
# Solar wind model
# ------------------------------------------------------------
def solar_elongation(ra_deg, dec_deg, time_dt):
    t   = Time(time_dt)
    src = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    sun = get_sun(t).icrs
    return src.separation(sun).deg


def infer_n0_from_si(si, elong_deg, eps=0.2):
    rmin = np.sin(np.deg2rad(elong_deg))
    if rmin <= 0:
        return np.nan
    return si * rmin ** 1.5 / eps


def dm_from_n0(n0, elong_deg):
    sin_eps = np.sin(np.deg2rad(elong_deg))
    if sin_eps <= 0:
        return np.nan
    return 2.0 * n0 * AU_TO_PC / sin_eps


# ------------------------------------------------------------
# Edge trimming — remove slewing / transition artefacts
# ------------------------------------------------------------
def trim_transition_edges(flux_1d, smooth_win=25, sigma_cut=5.0):
    """
    Detect sharp flux transitions at the start/end of a segment
    caused by telescope slewing.  Returns (trim_start, trim_end)
    indices — the "good" data lives in flux_1d[trim_start:trim_end].

    Algorithm
    ---------
    1. Estimate a robust baseline from the *central* 60 % of the
       segment (median + MAD).
    2. Smooth the flux with a boxcar of width `smooth_win` samples.
    3. Walk inward from each end; declare the edge "good" once the
       smoothed flux stays within `sigma_cut` × σ_MAD of the median
       for at least `smooth_win` consecutive samples.
    """
    N = len(flux_1d)
    if N < 3 * smooth_win:
        return 0, N                       # too short to trim

    # Robust baseline from centre
    c0, c1 = int(0.2 * N), int(0.8 * N)
    centre = flux_1d[c0:c1]
    med    = np.median(centre)
    mad    = np.median(np.abs(centre - med))
    sigma  = mad * 1.4826                  # MAD → Gaussian σ
    if sigma == 0:
        return 0, N
    thr = sigma_cut * sigma

    # Boxcar smooth
    kernel  = np.ones(smooth_win) / smooth_win
    smoothed = np.convolve(flux_1d, kernel, mode='same')

    # --- trim from the END ---
    trim_end = N
    for j in range(N - 1, int(0.7 * N), -1):
        if abs(smoothed[j] - med) < thr:
            trim_end = j + 1
            break

    # --- trim from the START ---
    trim_start = 0
    for j in range(0, int(0.3 * N)):
        if abs(smoothed[j] - med) < thr:
            trim_start = j
            break

    # Sanity: keep at least 50 % of data
    if (trim_end - trim_start) < 0.5 * N:
        return 0, N

    return trim_start, trim_end


# ============================================================
# Per-file worker  —  everything that can run in parallel
# ============================================================

def process_dat_file(dat_path: str):
    """
    Process a single DAT file end-to-end.

    Returns a dict:
        {
          "basename":    str,
          "start_dt":    datetime,
          "obs_list":    [...],
          "flux_full":   np.ndarray  (1-D, channel-averaged),
          "results":     [ per-source result dict, ... ],
          "raw_flux_map": { (source, iso_str): np.ndarray },
        }
    or None on failure.
    """
    basename    = os.path.basename(dat_path)
    # strips both .gz and .dat for files like ips01jun1.dat.gz
    name_no_ext = basename
    for ext in (".dat.gz", ".dat"):
        if name_no_ext.endswith(ext):
            name_no_ext = name_no_ext[:-len(ext)]
            break

    day_part = name_no_ext.removeprefix("ips").removeprefix("NS").removeprefix("dft").removeprefix("spl")

    par_file = None
    for _pdir in PAR_DIRS:
        for _prefix in ("NS", "si"):
            _candidate = os.path.join(_pdir, f"{_prefix}{day_part}.par.gz")
            if os.path.exists(_candidate):
                par_file = _candidate
                break
        if par_file:
            break
    if par_file is None:
        return None

    start_dt, data = load_dat_file(dat_path)
    if start_dt is None:
        return None

    obs_list = parse_par_file(par_file)
    obs_list.sort(key=lambda x: x["start"])

    flux_full = np.mean(data, axis=1).astype(float)

    file_results  = []
    raw_flux_map  = {}

    for i, curr in enumerate(obs_list):
        # Use the source's own scheduled end time; cap at next source's
        # start so segments never overlap.
        if i < len(obs_list) - 1:
            end_dt = min(curr["end_approx"], obs_list[i + 1]["start"])
        else:
            end_dt = curr["end_approx"]

        idx_start = int((curr["start"] - start_dt).total_seconds() * SAMPLING_FREQ)
        idx_end   = int((end_dt        - start_dt).total_seconds() * SAMPLING_FREQ)
        idx_start = max(idx_start, 0)
        idx_end   = min(idx_end, len(data))

        if idx_end <= idx_start + 100:
            continue

        chunk_raw = data[idx_start:idx_end].astype(float)

        # --- adaptive edge trimming ---
        flux_raw = np.mean(chunk_raw, axis=1)
        t0, t1   = trim_transition_edges(flux_raw)
        chunk_v  = chunk_raw[t0:t1]

        si, freqs, psd = process_segment(chunk_v)

        if si is None:
            continue

        elong = solar_elongation(curr["ra"], curr["dec"], curr["start"])
        n0    = infer_n0_from_si(si, elong)
        dm    = dm_from_n0(n0, elong)

        file_results.append({
            "file":       basename,
            "source":     curr["source"],
            "type":       curr["type"],
            "ra":         curr["ra"],
            "dec":        curr["dec"],
            "elong_deg":  elong,
            "start_time": curr["start"].isoformat(),
            "duration_s": (t1 - t0) * SAMPLE_RATE_MS / 1000.0,
            "si":         si,
            "n0_cm3":     n0,
            "dm":         dm,
        })

        flux_1d = np.mean(chunk_v, axis=1)
        key = (curr["source"], curr["start"].isoformat())
        raw_flux_map[key] = flux_1d

    return {
        "basename":    basename,
        "start_dt":    start_dt,
        "obs_list":    obs_list,
        "flux_full":   flux_full,
        "results":     file_results,
        "raw_flux_map": raw_flux_map,
    }


# ============================================================
# Plotting helpers  (unchanged from original)
# ============================================================

def _safe_label(s):
    return str(s).replace("/", "-").replace(" ", "_").replace(":", "")


def _compute_fft_for_plot(flux):
    ac     = flux - np.mean(flux)
    N      = len(ac)
    window = np.hanning(N)
    W_eff  = np.mean(window ** 2)

    fft_vals  = np.fft.rfft(ac * window)
    raw_power = np.abs(fft_vals) ** 2
    freqs     = np.fft.rfftfreq(N, d=1.0 / SAMPLING_FREQ)

    psd       = raw_power / (SAMPLING_FREQ * N * W_eff)
    psd[1:-1] *= 2.0
    tau        = 0.045
    psd       *= 1 + (2 * np.pi * freqs * tau) ** 2

    noise_mask  = (freqs >= 8.0) & (freqs <= 10.0)
    noise_level = np.mean(psd[noise_mask]) if np.any(noise_mask) else 0.0

    amp = np.abs(fft_vals)
    if amp.max() > 0:
        amp = amp / amp.max()

    return freqs, amp, psd, noise_level


def _ann(ax, text, loc="upper right", size=8):
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    ax.text(x, 0.93, text, transform=ax.transAxes,
            fontsize=size, color=TEXT_CLR, va="top", ha=ha,
            family="serif",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.85, edgecolor=SPINE_CLR, linewidth=0.8))

def plot_fullfile_timeseries(flux_full, start_dt, obs_list, basename, out_dir):
    N     = len(flux_full)
    t_hrs = np.arange(N) / SAMPLING_FREQ / 3600.0

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(t_hrs, flux_full, color=TS_CLR, lw=0.5, alpha=0.9)
    ax.set_xlim(t_hrs[0], t_hrs[-1])

    ylo  = np.percentile(flux_full, 0.5)
    yhi  = np.percentile(flux_full, 99.5)
    ypad = (yhi - ylo) * 0.18
    ax.set_ylim(ylo - (yhi - ylo) * 0.05, yhi + ypad)

    colours   = ["#d6eaf8", "#d5f5e3", "#fef9e7", "#f9ebea"]
    total_hrs = t_hrs[-1]

    for k, obs in enumerate(obs_list):
        t0_s = (obs['start']      - start_dt).total_seconds()
        t1_s = (obs['end_approx'] - start_dt).total_seconds()
        t0_s = max(t0_s, 0.0)
        t1_s = min(t1_s, total_hrs * 3600.0)
        if t1_s <= t0_s:
            continue
        t0_h  = t0_s / 3600.0
        t1_h  = t1_s / 3600.0
        mid_h = (t0_h + t1_h) / 2.0
        ax.axvspan(t0_h, t1_h, alpha=0.22,
                   color=colours[k % len(colours)], linewidth=0)
        x_frac = (mid_h - t_hrs[0]) / (t_hrs[-1] - t_hrs[0])
        ax.text(x_frac, 0.97, obs['source'],
                transform=ax.transAxes,
                fontsize=6.5, ha='center', va='top',
                color=TEXT_CLR, rotation=90, alpha=0.75, clip_on=True)

    date_str = start_dt.strftime("%Y-%m-%d")
    ax.set_title(f"Flux Time Series — {date_str}", pad=8)
    ax.set_xlabel("Time  [hours from start]")
    ax.set_ylabel("Flux  [ADC counts]")
    ax.grid(True)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax2 = ax.secondary_xaxis(
        'top',
        functions=(lambda h: h * 60, lambda m: m / 60)
    )
    ax2.set_xlabel("Time  [minutes from start]", fontsize=8, color=SPINE_CLR)
    ax2.tick_params(labelsize=7, colors=SPINE_CLR, direction="in")
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    path = os.path.join(out_dir, f"{_safe_label(basename)}_fullfile_timeseries.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

def _ax_timeseries(ax, t_sec, flux, rec):
    mean_f = np.mean(flux)
    ax.plot(t_sec, flux, color=TS_CLR, lw=0.7, alpha=0.92, label="Flux")
    ax.fill_between(t_sec, flux, mean_f, alpha=FILL_A, color=TS_CLR)
    ax.axhline(mean_f, color=TS_CLR, ls="--", lw=0.9,
               alpha=0.6, label=f"Mean = {mean_f:.1f}")
    ax.set_xlabel("Time  [seconds]")
    ax.set_ylabel("Flux  [ADC counts]")
    ax.set_title(f"Time Series — {rec['source']}", pad=6)
    ax.grid(True)
    ax.legend(loc="lower right")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    info = (f"$m$  = {rec['si']:.4f}\n"
            f"$\\varepsilon$ = {rec['elong_deg']:.1f}°")
    _ann(ax, info)


def _ax_fft(ax, freqs, amp, rec, fmax=12.0):
    mask = freqs <= fmax
    ax.plot(freqs[mask], amp[mask], color=FFT_CLR, lw=0.9, label="FFT amplitude")
    ax.fill_between(freqs[mask], amp[mask], alpha=FILL_A, color=FFT_CLR)
    ax.axvline(0.1, color=SPINE_CLR, ls=":", lw=0.9, alpha=0.7, label="Signal band (0.1–6 Hz)")
    ax.axvline(6.0, color=SPINE_CLR, ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel("Frequency  [Hz]")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(f"FFT Amplitude Spectrum — {rec['source']}", pad=6)
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    info = (f"RA  = {rec['ra']:.2f}°\n"
            f"Dec = {rec['dec']:.2f}°\n"
            f"$\\Delta t$ = {rec['duration_s']:.0f} s")
    _ann(ax, info)

def _ax_psd(ax, freqs, psd, noise_level, rec, fmax=12.0):
    mask = (freqs > 0) & (freqs <= fmax) & (psd > 0)
    f_m  = freqs[mask]
    p_m  = psd[mask]
    p_floor = p_m.min() * 0.1          # clean log-axis fill base
    ax.semilogy(f_m, p_m, color=PSD_CLR, lw=0.9, label="PSD")
    ax.fill_between(f_m, p_m, p_floor, alpha=FILL_A, color=PSD_CLR)
    ax.axvspan(0.1, 6.0, alpha=0.07, color=PSD_CLR,
               label="Integration band (0.1–6 Hz)")
    if noise_level > 0:
        ax.axhline(noise_level, color=NOISE_CLR, ls="--", lw=1.1,
                   alpha=0.85, label=f"Noise floor = {noise_level:.2e}")
    ax.axvline(0.1, color=SPINE_CLR, ls=":", lw=0.9, alpha=0.7)
    ax.axvline(6.0, color=SPINE_CLR, ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel("Frequency  [Hz]")
    ax.set_ylabel(r"PSD  [ADC$^2$ Hz$^{-1}$]")
    ax.set_title(f"Power Spectral Density — {rec['source']}", pad=6)
    ax.grid(True, which="both", color=GRID_CLR, linewidth=0.5,
            linestyle="--", alpha=0.6)
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(loc="upper right")
    _ann(ax, r"RC $\tau$ = 45 ms corrected", loc="upper left")

def _decorate(fig, rec, combined=False):
    src   = rec.get("source", "?")
    dt    = rec.get("start_time", "")
    si    = rec.get("si", float("nan"))
    elong = rec.get("elong_deg", float("nan"))
    title = (f"ORT · IPS Observation · {src} · {dt}"
             if combined else f"{src} · {dt}")
    fig.suptitle(title, fontsize=11 if combined else 10,
                 fontweight="bold", color=TEXT_CLR, y=0.98)
    footer = (f"$m$ = {si:.4f}  |  "
              f"$\\varepsilon$ = {elong:.1f}°  |  "
              f"RA = {rec.get('ra', 0):.2f}°  "
              f"Dec = {rec.get('dec', 0):.2f}°  |  "
              f"$\\Delta t$ = {rec.get('duration_s', 0):.0f} s")
    fig.text(0.5, 0.01, footer, ha="center", va="bottom",
             fontsize=7.5, color=SPINE_CLR, style="italic")

# ============================================================
# Plotting helpers  — PATCHED: parallel + no duplicate FFT
# ============================================================

def _plot_worker(args):
    """
    Unchanged signature for compatibility; now skips 'combined' by default
    and reuses a single FFT pass.
    """
    rec, flux_arr, out_dirs, skip_combined = args   # <-- added skip_combined

    if flux_arr is None or len(flux_arr) < 50:
        return f"SKIP  {rec['source']} — too few samples"

    flux  = np.asarray(flux_arr, dtype=float)
    N     = len(flux)
    t_sec = np.arange(N) / SAMPLING_FREQ

    # Single FFT pass — reused by all three sub-plots
    freqs, amp, psd, noise_level = _compute_fft_for_plot(flux)

    src    = _safe_label(rec["source"])
    dt     = rec.get("start_time", "unknown")
    dt_str = (dt.replace(":", "").replace("-", "").replace(" T ", "_")[:13]
              if isinstance(dt, str) else "unknown")
    stem   = f"{src}_{dt_str}"

    def _save(fig, subdir, suffix):
        path = os.path.join(out_dirs[subdir], f"{stem}_{suffix}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    _ax_timeseries(ax, t_sec, flux, rec)
    _decorate(fig, rec)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, "timeseries", "timeseries")

    fig, ax = plt.subplots(figsize=(8, 4))
    _ax_fft(ax, freqs, amp, rec)
    _decorate(fig, rec)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, "fft", "fft")

    fig, ax = plt.subplots(figsize=(8, 4))
    _ax_psd(ax, freqs, psd, noise_level, rec)
    _decorate(fig, rec)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, "psd", "psd")

    if not skip_combined:
        fig = plt.figure(figsize=(13, 10))
        gs  = GridSpec(3, 1, figure=fig, hspace=0.50,
                       left=0.08, right=0.97, top=0.92, bottom=0.07)
        _ax_timeseries(fig.add_subplot(gs[0]), t_sec, flux, rec)
        _ax_fft(fig.add_subplot(gs[1]),        freqs, amp, rec)
        _ax_psd(fig.add_subplot(gs[2]),        freqs, psd, noise_level, rec)
        _decorate(fig, rec, combined=True)
        _save(fig, "combined", "combined")

    return f"OK  {src}"


def _fullfile_plot_worker(args):
    flux_full, start_dt, obs_list, basename, out_dir = args
    try:
        flux_full = np.asarray(flux_full, dtype=float)   # add this
        plot_fullfile_timeseries(flux_full, start_dt, obs_list, basename, out_dir)
        return f"OK  {basename}"
    except Exception as e:
        return f"ERR  {basename}: {e}"

def render_all_plots(results, raw_flux_map, fullfile_map,
                     out_root="plots", skip_combined=True):
    subdirs = {k: os.path.join(out_root, k)
               for k in ("timeseries", "fft", "psd", "combined", "fullfile")}
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    # ── Full-file time-series — now PARALLEL ────────────────────────────────
    print(f"\n[plots] Rendering {len(fullfile_map)} full-file time-series ...")
    fullfile_args = [
        (flux_full.tolist(), start_dt, obs_list, basename, subdirs["fullfile"])
        for basename, (flux_full, start_dt, obs_list) in fullfile_map.items()
    ]

    with ProcessPoolExecutor(max_workers=PLOT_WORKERS) as pool:
        futures = {pool.submit(_fullfile_plot_worker, a): a[3] for a in fullfile_args}
        with tqdm(total=len(futures), desc="Full-file plots") as pbar:
            for fut in as_completed(futures):
                pbar.update(1)
                try:
                    msg = fut.result()
                    if msg.startswith("ERR"):
                        tqdm.write(f"  {msg}")
                except Exception as exc:
                    tqdm.write(f"  Fullfile plot error: {exc}")

    # ── Per-source plots — PARALLEL ──────────────────────────────────────────
    plot_args = []
    for rec in results:
        key  = (rec["source"],
                rec["start_time"] if isinstance(rec["start_time"], str)
                else rec["start_time"].isoformat())
        flux = raw_flux_map.get(key)
        flux_list = flux.tolist() if flux is not None else None
        plot_args.append((rec, flux_list, subdirs, skip_combined))

    print(f"[plots] Rendering {len(results)} per-source plots "
          f"({PLOT_WORKERS} workers, skip_combined={skip_combined}) ...")

    with ProcessPoolExecutor(max_workers=PLOT_WORKERS) as pool:
        futures = {pool.submit(_plot_worker, a): a[0]["source"]
                   for a in plot_args}
        with tqdm(total=len(futures), desc="Per-source plots") as pbar:
            for fut in as_completed(futures):
                pbar.update(1)
                try:
                    msg = fut.result()
                    if msg.startswith("SKIP"):
                        tqdm.write(f"  {msg}")
                except Exception as exc:
                    tqdm.write(f"  Plot error ({futures[fut]}): {exc}")

    print(f"[plots] Done — saved under '{out_root}/'")

# ============================================================
# Helpers
# ============================================================

def default_serializer(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return str(obj)


# ============================================================
# Main  —  parallel processing, serial plotting
# ============================================================
if __name__ == "__main__":

    dat_files = sorted(
        f for d in DAT_DIRS
        for pattern in ("ips*.dat", "ips*.dat.gz")
        for f in glob.glob(os.path.join(d, pattern))
    )
    print(f"Found {len(dat_files)} data files.")
    # DEBUG — print first few from each dir
    for d in DAT_DIRS:
        files = sorted(glob.glob(os.path.join(d, "ips*.dat")) + glob.glob(os.path.join(d, "ips*.dat.gz")))
        print(f"  {d}: {len(files)} files, e.g. {os.path.basename(files[0]) if files else 'NONE'}")

    # Also check what PAR files look like
    for d in PAR_DIRS:
        files = sorted(glob.glob(os.path.join(d, "*.par.gz")))
        print(f"  {d}: {len(files)} par files, e.g. {os.path.basename(files[0]) if files else 'NONE'}")

    print(f"Found {len(dat_files)} data files.")
    print(f"Using {N_WORKERS} worker processes.")

    results      = []
    raw_flux_map = {}
    fullfile_map = {}

    # ── Parallel processing ──────────────────────────────────────────────────
    # Each DAT file is processed in a separate worker process.
    # ProcessPoolExecutor uses 'spawn' on Windows and 'fork' on Linux/macOS,
    # both safe here because no shared mutable state is used.
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_dat_file, f): f for f in dat_files}

        with tqdm(total=len(futures), desc="Processing DAT files") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    res = future.result()
                except Exception as exc:
                    print(f"  Worker error ({futures[future]}): {exc}")
                    continue

                if res is None:
                    continue

                fullfile_map[res["basename"]] = (
                    res["flux_full"],
                    res["start_dt"],
                    res["obs_list"],
                )
                results.extend(res["results"])
                raw_flux_map.update(res["raw_flux_map"])

    print(f"\nProcessed {len(dat_files)} files — {len(results)} source scans found.")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = os.path.join(_SCRIPT_DIR, "ips_continuous_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, default=default_serializer, indent=2)
    print(f"Saved {len(results)} records to {out_json}")

    # ── Generate plots (serial — matplotlib is not fork-safe) ───────────────
    render_all_plots(results, raw_flux_map, fullfile_map,
                 out_root="plots", skip_combined=False)  # <-- was True