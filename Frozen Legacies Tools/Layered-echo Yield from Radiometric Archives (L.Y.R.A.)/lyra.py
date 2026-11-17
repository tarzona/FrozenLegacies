"""
lyra.py — LYRA: Layered-echo Yield from Radiometric Archives
=============================================================

Algorithmic A-scope waveform extraction from raw TIFF film scans.
Named following the existing tool constellation convention:
  URSA (bear), ASTRA (stars), TERRA (earth), ARIES (ram), LYRA (lyre).

For each A-scope frame in a TIFF this module:
  1.  Detects frame boundaries (white inter-frame gaps)
  2.  Extracts the oscilloscope trace (darkest pixel per column)
  3.  Locates the main bang (origin of the time axis)
  4.  Estimates per-frame noise floor from pre-main-bang baseline
  5.  Finds surface and bed echo peaks via prominence-based detection
  6.  Walks leading / trailing edges at +5 dB and +10 dB above noise floor
  7.  Computes waveform shape metrics:
        peakiness        = peak_linear / mean_linear  (within +10 dB window)
        asymmetry        = trailing_width_10 / leading_width_10
        trailing_tail_us = trail_5 - trail_10  (subsurface scattering length)
        leading_rise_us  = peak_twt - lead_10  (onset steepness)
  8.  Returns a flat pandas DataFrame (one row per frame)

Grid calibration (validated vs Neal 1977 Fig 1.3a, CBD 0465, F125;
  corrected 2026-03-10: true sweep is 2.0 us/div, not 1.5):
  X-axis : 2.0 µs / major division  (205.54 px/major; 4 minor ticks -> 0.4 µs/minor)
  Y-axis : 10 dB  / major division  (205.0  px/major)
  Noise floor reference : y = 1507 px  -> -60 dB

WARNING: Do NOT use ASTRA surface_us / bed_us for geometry — those times are
         1.5× too large (ASTRA assumed 3.0 µs/major; correct value is 2.0 µs/major).
         Use only LYRA-derived travel times.

Physical interpretation of waveform metrics:
  High peakiness + low asymmetry + short trailing_tail  -> specular / liquid water
  Low peak power + long trailing_tail                   -> saline / frozen marine ice
  Broad symmetric echo, moderate peakiness              -> rough / incoherent bed

Flight processing:
    python lyra.py --tiff "path\\to\\TIFFs" --output "path\\to\\output" --nav "path\\to\\navigation"

Run an embedded standalone utility:
    python lyra.py --tool pick_calibration [utility arguments]

Contact: Angelo Tarzona {tarzona@gatech.edu}
"""
from __future__ import annotations

import re
import os
import warnings
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ASTRA_X_POINTS = 8
ASTRA_Y_POINTS = 5
ASTRA_DEFAULT_CROP_START_Y = 1000
ASTRA_DEFAULT_CROP_END_Y = 1800


def astra_pick_signature(pick: dict) -> str:
    """Fingerprint the approved inputs that determine frame calibration."""
    data = {key: pick.get(key) for key in
            ("x_ref", "ref", "x_grid", "y_grid", "montage_approved",
             "exclude", "exclude_reason")}
    for key in ("crop_start_y", "crop_end_y"):
        if key in pick:
            data[key] = pick[key]
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def astra_trace(frame: np.ndarray,
                crop_start: int = ASTRA_DEFAULT_CROP_START_Y,
                crop_end: int = ASTRA_DEFAULT_CROP_END_Y,
                smoothed: bool = False) -> np.ndarray:
    """Return ASTRA's first thresholded dark pixel per column in image rows."""
    from scipy.ndimage import gaussian_filter

    image = np.asarray(frame, dtype=np.float64)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2 or not 0 <= crop_start < min(crop_end, image.shape[0]):
        raise ValueError("ASTRA trace crop is empty or the image is not grayscale")
    crop = image[crop_start:min(crop_end, image.shape[0]), :]
    filtered = crop if smoothed else gaussian_filter(crop, sigma=4)
    dark = filtered < np.percentile(filtered, 10)
    valid = dark.any(axis=0)
    trace = np.full(image.shape[1], np.nan)
    trace[valid] = np.argmax(dark[:, valid], axis=0) + crop_start
    return trace


def astra_trace_with_fallback(frame: np.ndarray, fallback: np.ndarray,
                              context: str,
                              crop_start_y: int = ASTRA_DEFAULT_CROP_START_Y,
                              crop_end_y: int = ASTRA_DEFAULT_CROP_END_Y) -> np.ndarray:
    """Fill missing ASTRA columns from the LYRA trace and report the fallback."""
    trace = astra_trace(frame, crop_start_y, crop_end_y)
    if trace.shape != fallback.shape:
        raise ValueError("ASTRA and LYRA traces have different widths")
    missing = ~np.isfinite(trace)
    if np.any(missing):
        warnings.warn(
            f"{context}: ASTRA red trace missing in {int(missing.sum())} columns; "
            "using the LYRA trace for those columns.",
            RuntimeWarning,
        )
        trace[missing] = fallback[missing]
    return trace


def astra_peak_row(trace: np.ndarray, x: int, y_grid: list,
                   context: str) -> float:
    """Read the selected column and flag a possible graticule intersection."""
    if not 0 <= x < len(trace) or not np.isfinite(trace[x]):
        raise ValueError(f"{context}: selected column has no valid trace point")
    row = float(trace[x])
    if any(abs(row - float(grid_y)) <= 3 for grid_y in y_grid):
        warnings.warn(f"{context}: red trace at x={x} intersects an approved Y grid "
                      "line; inspect this peak for a graticule artifact.", RuntimeWarning)
    return row


# -- TIFF canonical naming utility ---------------------------------------------

_CANONICAL_RE = re.compile(r'^\d+_\d+_\d+-reel_begin_end\.tiff$', re.IGNORECASE)
_FLIGHT_TIFF_RE = re.compile(
    r'^F(?P<flight>\d+)-C(?P<first>\d+)_C(?P<last>\d+)\.tiff$',
    re.IGNORECASE,
)


def parse_tiff_filename(tiff_path: Path) -> tuple[int, int, int] | None:
    """Return (flight, first CBD, last CBD) for the current TIFF naming format."""
    match = _FLIGHT_TIFF_RE.match(Path(tiff_path).name)
    if not match:
        return None
    flight, first, last = (int(match.group(key)) for key in ("flight", "first", "last"))
    if first > last:
        raise ValueError(f"TIFF CBD range is reversed: {tiff_path.name}")
    return flight, first, last


def output_root(default_root: Path | None = None) -> Path:
    """Return the configured output root, or the project output directory."""
    fallback = default_root or Path(__file__).resolve().parent.parent / "output"
    return Path(os.environ.get("LYRA_OUTPUT_DIR", str(fallback))).expanduser()


def display_path(path: Path, root: Path) -> Path:
    """Display paths relative to the project when possible, or as given."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def navigation_root(default_root: Path | None = None) -> Path:
    """Return the configured navigation directory."""
    fallback = default_root or Path(
        r"C:\Users\dtarzona3\GaTech Dropbox\CoS-EXT\EAS-Ext"
        r"\EAS-Chu-TEAM-Ext\ArchivalData\SPRI\SPRI_Navigation"
        r"\RadarfilmStudio"
    )
    return Path(os.environ.get("LYRA_NAV_DIR", str(fallback))).expanduser()


def _parse_rename_log(log_path: Path) -> dict[str, str]:
    """Parse a LYRA rename log. Returns {current_name: canonical_name} mapping.

    The log format has blocks like:
        Old: 43_0008375_0008399-reel_begin_end.tiff   <- canonical (target)
        New: F141-C0038_C0047.tiff                    <- current (on-disk) name
    """
    mapping: dict[str, str] = {}
    old_name = new_name = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("Old: "):
            old_name = line[5:].strip()
        elif line.startswith("New: "):
            new_name = line[5:].strip()
            if old_name and new_name:
                mapping[new_name] = old_name
            old_name = new_name = None
    return mapping


def ensure_canonical_name(tiff_path: Path) -> Path:
    """Return a supported TIFF path, converting only legacy rename-log names.

    Current format: ``F<flight>-C<first>_C<last>.tiff``.
    Legacy canonical format: ``<reel>_<start>_<end>-reel_begin_end.tiff``.

    Current and legacy canonical names are returned unchanged. Other names may
    still be resolved through a legacy rename log.
    """
    if _CANONICAL_RE.match(tiff_path.name) or parse_tiff_filename(tiff_path):
        return tiff_path  # already canonical — nothing to do

    log_files = sorted(tiff_path.parent.glob("*_rename_log*.txt"))
    if not log_files:
        raise FileNotFoundError(
            f"TIFF '{tiff_path.name}' is not in canonical format and no "
            f"'*_rename_log*.txt' was found in {tiff_path.parent}.\n"
            f"  Canonical format: {{reel}}_{{start}}_{{end}}-reel_begin_end.tiff"
        )

    mapping = _parse_rename_log(log_files[0])
    canonical = mapping.get(tiff_path.name)
    if canonical is None:
        raise KeyError(
            f"'{tiff_path.name}' not found in rename log '{log_files[0].name}'.\n"
            f"  Log covers: {sorted(mapping.keys())[:5]} …"
        )

    new_path = tiff_path.parent / canonical
    if new_path.exists():
        print(f"  [ensure_canonical_name] Already renamed: {new_path.name}")
        return new_path

    tiff_path.rename(new_path)
    print(f"  [ensure_canonical_name] Renamed:\n"
          f"      {tiff_path.name}\n"
          f"    -> {new_path.name}")
    return new_path


def tiff_id(tiff_path: Path) -> str:
    """Extract the TIFF's starting CBD or legacy numeric start identifier.

    ``F103-C1024_C1036.tiff`` -> ``"1024"``. Legacy names retain the former
    numeric start extraction.
    """
    metadata = parse_tiff_filename(tiff_path)
    if metadata:
        return str(metadata[1])
    for part in tiff_path.stem.split("_"):
        if part.isdigit() and len(part) >= 5:
            return str(int(part[-8:]))[-4:]
    return tiff_path.stem[:8]  # fallback


def resolve_tiff_arg(arg: str, root: Path) -> Path:
    """Resolve a TIFF argument to a full path.

    Accepted formats (in order of preference):
      - ``FLT/TIFF_ID``:  ``125/8700``  (recommended shorthand)
      - Full or relative path to a TIFF file

    Returns the resolved absolute Path.  Raises SystemExit on no match.
    """
    # If it has a file extension -> treat as a literal path
    p = Path(arg)
    if p.suffix:
        return root / p if not p.is_absolute() else p

    # FLT/TIFF_ID shorthand: "125/8700" -> search in tiff/125/
    if "/" in arg:
        parts = arg.strip("/").split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            configured_dir = os.environ.get("LYRA_TIFF_DIR")
            flt_dir = (
                Path(configured_dir).expanduser()
                if configured_dir else root / "tiff" / parts[0]
            )
            if not flt_dir.exists() and not configured_dir:
                flt_dir = root / "tiff"  # legacy single-flight flat directory
            os.environ["LYRA_FLIGHT"] = parts[0]
            if not flt_dir.exists():
                raise SystemExit(
                    f"ERROR: Flight directory not found: {flt_dir}")
            matches = [
                candidate for candidate in flt_dir.glob("*.tiff")
                if (metadata := parse_tiff_filename(candidate)) is not None
                and metadata[0] == int(parts[0])
                and metadata[1] == int(parts[1])
            ]
            if not matches:
                padded = parts[1].zfill(7)
                matches = sorted(flt_dir.glob(f"*_{padded}_*-reel_begin_end.tiff"))
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise SystemExit(
                    f"ERROR: No TIFF matching ID '{parts[1]}' in F{parts[0]}.\n"
                    f"  Searched: {flt_dir}/")
            # Multiple matches (unlikely within one flight)
            raise SystemExit(
                f"ERROR: Multiple TIFFs match ID '{parts[1]}' in F{parts[0]}.")
        # Otherwise treat as a regular relative path
        return root / p if not p.is_absolute() else p

    # If it exists as a relative path from root -> use it
    if (root / p).exists():
        return root / p

    # Bare numeric ID without flight -> reject with helpful message
    if arg.isdigit():
        raise SystemExit(
            f"ERROR: Please use FLT/TIFF_ID format, e.g. 125/{arg}\n"
            f"  Usage: python codes/<script>.py 125/{arg}")

    raise SystemExit(
        f"ERROR: Cannot resolve TIFF '{arg}'.\n"
        f"  Usage: python codes/<script>.py FLT/TIFF_ID  (e.g. 125/8700)"
    )


# -- Pure-numpy replacements (no scipy dependency) -----------------------------

def _connected_components_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each run of True in a boolean mask."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask.tolist()):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def _find_peaks_numpy(arr: np.ndarray,
                      prominence: float = 0,
                      distance: int = 1) -> np.ndarray:
    """
    Find local minima in arr with minimum prominence and minimum distance.
    Equivalent (for our use case) to scipy.signal.find_peaks(-arr, ...).

    prominence : minimum depth of the minimum below surrounding neighbours
    distance   : minimum gap (pixels) between returned minima
    """
    n = len(arr)
    # Strict local minima
    candidates = [i for i in range(1, n - 1)
                  if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]]
    if not candidates:
        return np.array([], dtype=int)
    candidates = np.array(candidates, dtype=int)

    # Distance filter: greedy — keep most prominent, suppress neighbours
    if distance > 1 and len(candidates) > 1:
        order = np.argsort(arr[candidates])   # most-prominent first
        kept: list[int] = []
        used: set[int] = set()
        for idx in order:
            p = int(candidates[idx])
            if p not in used:
                kept.append(p)
                for near in candidates:
                    if abs(int(near) - p) < distance:
                        used.add(int(near))
        candidates = np.array(sorted(kept), dtype=int)

    # Prominence filter
    if prominence > 0 and len(candidates) > 0:
        valid = []
        for p in candidates:
            d     = max(int(distance) * 3, 30)
            lo, hi = max(0, p - d), min(n - 1, p + d)
            lmax  = arr[lo:p].max()   if p   > lo else arr[p]
            rmax  = arr[p + 1:hi + 1].max() if hi > p else arr[p]
            if (min(lmax, rmax) - arr[p]) >= prominence:
                valid.append(p)
        candidates = np.array(valid, dtype=int)

    return candidates

# -- Physical constants ---------------------------------------------------------
C_AIR_M_PER_US = 300.0   # m / µs  (speed of light in air)
C_ICE_M_PER_US = 168.0   # m / µs  (Millar 1981; Neal 1977 uses 169)

# -- Default grid calibration ---------------------------------------------------
# Measured algorithmically from raw TIFFs; validated vs Neal 1977 Fig 1.3a.
_X_SPACING_PX = 205.54   # pixels per major X division (measured from TIFF)
_Y_SPACING_PX = 205.0    # pixels per major Y division (user confirmed)

DEFAULT_CAL: dict = dict(
    us_per_px     = 2.0 / _X_SPACING_PX,   # 0.009732 µs/px   <- corrected 2026-03-10
    db_per_px     = 10.0 / _Y_SPACING_PX,  # 0.048780 dB/px
    y_ref_px      = 1507,    # row of -60 dB reference line
    y_ref_db      = -60.0,   # dB at the reference row
    y_disp_lo     = 300,     # top of usable display band (row index; lower = higher on image)
    y_disp_hi     = 1700,    # bottom of usable display band
    mb_x_guess    = 800,     # expected main bang column (frame-relative; heuristic)
    gauss_sigma   = 3,       # trace-smoothing Gaussian sigma (pixels)
    consec        = 5,       # consecutive threshold crossings to confirm trailing edge
    peak_min_prom_dB = 5.0,  # minimum peak prominence above noise floor for detection (dB)
    peak_min_dist_px = 80,   # minimum inter-peak distance (pixels)
    mb_skip_us    = 2.0,     # skip this many µs after main bang before searching for echoes
    min_bed_gap_us = 1.5,    # minimum surface->bed TWT gap (µs); ~126 m minimum ice thickness
    max_walk_us   = 17.0,    # rightward envelope-walk limit (µs from MB); covers all RIS depths
    graticule_mask_half_px = 5,  # ±px to suppress around each major graticule line
    signal_binary_thresh  = 0.30, # normalized pixel threshold for dark-pixel binary mask.
                                  # CRT trace ~ 0.0–0.15; background ~ 0.55–0.65.
    signal_density_sigma  = 30,   # Gaussian sigma for smoothing per-column dark-pixel count.
    signal_density_thresh = 50,   # smoothed dark-pixel count threshold for signal detection.
                                  # Used as the CAP in the adaptive threshold formula:
                                  #   thresh = max(floor, min(cap, fraction × peak_density))
                                  # For bright frames (peak ~200): thresh = 50 (same as original).
                                  # For faint frames (peak ~40): thresh = 12 (floor).
    signal_density_floor  = 12,   # minimum adaptive threshold; above film-grain noise (~5-6 px/col).
    signal_density_frac   = 0.25, # fraction of peak column density used for adaptive threshold.
    signal_grat_half      = 8,    # ±px to exclude around graticule rows in signal detection
                                  # (wider than graticule_mask_half_px to avoid dark-line leakage).
    surface_window_us   = 8.0,   # surface search window end (µs from MB); covers altitude <=1200 m
    min_surface_lead_dB = 15.0,  # if strongest surface cand. > first by this, promote it
    quiet_gap_lead_us   = 1.0,   # µs after surface echo before checking for quiet gap
    quiet_gap_dB        = 12.0,  # trace must return within this many dB of NF in the gap
                                 # before a bed candidate is accepted.  Rejects surface-echo
                                 # trailing oscillations (which are continuous, never return
                                 # to NF); passes genuine thin-ice beds (surface echo decays
                                 # to NF within <=1 µs of the surface peak, then quiet gap).
                                 # Not applied when gap < 1 µs wide (very thin ice, <=84 m).
    min_bed_width_us    = 0.2,   # minimum bed echo width (µs) at NF + min_bed_width_dB.
                                 # Rejects very narrow film artifacts / spikes (< 0.15 µs).
                                 # Thin-ice bed echoes can be as narrow as 0.25 µs (F126);
                                 # thick-ice bed echoes >= 0.45 µs (F125 TIFF 7700 CBD0817).
    min_bed_width_dB    = 10.0,  # dB above NF used for the width measurement.
)


# -- Data classes ---------------------------------------------------------------

@dataclass
class EchoResult:
    """Waveform extraction result for one echo (surface or bed)."""
    # Peak location
    peak_x:        int   = 0      # column (frame-relative pixels)
    peak_y:        int   = 0      # row (pixels; lower = higher power)
    peak_twt_us:   float = np.nan # two-way travel time from main bang (µs)
    peak_power_dB: float = np.nan # power (dB, calibration reference frame)
    peak_snr_dB:   float = np.nan # SNR above measured noise floor (dB)

    # Envelope at +10 dB above noise floor
    lead_10_us:  float = np.nan   # leading edge TWT (µs from MB)
    trail_10_us: float = np.nan   # trailing edge TWT (µs from MB)
    width_10_us: float = np.nan   # width = trail_10 - lead_10

    # Envelope at +5 dB above noise floor (wider, captures subsurface tail)
    lead_5_us:   float = np.nan
    trail_5_us:  float = np.nan
    width_5_us:  float = np.nan

    # First-break onset: where signal first rises above NF + onset_db threshold
    # Use onset_twt_us for ice thickness (shortest travel path = nadir reflection)
    # Use peak_twt_us for reflectivity R0 (maximum returned power)
    onset_twt_us: float = np.nan     # TWT at first break (us from MB)
    onset_power_dB: float = np.nan   # power at onset point (dB)

    # Derived shape metrics (NF-referenced, legacy)
    leading_rise_us:  float = np.nan  # peak_twt - lead_10  (onset steepness; lower -> more specular)
    trailing_tail_us: float = np.nan  # trail_5 - trail_10  (subsurface scattering tail length)
    peakiness:        float = np.nan  # peak_linear / mean_linear within +10 dB window (>1 = peaked)
    asymmetry:        float = np.nan  # (peak_twt - lead_10) / (trail_10 - peak_twt) (>1 = trailing-heavy)

    # Peak-relative shape metrics (SNR-independent)
    # These measure echo decay rate relative to own peak, not noise floor.
    # Directly encode interface sharpness vs gradational structure.
    decay_10_us:    float = np.nan  # time from peak to peak-10 dB on trailing side
    decay_20_us:    float = np.nan  # time from peak to peak-20 dB on trailing side
    rise_10_us:     float = np.nan  # time from peak-10 dB to peak on leading side
    pk_asymmetry:   float = np.nan  # decay_10 / rise_10 (>1 = trailing-heavy)


@dataclass
class LyraFrame:
    """Complete LYRA result for one A-scope frame."""
    frame_idx:   int = 0
    frame_left:  int = 0   # left pixel column in original TIFF
    frame_right: int = 0   # right pixel column in original TIFF
    frame_w:     int = 0
    mb_x:        int = 0   # main bang column (frame-relative)
    noise_floor_dB: float = -60.0
    surface: EchoResult = field(default_factory=EchoResult)
    bed:     EchoResult = field(default_factory=EchoResult)
    surface_detected: bool = False
    bed_detected:     bool = False
    quality:          str  = "ok"  # "ok" | "no_surface" | "no_bed" | "failed"


# -- Unit conversions -----------------------------------------------------------

def px_to_db(y, cal: dict):
    """Row pixel -> power (dB). Lower y (higher on image) = higher power."""
    return cal["y_ref_db"] + (cal["y_ref_px"] - y) * cal["db_per_px"]


def db_to_px(db: float, cal: dict) -> float:
    """Power (dB) -> row pixel."""
    return cal["y_ref_px"] - (db - cal["y_ref_db"]) / cal["db_per_px"]


def px_to_us(x, mb_x: int, cal: dict):
    """Column pixel (frame-relative) -> TWT (µs from main bang)."""
    return (x - mb_x) * cal["us_per_px"]


def us_to_px(us: float, mb_x: int, cal: dict) -> int:
    """TWT (µs from main bang) -> column pixel (frame-relative)."""
    return int(round(mb_x + us / cal["us_per_px"]))


# -- Gaussian smoothing ---------------------------------------------------------

def _gauss_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    r = int(4 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(arr.astype(float), k, mode="same")


def _running_median(arr: np.ndarray, window: int = 11) -> np.ndarray:
    """Running median filter (rejects sporadic outliers, preserves real echoes)."""
    out = arr.astype(float).copy()
    hw  = window // 2
    n   = len(arr)
    for i in range(hw, n - hw):
        out[i] = np.median(arr[i - hw : i + hw + 1])
    return out


# -- Frame boundary detection ---------------------------------------------------

def detect_frames(img_norm: np.ndarray,
                  band_frac: tuple = (0.20, 0.80),
                  smooth_sigma: float = 20,
                  gap_threshold: float = 0.90,
                  min_gap_px: int = 80,
                  min_frame_px: int = 200,
                  buffer_px: int = 5,
                  expected_frames: int = 14) -> list[tuple[int, int]]:
    """
    Detect A-scope frame boundaries from inter-frame white gaps.

    Ports the working approach from extract_noise_floor_ursa_frames.py:
      1. Compute column means in central band (20%–80% of height)
      2. Gaussian smooth
      3. Normalize to [0, 1]           <- CRITICAL: do NOT threshold raw values
      4. Threshold normalized at 0.90 -> gap mask
      5. Find gap CENTERS (mean col of each run >= min_gap_px wide)
      6. Frames = intervals between gap centers, with buffer_px inward

    Parameters
    ----------
    img_norm        : (H, W) float array, values in [0, 1]
    band_frac       : (lo, hi) fractional row range for gap analysis
    smooth_sigma    : Gaussian sigma for column-mean smoothing
    gap_threshold   : threshold on normalized [0,1] profile (default 0.90)
    min_gap_px      : minimum gap width (pixels) to count as a separator
    min_frame_px    : minimum frame width (pixels) to keep
    buffer_px       : pixels to trim inward from each gap center
    expected_frames : expected number of frames per TIFF (default 14 = 12 complete
                      + 2 partial); used both for the widest-gap filter (keeps at
                      most expected_frames-1 gaps) and fallback if too few gaps found

    Returns
    -------
    List of (left, right) frame boundary tuples, sorted left-to-right.
    """
    H, W = img_norm.shape
    y_lo = int(H * band_frac[0])
    y_hi = int(H * band_frac[1])

    col_mean   = img_norm[y_lo:y_hi, :].mean(axis=0)
    col_smooth = _gauss_smooth(col_mean, smooth_sigma)

    # Normalize to [0, 1] — this is the key step that makes thresholding work
    c_min, c_max = col_smooth.min(), col_smooth.max()
    if c_max > c_min:
        normalized = (col_smooth - c_min) / (c_max - c_min)
    else:
        normalized = np.zeros_like(col_smooth)

    gap_mask = normalized > gap_threshold

    # Find centers of sufficiently wide gap regions; track widths for filtering.
    # First merge nearby runs: the smoothing boundary effects can fragment a
    # single physical gap into 2-3 bright runs separated by a narrow dark dip.
    # Any two above-threshold runs separated by < min_frame_px are part of the
    # same inter-frame gap and should be merged.
    raw_runs = _connected_components_1d(gap_mask)
    merged_runs: list[tuple[int, int]] = []
    for s, e in raw_runs:
        if merged_runs and s - merged_runs[-1][1] < min_frame_px:
            merged_runs[-1] = (merged_runs[-1][0], e)
        else:
            merged_runs.append((s, e))

    gap_info: list[tuple[int, int]] = []   # (width_px, center_px)
    for s, e in merged_runs:
        w = e - s + 1
        if w >= min_gap_px:
            gap_info.append((w, int(round((s + e) / 2))))

    # Filter to true inter-frame gaps using bimodal width distribution.
    #
    # True inter-frame gaps (white film base between A-scopes) are always much
    # wider than any intra-frame feature (graticule lines, CRT edge artifacts,
    # partial-frame edges).  Sorting by width descending and looking for the
    # largest *relative* drop between adjacent entries reliably separates the two
    # populations — e.g. F141 TIFF 8525 shows a 91 % drop from the 12th true gap
    # (1024 px) to a reel-edge artifact (92 px) that would otherwise create a
    # spurious "partial" frame at the beginning.
    #
    # The bimodal cut is applied only when the drop is dramatic (> 50 %) and
    # enough gaps remain (>= 3) to form a sensible frame sequence.  After the
    # bimodal cut the hard cap max_gaps = expected_frames - 1 still applies as
    # a safety upper bound, so the old behaviour is preserved for TIFFs where
    # gap widths are all similar (no dramatic drop).
    gap_info.sort(key=lambda t: -t[0])   # widest first
    gap_info_all = list(gap_info)  # preserve pre-cut list for validation
    if len(gap_info) > 1:
        best_cut   = len(gap_info)
        best_ratio = 0.0
        for i in range(1, len(gap_info)):
            ratio = 1.0 - gap_info[i][0] / gap_info[i - 1][0]
            if ratio > best_ratio:
                best_ratio = ratio
                best_cut   = i
        if best_ratio > 0.50 and best_cut >= 3:
            # Validate: only apply bimodal cut if the discarded gaps are truly
            # tiny relative to the kept ones.  If the largest discarded gap is
            # > 20% of the smallest kept gap, both populations are real features
            # and the bimodal model is wrong (e.g. faint-trace TIFFs where
            # dim frames are nearly as bright as inter-frame borders).
            min_kept    = gap_info[best_cut - 1][0]
            max_discard = gap_info[best_cut][0]
            if max_discard < 0.20 * min_kept:
                print(f"  [detect_frames] bimodal gap cut: keeping {best_cut} wide gaps "
                      f"(drop {best_ratio:.0%} at rank {best_cut}: "
                      f"{gap_info[best_cut-1][0]}->{gap_info[best_cut][0]} px)")
                gap_info = gap_info[:best_cut]
            else:
                print(f"  [detect_frames] bimodal cut skipped: discarded gaps too large "
                      f"({max_discard} px >= 20% of {min_kept} px)")

    max_gaps = expected_frames - 1
    if len(gap_info) > max_gaps:
        gap_info = gap_info[:max_gaps]

    gap_centers = sorted([c for _, c in gap_info])

    # Fallback: if too few gaps, estimate from expected frame count
    if len(gap_centers) < min(3, expected_frames - 1):
        est_w = W / expected_frames if expected_frames > 0 else W
        gap_centers = [int((i + 0.5) * est_w) for i in range(expected_frames - 1)]

    # -- Spacing-guided gap prediction -----------------------------------------
    # When the film has uneven exposure (e.g. bright first half, dim second half),
    # global normalization fails to detect gaps in the dim section.  Use the
    # regular frame spacing from the detected gaps to predict where missing gaps
    # should be, then confirm each prediction against a local brightness peak.
    #
    # Physical basis: the 35 mm camera advance is mechanical and regular, so
    # inter-frame gaps have approximately constant spacing across the TIFF.
    min_needed = expected_frames - 1
    # Only trigger when >30% of expected gaps are missing (avoids interfering
    # with TIFFs that just have 1-2 missed gaps handled by the oversized splitter)
    if len(gap_centers) >= 3 and len(gap_centers) < 0.7 * min_needed:
        # Compute median spacing from consecutive detected gaps
        spacings = [gap_centers[i+1] - gap_centers[i]
                    for i in range(len(gap_centers) - 1)]
        med_spacing = float(np.median(spacings))

        # Build predicted gap positions: fill oversized intervals between
        # consecutive gaps, then extend from the endpoints
        predicted = set()
        # Fill within oversized intervals between consecutive detected gaps
        all_edges = [0] + gap_centers + [W]
        for i in range(len(all_edges) - 1):
            interval = all_edges[i + 1] - all_edges[i]
            if interval > 1.5 * med_spacing:
                n_gaps = int(round(interval / med_spacing)) - 1
                if n_gaps >= 1:
                    step = interval / (n_gaps + 1)
                    for j in range(1, n_gaps + 1):
                        p = int(round(all_edges[i] + j * step))
                        if min_frame_px < p < W - min_frame_px:
                            predicted.add(p)

        # Remove predictions close to existing gap centers
        tol = med_spacing * 0.25
        predicted = {p for p in predicted
                     if all(abs(p - g) > tol for g in gap_centers)}

        # Confirm each prediction: find the local brightness maximum within
        # +/- tol of the predicted position; accept if it's a local peak
        added = []
        for pred in sorted(predicted):
            search_lo = max(0, int(pred - tol))
            search_hi = min(W - 1, int(pred + tol))
            if search_hi <= search_lo:
                continue
            local = col_smooth[search_lo:search_hi + 1]
            peak_offset = int(np.argmax(local))
            peak_col = search_lo + peak_offset
            peak_val = col_smooth[peak_col]
            # Accept if the peak stands out from the surrounding frame content:
            # check that it's above the local mean in a wider neighborhood
            neigh_lo = max(0, peak_col - int(med_spacing * 0.5))
            neigh_hi = min(W - 1, peak_col + int(med_spacing * 0.5))
            neigh_mean = col_smooth[neigh_lo:neigh_hi + 1].mean()
            if peak_val > neigh_mean:
                added.append(peak_col)

        if added:
            gap_centers = sorted(gap_centers + added)
            # Re-apply the max_gaps cap
            if len(gap_centers) > max_gaps:
                # Keep the ones closest to expected regular spacing
                gap_centers = gap_centers[:max_gaps]
            print(f"  [detect_frames] spacing-guided prediction: added {len(added)} "
                  f"gaps (median spacing={med_spacing:.0f} px, "
                  f"total gaps now={len(gap_centers)})")

    # Build frames as intervals between gap centers
    frame_edges = [0] + gap_centers + [W]
    frames = []
    for i in range(len(frame_edges) - 1):
        left  = int(frame_edges[i])  + (buffer_px if i > 0 else 0)
        right = int(frame_edges[i + 1]) - (buffer_px if i < len(frame_edges) - 2 else 0)
        if right - left > min_frame_px:
            frames.append((max(0, left), min(W - 1, right)))

    # -- Post-process: split remaining oversized frames ----------------------------
    # After spacing-guided prediction, individual frames may still be oversized
    # (e.g. one missed gap in an otherwise well-detected TIFF).  Find the
    # brightest column in the middle region and split there.
    if len(frames) >= 2:
        changed = True
        while changed:
            changed = False
            widths = [r - l for l, r in frames]
            med_w  = float(np.median(widths))
            new_frames = []
            for l, r in frames:
                if (r - l) > 1.5 * med_w and (r - l) > min_frame_px * 2:
                    sub_l    = l + int(0.20 * (r - l))
                    sub_r    = l + int(0.80 * (r - l))
                    sub_peak = sub_l + int(np.argmax(col_smooth[sub_l:sub_r]))
                    left_end  = max(l,     sub_peak - buffer_px - 1)
                    right_beg = min(W - 1, sub_peak + buffer_px + 1)
                    if (left_end - l > min_frame_px
                            and r - right_beg > min_frame_px):
                        new_frames.append((l, left_end))
                        new_frames.append((right_beg, r))
                        print(f"  [detect_frames] oversized frame {l}-{r} "
                              f"(w={r-l}) split at col {sub_peak}")
                        changed = True
                    else:
                        new_frames.append((l, r))
                else:
                    new_frames.append((l, r))
            frames = sorted(new_frames)

    # -- Post-process: discard empty-gap frames (no CRT signal) --------------------
    # Some TIFFs have extra-wide inter-frame gaps (4000+ px) that survive gap
    # detection and get treated as oversized frames, then split.  The resulting
    # "frames" are just bright film base with no dark CRT trace.  Detect these
    # by measuring column-brightness contrast in the CRT display region: real
    # frames have high std (bright background + dark trace), empty gaps have low.
    if len(frames) >= 3:
        crt_lo = int(H * 0.20)
        crt_hi = int(H * 0.80)
        crt_band = img_norm[crt_lo:crt_hi, :]
        contrasts = []
        for l, r in frames:
            col_means = crt_band[:, l:r+1].mean(axis=0)
            contrasts.append(float(col_means.std()))
        med_contrast = float(np.median(contrasts))
        # Keep frames with contrast > 15% of median (generous threshold to
        # retain faint CRT traces while rejecting truly empty gaps)
        if med_contrast > 0:
            valid = [(f, c) for f, c in zip(frames, contrasts)
                     if c > 0.15 * med_contrast]
            n_removed = len(frames) - len(valid)
            if n_removed > 0:
                print(f"  [detect_frames] removed {n_removed} empty-gap "
                      f"frame(s) (no CRT signal, contrast < 15% of median "
                      f"{med_contrast:.4f})")
                frames = [f for f, _ in valid]

    return frames


# -- Signal extent detection ----------------------------------------------------

def detect_signal_extent(frame: np.ndarray,
                         cal: dict) -> tuple[int, int]:
    """
    Detect the horizontal extent of the CRT A-scope signal within a frame.

    The CRT beam sweeps left-to-right, creating a continuous dark trace on the
    phosphor screen.  Outside the sweep, only film grain exists — no distinct
    dark line.  This function finds the first and last columns where the CRT
    trace is present.

    Algorithm
    ---------
    1. Binary threshold on the globally-normalised frame (< signal_binary_thresh)
       to isolate dark pixels (CRT trace + some film grain).
    2. Exclude graticule rows (±signal_grat_half px around each 10-dB major
       grid line) so that horizontal grid lines do not inflate the count.
    3. Sum the dark-pixel count per column and smooth with a Gaussian (sigma = signal_density_sigma).
    4. Adaptive threshold: max(floor, min(cap, fraction × peak_density)).
       Bright frames (peak ~200) -> cap = 50 (same as original fixed threshold).
       Faint frames (peak ~40) -> floor = 12 (above film-grain noise ~5-6 px/col).
    5. Signal columns are those where the smoothed count >= adaptive threshold.
    6. Return the first and last such column.

    Parameters
    ----------
    frame : (H, W) float array, globally normalised [0, 1]
        The full-height frame (not the y_disp_lo:y_disp_hi band).
    cal   : calibration dict (must include y_ref_px, db_per_px, y_disp_lo, y_disp_hi).

    Returns
    -------
    (sig_start, sig_end) : column indices (frame-relative, inclusive).
        If detection fails, returns (0, W-1) as a safe fallback.
    """
    y_lo = cal["y_disp_lo"]
    y_hi = min(cal["y_disp_hi"], frame.shape[0])
    band = frame[y_lo:y_hi, :]
    bH, W = band.shape

    # -- Binary: dark pixels below threshold ----------------------------------
    binary_thresh = cal.get("signal_binary_thresh", 0.30)
    binary = (band < binary_thresh).astype(np.uint8)

    # -- Exclude graticule rows -----------------------------------------------
    grat_half = cal.get("signal_grat_half", 8)
    y_sp_px   = 10.0 / cal["db_per_px"]
    n_lo = int(np.floor((y_lo - cal["y_ref_px"]) / y_sp_px)) - 1
    n_hi = int(np.ceil ((y_hi - cal["y_ref_px"]) / y_sp_px)) + 1
    for _n in range(n_lo, n_hi + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px
        gl_rel = int(round(gl_abs - y_lo))
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - grat_half)
            r1 = min(bH, gl_rel + grat_half + 1)
            binary[r0:r1, :] = 0

    # -- Per-column dark-pixel count, smoothed --------------------------------
    col_count  = binary.sum(axis=0).astype(float)
    sigma      = cal.get("signal_density_sigma", 30)
    col_smooth = _gauss_smooth(col_count, sigma)

    # -- Adaptive threshold -------------------------------------------------
    # Fixed threshold (50) works for bright frames but excludes real echo
    # columns in faint CRT traces (e.g. F127 CBD0440: mean 18.5 dark px/col).
    # Adaptive formula: max(floor, min(cap, fraction × peak_density)).
    # For bright frames (peak ~200): cap=50 reached -> identical to original.
    # For faint frames (peak ~40): floor=12 reached -> catches real signal.
    # Film grain noise ~ 5-6 dark px/col; floor=12 stays safely above this.
    cap      = cal.get("signal_density_thresh", 50)
    floor    = cal.get("signal_density_floor",  12)
    fraction = cal.get("signal_density_frac",   0.25)
    peak_density   = float(np.max(col_smooth)) if len(col_smooth) > 0 else 0.0
    density_thresh = max(floor, min(cap, fraction * peak_density))

    # -- Signal extent: first/last column above threshold ---------------------
    above = col_smooth >= density_thresh
    signal_cols = np.where(above)[0]

    if len(signal_cols) == 0:
        # Fallback: no signal detected -> treat entire frame as signal
        return 0, W - 1

    sig_start, sig_end = int(signal_cols[0]), int(signal_cols[-1])

    # -- Local-normalization fallback for faint CRT traces ---------------------
    # A normal CRT sweep spans ~2000+ px.  If the standard thresholding finds
    # a suspiciously narrow extent (< 500 px), the CRT trace is likely too faint
    # for the global normalisation to resolve.  Re-try with per-band local
    # normalisation, which stretches contrast so that even faint traces cross
    # the binary threshold.  This path is never reached for flights with strong
    # CRT traces (F125, F127) because their initial extent is already wide.
    if sig_end - sig_start < 500:
        b_min, b_max = float(band.min()), float(band.max())
        if b_max - b_min > 1e-9:
            band_local = (band - b_min) / (b_max - b_min)
            binary2 = (band_local < binary_thresh).astype(np.uint8)
            # Re-apply graticule exclusion
            for _n in range(n_lo, n_hi + 1):
                gl_abs = cal["y_ref_px"] + _n * y_sp_px
                gl_rel = int(round(gl_abs - y_lo))
                if 0 <= gl_rel < bH:
                    r0 = max(0, gl_rel - grat_half)
                    r1 = min(bH, gl_rel + grat_half + 1)
                    binary2[r0:r1, :] = 0
            col_count2  = binary2.sum(axis=0).astype(float)
            col_smooth2 = _gauss_smooth(col_count2, sigma)
            peak2       = float(np.max(col_smooth2)) if len(col_smooth2) > 0 else 0.0
            thresh2     = max(floor, min(cap, fraction * peak2))
            signal_cols2 = np.where(col_smooth2 >= thresh2)[0]
            if len(signal_cols2) > 0:
                s2, e2 = int(signal_cols2[0]), int(signal_cols2[-1])
                if e2 - s2 > sig_end - sig_start:
                    sig_start, sig_end = s2, e2

    return sig_start, sig_end


# -- Trace extraction -----------------------------------------------------------

def extract_trace(frame: np.ndarray,
                  cal: dict,
                  robust: bool = False,
                  mb_x: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract oscilloscope trace from one A-scope frame.

    Finds the darkest (minimum intensity) pixel per column within the
    oscilloscope display band, then applies Gaussian smoothing.

    The A-scope trace appears as a dark line on a lighter background in the
    scanned TIFF (CRT phosphor -> film -> scan inversion).

    Parameters
    ----------
    robust : bool
        If True, use a constrained-argmin approach that rejects film-grain
        artefacts and T/R ringing noise.  Requires mb_x.

        Algorithm
        ---------
        1. Compute unconstrained argmin per column.
        2. Pre-filter: replace the T/R ringing region
           [0, mb_x + mb_skip_px] and frame edges (±20 px) with the noise
           floor row so the coarse trajectory is not biased by ringing.
        3. Apply a coarse Gaussian smooth (sigma=30) to the pre-filtered trace
           to obtain a stable "expected" row position for every column.
        4. Constrained argmin: at each column, restrict the search band to
           [coarse - 250, coarse + 250].  Wide enough for deep-ice bed
           echoes (~500 px above NF) while rejecting gross outliers.
        5. Running median (window=11) on the constrained trace to reject
           sporadic film artifacts (1–5 columns of dark pixels at y=500–800
           that produce phantom dips ~30 dB above NF in the post-echo
           region).  Real echoes span hundreds of columns and are preserved.
        6. Apply fine Gaussian smooth (sigma=5) to the median-filtered result
           -> trace_y_s used for envelope walking.

        Why it works: film grain creates isolated 1–3 px dark hits; real
        echoes are 50–500 px wide.  The coarse smooth (sigma=30) is largely
        unaffected by sparse grain hits (their total weight << 1) but
        accurately tracks the broad echo envelope.  The ±250 px search
        window is wide enough to capture both shallow and deep-ice echoes.
        The median filter then rejects any remaining sporadic artifacts
        without narrowing the search window (which would miss deep-ice
        bed echoes, as observed in F125 TIFF 8300).

    mb_x : int | None
        Main-bang column (frame-relative).  Required in robust mode.

    Returns
    -------
    trace_y   : (W,) row positions (pixels; lower = higher power)
    trace_y_s : (W,) Gaussian-smoothed row positions
    """
    H, W   = frame.shape
    y_lo   = cal["y_disp_lo"]
    y_hi   = min(cal["y_disp_hi"], H)
    band   = frame[y_lo:y_hi, :]
    bH     = band.shape[0]

    # -- Graticule masking ------------------------------------------------------
    # Major graticule lines (10 dB horizontal rules) are printed with dense ink
    # and appear DARKER than the CRT echo trace in the globally-normalised TIFF.
    # Without masking, argmin picks a graticule line instead of the signal peak,
    # producing a systematic ~200 px (~10 dB) power underestimation.
    #
    # Fix: set ±graticule_mask_half_px around every known major grid row to 1.0
    # (white -> never selected by argmin).  Grid positions are computed exactly
    # from step-2 calibration (y_ref_px, db_per_px); the formula is
    # flight-independent and adapts to F125 / F127 / F141 automatically.
    mask_half = cal.get("graticule_mask_half_px", 3)
    y_sp_px   = 10.0 / cal["db_per_px"]          # px per 10 dB major division
    band_gm   = band.astype(float).copy()         # graticule-masked band
    n_lo_g = int(np.floor((y_lo - cal["y_ref_px"]) / y_sp_px)) - 1
    n_hi_g = int(np.ceil( (y_hi - cal["y_ref_px"]) / y_sp_px)) + 1
    for _n in range(n_lo_g, n_hi_g + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px   # absolute image row
        gl_rel = int(round(gl_abs - y_lo))          # band-relative row
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - mask_half)
            r1 = min(bH, gl_rel + mask_half + 1)
            band_gm[r0:r1, :] = 1.0               # white -> never picked by argmin

    if robust and mb_x is not None:
        # -- Robust trace extraction — signal-extent–based masking ------------
        #
        # Root cause of spurious peaks: outside the CRT sweep region (before
        # signal start, after signal end), direct argmin picks random dark
        # film-grain pixels, producing a wiggly trace with local minima
        # misidentified as echoes.
        #
        # Fix: detect_signal_extent() finds the horizontal range where the CRT
        # trace is actually present (per-column dark-pixel density above
        # threshold).  Outside that range, force trace_y to the NF reference
        # row.  Inside, use graticule-masked argmin.
        #
        # Pre-MB columns (0..mb_x) within the signal region are kept with raw
        # argmin (no graticule masking) — this feeds estimate_noise_floor().
        sig_start, sig_end = detect_signal_extent(frame, cal)

        mb_skip_px = max(1, int(cal.get("mb_skip_us", 2.0) / cal["us_per_px"]))
        tr_end     = min(W, mb_x + mb_skip_px)

        # -- Pass 1: unconstrained argmin -------------------------------------
        trace_raw = np.argmin(band,    axis=0).astype(float) + y_lo  # no grat mask
        trace_gm  = np.argmin(band_gm, axis=0).astype(float) + y_lo  # grat masked

        nf_row = float(cal["y_ref_px"])

        # Assemble pass-1 trace: NF outside signal, argmin inside
        trace_pass1 = np.full(W, nf_row)
        trace_pass1[sig_start:sig_end+1] = trace_gm[sig_start:sig_end+1]
        # Pre-MB within signal: raw argmin (for NF estimation)
        pre_mb_start = max(sig_start, 0)
        pre_mb_end   = min(mb_x, sig_end + 1)
        if pre_mb_start < pre_mb_end:
            trace_pass1[pre_mb_start:pre_mb_end] = trace_raw[pre_mb_start:pre_mb_end]
        # T/R ringing zone -> NF
        trace_pass1[mb_x:tr_end] = nf_row

        # -- Constrained argmin (±250 px from coarse guide) -----------------
        # Two-pass strategy:
        #
        # Pass A: NF-clamped guide (trace_pass1).  Works for F125-type signals
        #   where echoes start near NF — the ±250 px window follows signal.
        #
        # Pass B (fallback): trace_gm guide.  Activated when Pass A leaves
        #   >80% of the echo region stuck at NF.  This handles F128-type
        #   multi-pulse frames where the signal starts 500-1000 px above NF
        #   immediately after T/R — outside the ±250 px window from NF.
        max_jump = 250

        def _run_constrained(guide_trace):
            """Constrained argmin over [sig_start, sig_end] using guide."""
            _ty = trace_pass1.copy()
            _coarse = _running_median(guide_trace, window=61)
            for col in range(sig_start, sig_end + 1):
                expected_row = _coarse[col] - y_lo
                lo_row = max(0,  int(expected_row - max_jump))
                hi_row = min(bH, int(expected_row + max_jump))
                if lo_row < hi_row:
                    best_row = lo_row + np.argmin(band_gm[lo_row:hi_row, col])
                    _ty[col] = best_row + y_lo
            if pre_mb_start < pre_mb_end:
                _ty[pre_mb_start:pre_mb_end] = trace_raw[pre_mb_start:pre_mb_end]
            _ty[mb_x:tr_end] = nf_row
            return _ty

        # Pass A: NF-clamped guide
        trace_y = _run_constrained(trace_pass1)

        # Check if echo region is stuck at NF — trigger Pass B if so
        _echo_lo = max(sig_start, tr_end)
        _echo_hi = min(sig_end, W - 1)
        if _echo_hi > _echo_lo + 30:
            _echo_trace = trace_y[_echo_lo:_echo_hi + 1]
            _stuck_frac = float(np.mean(np.abs(_echo_trace - nf_row) < 50))
            if _stuck_frac > 0.40:
                trace_y = _run_constrained(trace_gm)

        # -- Median filter + fine smooth for envelope walking -----------------
        # The constrained trace may have sporadic film artifacts (1–5 columns
        # where dark pixels at y=500–800 land within the ±250 band, producing
        # phantom dips ~30 dB above NF).  These artifacts don't affect peak
        # power reading (trace_y = trace_pass1 below), but they contaminate
        # the smooth trace used for envelope walking — a single phantom dip
        # resets the consecutive-below-NF count, extending the trailing edge.
        #
        # Running median (window=11) rejects artifacts spanning <=5 columns
        # while preserving real echoes (span hundreds of columns).
        trace_constrained = trace_y.copy()
        trace_mf = _running_median(trace_constrained, window=11)
        trace_y_s = _gauss_smooth(trace_mf, sigma=5.0)

        # -- Artifact detection data ----------------------------------------
        # Store per-column difference (in pixels) between the constrained
        # trace and the median-filtered trace.  Positive values indicate
        # that the median filter removed a downward artifact (constrained
        # trace was pulled below the median by a film artifact).
        # The per-column data is passed to step3, which checks for artifacts
        # specifically in the post-bed region (artifacts elsewhere are
        # benign for bed envelope estimation).
        art_diff_px = trace_mf - trace_constrained  # positive = artifact removed
        trace_info = dict(
            art_diff_px   = art_diff_px,   # full per-column array
            mb_x          = mb_x,
            db_per_px     = cal["db_per_px"],
            us_per_px     = cal["us_per_px"],
            signal_start  = sig_start,
            signal_end    = sig_end,
            echo_start    = tr_end,
        )

        # Return the UNCONSTRAINED pass-1 trace as trace_y for peak power
        # reading.  The unconstrained argmin correctly tracks the CRT trace
        # at echo peaks; grain contamination in flat regions does not matter
        # because peak power is only read at detected peak columns.
        trace_y = trace_pass1

    else:
        trace_y   = np.argmin(band_gm, axis=0).astype(float) + y_lo
        trace_y_s = _gauss_smooth(trace_y, cal["gauss_sigma"])
        trace_info = dict(art_diff_px=None)

    return trace_y.astype(int), trace_y_s, trace_info


def extract_astra_science_trace(
        frame: np.ndarray, cal: dict, mb_x: int, context: str,
        crop_start_y: int = ASTRA_DEFAULT_CROP_START_Y,
        crop_end_y: int = ASTRA_DEFAULT_CROP_END_Y
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use ASTRA's red trace inside LYRA's robust echo-detection region."""
    fallback, smooth, info = extract_trace(frame, cal, robust=True, mb_x=mb_x)
    unmasked, _, _ = extract_trace(frame, cal)
    fallback[:info["echo_start"]] = unmasked[:info["echo_start"]]
    red = astra_trace_with_fallback(
        frame, fallback, context, crop_start_y, crop_end_y)
    median_red = _running_median(red, window=11)
    filtered = _gauss_smooth(median_red, sigma=5.0)
    lo = max(info["signal_start"], info["echo_start"])
    hi = min(len(red), info["signal_end"] + 1)
    if lo < hi:
        smooth[lo:hi] = filtered[lo:hi]
        info["art_diff_px"][lo:hi] = (median_red - red)[lo:hi]
    return red, smooth, info


# -- Main bang detection --------------------------------------------------------

def detect_mb(trace_y_s: np.ndarray, cal: dict,
              search_frac: float = 0.25,
              guide_x: int | None = None,
              frame: np.ndarray | None = None) -> int:
    """
    Locate the main bang (transmitter pulse) — leftmost maximum power column.

    Five-tier detection strategy:

    1. GUIDED (optional guide_x):
       Narrow search window [guide_x-100, guide_x+50].  Baseline sampled from
       200 px BEFORE the window.  Finds the first 5-column sustained crossing
       20 dB above baseline, then the first local minimum >=15 dB below baseline.

    2. GUIDE POSITION CHECK (guide_x +/-20 px):
       For stable-MB flights, verifies guide_x sits near strong signal.

    3. FIRST-DIP (trace-based, pre-bang check):
       Scans left-to-right for the first sustained dip (>=10 columns below
       noise-10 dB) whose pre-bang region (80 px before) is clean (at most
       15% of columns below noise-8 dB).  The MB is the leading edge
       (run_start) of this dip.  Rejects frame-edge artifacts (x < 50) and
       secondary radar pulses (whose pre-bang overlaps echo signal).

    4. BROAD FALLBACK (legacy):
       Global argmin in left half of the frame.

    5. LAST RESORT: guide_x or mb_x_guess (800 px).

    Parameters
    ----------
    trace_y_s   : smoothed trace Y positions (from extract_trace)
    cal         : calibration dict (uses db_per_px, y_ref_db, mb_x_guess)
    search_frac : unused; kept for API compatibility
    guide_x     : optional search guide for the automatic detector
    frame       : normalised frame image (optional; enables image-aware Tier 3)

    Returns
    -------
    mb_x : column index (frame-relative)
    """
    W         = len(trace_y_s)
    db_per_px = max(cal.get("db_per_px", 0.049), 1e-6)
    mb_x: int | None = None
    _tier3_found = False   # Tier 3 validates with its own trace; skip sanity

    # -- TIER 1: GUIDED PATH (narrow window around optional guide) ---------
    if guide_x is not None:
        lo = max(30, guide_x - 100)
        hi = min(W,  guide_x + 50)
        if hi > lo:
            # Baseline from 200 px BEFORE window — avoids leading-edge corruption
            bl_start = max(30, lo - 200)
            bl_end   = max(bl_start + 10, lo)
            baseline = float(np.percentile(trace_y_s[bl_start:bl_end], 75))

            region   = trace_y_s[lo:hi]
            thresh_y = baseline - 20.0 / db_per_px

            _run = 0
            crossing = None
            for _i, _y in enumerate(region.tolist()):
                if _y < thresh_y:
                    _run += 1
                    if _run >= 5:
                        crossing = _i - 4   # start of sustained run
                        break
                else:
                    _run = 0

            if crossing is not None:
                # Collect ALL qualifying local minima in the search window,
                # then pick the one closest to guide_x.  This prevents
                # pre-main-bang artifacts from being selected over the real
                # MB when both are within the Tier 1 window.
                # (e.g. CBD 0457: artifact at x=757 vs real MB at x=812;
                #  guide_x=810; closest-to-guide correctly picks x=812.)
                min_depth = 15.0 / db_per_px
                sub = region[crossing : min(len(region), crossing + 300)]
                candidates = []  # list of (abs_x, depth_px)
                for i in range(1, len(sub) - 1):
                    if sub[i] < sub[i - 1] and sub[i] < sub[i + 1]:
                        if (baseline - sub[i]) >= min_depth:
                            candidates.append(lo + crossing + i)
                if candidates:
                    mb_x = min(candidates, key=lambda c: abs(c - guide_x))

    # -- Fallback tiers (when guided narrow-window found nothing) ---------
    # Noise floor reference for all fallback tiers
    global_noise_y = float(np.percentile(trace_y_s, 75))
    min_depth_px   = 10.0 / db_per_px

    # -- TIER 2: GUIDE POSITION CHECK -------------------------------------
    # When the guided narrow-window algorithm fails but guide_x exists,
    # check whether guide_x itself sits near a strong signal.  This handles
    # flights with STABLE MB positions (F125 ~800, F141 ~540) where
    # the optional guide is correct but the narrow-window algorithm couldn't
    # find a clean crossing (e.g. noisy trace, edge effects).
    # We search ±20 px around guide_x for the local minimum and verify
    # it's >=10 dB above noise.  This avoids falling through to the broad
    # search, which can latch onto pre-trigger artifacts (observed in F141:
    # artifact at ~150 px, -17.8 dB, stronger than the real MB at ~540 px).
    if mb_x is None and guide_x is not None:
        check_lo  = max(0, guide_x - 20)
        check_hi  = min(W, guide_x + 20)
        local_min = float(np.min(trace_y_s[check_lo:check_hi]))
        if (global_noise_y - local_min) >= min_depth_px:
            mb_x = check_lo + int(np.argmin(trace_y_s[check_lo:check_hi]))

    # -- TIER 3: FIRST-DIP (pre-bang check) ---------------------------------
    # Compute an INDEPENDENT unconstrained trace from the raw frame image
    # (argmin per column in the display band, lightly smoothed).  This is
    # critical because extract_trace()'s constrained two-pass method can
    # miss faint CRT signals (observed on F128 where the trace stays at
    # noise floor even at the real MB position).
    #
    # Scan left-to-right for the first SUSTAINED dip whose preceding region
    # is clean (noise floor).  Frame-edge artifacts and secondary radar
    # pulses fail the pre-bang check because their preceding region contains
    # strong signal or frame-separator noise.
    #
    # Validated on F125, F127, F128: 94%+ correct within 20 px of user
    # picks across 229 test frames.
    if mb_x is None and frame is not None:
        _disp_lo = int(cal.get("y_disp_lo", 300))
        _disp_hi = int(cal.get("y_disp_hi", 1700))
        _H_frame = frame.shape[0]
        _disp_hi = min(_disp_hi, _H_frame)
        # Unconstrained argmin per column (independent of extract_trace)
        _raw_trace = (np.argmin(frame[_disp_lo:_disp_hi, :], axis=0)
                      .astype(float) + _disp_lo)
        _light_smooth = _gauss_smooth(_raw_trace, sigma=5)

        _skip = 50
        _min_run = 10       # sustained dip = 10+ consecutive columns
        _noise_y = float(np.percentile(
            _light_smooth[_skip : max(_skip + 1, W - 50)], 90))
        _dip_thresh = _noise_y - 10.0 / db_per_px      # must dip >= 10 dB
        _pre_check  = _noise_y - 8.0 / db_per_px       # pre-bang ceiling
        _pre_len    = 80                                # pre-bang window size
        _pre_pct    = 0.15                              # max 15% below ceiling

        _run_start: int | None = None
        for _x in range(_skip, W - 50):
            if _light_smooth[_x] < _dip_thresh:
                if _run_start is None:
                    _run_start = _x
                if _x - _run_start + 1 >= _min_run:
                    # Check pre-bang cleanliness
                    _pb_lo = max(_skip, _run_start - _pre_len)
                    _pb_hi = _run_start
                    if _pb_hi - _pb_lo < 30:
                        _run_start = None
                        continue
                    _pre_bang = _light_smooth[_pb_lo:_pb_hi]
                    _n_below = int(np.sum(_pre_bang < _pre_check))
                    if _n_below <= _pre_pct * len(_pre_bang):
                        mb_x = _run_start     # leading edge of the dip
                        _tier3_found = True
                        break
                    else:
                        _run_start = None      # contaminated pre-bang
            else:
                _run_start = None

    # -- TIER 4: BROAD FALLBACK (legacy global argmin) ----------------------
    if mb_x is None:
        broad_smooth = _gauss_smooth(trace_y_s.astype(float), sigma=20)
        search_lo = 50
        search_hi = max(search_lo + 1, W // 2)
        candidate = search_lo + int(np.argmin(broad_smooth[search_lo:search_hi]))
        if (global_noise_y - broad_smooth[candidate]) >= min_depth_px:
            mb_x = candidate

    # -- TIER 5: LAST RESORT -----------------------------------------------
    if mb_x is None:
        mb_x = guide_x if guide_x is not None else cal.get("mb_x_guess", 800)

    # Sanity: main bang must be well above the noise-floor reference.
    # Threshold relaxed from 15 dB to 10 dB to accommodate weaker frames
    # (e.g. F127 CBD0421 at -45.9 dB = 14.1 dB above noise floor).
    # Skip when Tier 3 found the MB: it uses its own independent trace for
    # validation (the constrained trace_y_s can be at noise floor at the
    # real MB position when the CRT signal is faint, as on F128).
    if not _tier3_found:
        mb_power = float(px_to_db(float(trace_y_s[mb_x]), cal))
        if mb_power < cal["y_ref_db"] + 10:
            mb_x = guide_x if guide_x is not None else cal.get("mb_x_guess", 800)

    return mb_x


# -- Noise floor estimation -----------------------------------------------------

def estimate_noise_floor(trace_y_s: np.ndarray,
                         mb_x: int,
                         cal: dict) -> float:
    """
    Estimate per-frame noise floor from the pre-main-bang baseline.

    The region before the main bang contains no real signal, so the
    trace sits at the system noise floor level.

    Returns
    -------
    noise_floor_dB : estimated noise floor in dB
    """
    pre_mb = trace_y_s[:max(1, mb_x - 30)]
    if len(pre_mb) >= 10:
        # Median of the highest-Y (lowest-power) portion -> noise baseline
        p75 = float(np.percentile(pre_mb, 75))
        baseline_px = float(np.median(pre_mb[pre_mb >= p75]))
        return float(px_to_db(baseline_px, cal))
    return float(cal["y_ref_db"])


# -- Echo peak detection --------------------------------------------------------

def detect_echoes(trace_y_s: np.ndarray,
                  mb_x: int,
                  noise_floor_dB: float,
                  cal: dict) -> tuple[int | None, int | None]:
    """
    Detect surface and bed echo peak columns.

    Surface: first prominent peak after main bang decay.
    Bed:     second prominent peak, at least 0.3 µs after surface.

    Returns
    -------
    surface_x : frame-relative column of surface peak, or None
    bed_x     : frame-relative column of bed peak, or None
    """
    W = len(trace_y_s)

    # Skip main bang + its decay
    x_start = mb_x + int(cal["mb_skip_us"] / cal["us_per_px"])
    x_end   = min(W - 1, mb_x + int(20.0 / cal["us_per_px"]))

    if x_start >= x_end:
        return None, None

    region = trace_y_s[x_start:x_end]

    # find_peaks on -trace (minima in y = maxima in power)
    prom_px  = cal["peak_min_prom_dB"] / cal["db_per_px"]
    dist_px  = cal["peak_min_dist_px"]
    peak_rel = _find_peaks_numpy(region, prominence=prom_px, distance=dist_px)

    if len(peak_rel) == 0:
        return None, None

    peak_cols   = peak_rel + x_start
    peak_powers = np.array([float(px_to_db(trace_y_s[c], cal)) for c in peak_cols])

    # Must be above noise floor by at least 3 dB
    valid       = peak_powers - noise_floor_dB > 3.0
    peak_cols   = peak_cols[valid]
    peak_powers = peak_powers[valid]

    if len(peak_cols) == 0:
        return None, None

    # -- Surface selection ------------------------------------------------------
    # Default: first peak after mb_skip_us.  Robustness guard: if a later peak
    # within the surface search window is substantially stronger (>min_surface_
    # lead_dB), prefer it.  This rejects weak T/R ringing artefacts that slip
    # past mb_skip_us (e.g. CBD0461 false peak at 2.77 µs, 30 dB below real
    # surface at 5.68 µs).  The surface_window_us bound prevents accidentally
    # promoting the bed echo (which is outside the window).
    srf_win_px   = mb_x + int(cal.get("surface_window_us", 8.0) / cal["us_per_px"])
    srf_mask     = peak_cols <= srf_win_px
    srf_cands    = peak_cols[srf_mask]
    srf_powers_c = peak_powers[srf_mask]

    if len(srf_cands) == 0:           # no peak in window -> fall back to first
        srf_cands    = peak_cols[:1]
        srf_powers_c = peak_powers[:1]

    if len(srf_cands) >= 2:
        first_power = srf_powers_c[0]
        best_idx    = int(np.argmax(srf_powers_c))
        best_power  = srf_powers_c[best_idx]
        # Promote strongest candidate if first peak is a weak artefact
        if best_power - first_power > cal.get("min_surface_lead_dB", 15.0):
            surface_x = int(srf_cands[best_idx])
        else:
            surface_x = int(srf_cands[0])
    else:
        surface_x = int(srf_cands[0])

    if len(peak_cols) < 2:
        return surface_x, None

    # Bed: first peak at least min_bed_gap_us after surface (~168 m minimum ice)
    # that also passes the quiet-gap check.
    #
    # Quiet-gap check: somewhere in the window [surface_x + lead_px, cand_x],
    # the smoothed trace must return within quiet_gap_dB of the noise floor.
    # This rejects candidates that are continuous with the surface echo (e.g.
    # phosphor afterglow, trailing ringing, film blemishes in the inter-echo
    # region) while accepting genuine bed echoes, which are preceded by a quiet
    # region where the surface echo has fully decayed.
    # The check is skipped when the gap is < 1 µs (very thin ice, h < 84 m).
    min_gap_px    = int(cal.get("min_bed_gap_us", 2.0) / cal["us_per_px"])
    candidates    = peak_cols[peak_cols >= surface_x + min_gap_px]
    if len(candidates) == 0:
        return surface_x, None

    quiet_lead_px   = int(cal.get("quiet_gap_lead_us", 1.0)  / cal["us_per_px"])
    quiet_gap_px    = cal.get("quiet_gap_dB",   12.0) / cal["db_per_px"]
    nf_row          = float(cal["y_ref_px"])
    min_bed_w_px    = int(cal.get("min_bed_width_us",  0.5) / cal["us_per_px"])
    width_thresh_y  = nf_row - cal.get("min_bed_width_dB", 10.0) / cal["db_per_px"]
    W               = len(trace_y_s)

    # Collect ALL candidates that pass quiet-gap and width checks, then
    # select the most prominent one.  Previous logic took the first passing
    # candidate, which misidentified weak intermediate features (surface
    # ringing, shallow internal reflectors) as the bed when the real bed
    # echo was further away and much stronger.
    valid_beds = []   # list of (column, power_dB)
    for cand in candidates:
        c      = int(cand)
        gap_lo = surface_x + quiet_lead_px
        gap_hi = c

        # -- Quiet-gap check ---------------------------------------------------
        # Skip when gap is < 1 µs (very thin ice, h <= 84 m)
        if gap_hi - gap_lo >= int(1.0 / cal["us_per_px"]):
            gap_trace = trace_y_s[gap_lo:gap_hi]
            if float(np.max(gap_trace)) < nf_row - quiet_gap_px:
                continue   # trace never returns to NF -> continuous artifact

        # -- Width check -------------------------------------------------------
        # The smoothed trace must stay above width_thresh_y for at least
        # min_bed_w_px consecutive columns around the candidate.
        # Real bed echoes: >= 0.83 µs wide (F125 TIFF 8400 validated data).
        # Film artifacts / surface-echo ringing: < 0.45 µs wide.
        w_lo    = max(0, c - min_bed_w_px * 3)
        w_hi    = min(W, c + min_bed_w_px * 3)
        above   = (trace_y_s[w_lo:w_hi] < width_thresh_y).astype(np.int8)
        # Maximum consecutive run of True (columns clearly above threshold)
        padded  = np.zeros(len(above) + 2, dtype=np.int8)
        padded[1:-1] = above
        diff    = np.diff(padded)
        starts  = np.where(diff > 0)[0]
        ends    = np.where(diff < 0)[0]
        max_run = 0 if len(starts) == 0 else int((ends - starts).max())
        if max_run < min_bed_w_px:
            continue   # too narrow -> film artifact or surface-echo ringing

        cand_power = float(px_to_db(trace_y_s[c], cal))
        valid_beds.append((c, cand_power))

    # Select the most prominent (strongest power) bed candidate
    if valid_beds:
        bed_x = int(max(valid_beds, key=lambda b: b[1])[0])
    else:
        bed_x = None

    return surface_x, bed_x


# -- Envelope walker ------------------------------------------------------------

def walk_envelope(trace_y_s: np.ndarray,
                  x_peak: int,
                  thresh_db: float,
                  mb_x: int,
                  frame_w: int,
                  cal: dict,
                  max_half_width_us: float = 0.0) -> tuple[float, float, float]:
    """
    Walk outward from a confirmed peak to find echo leading/trailing edges
    at a given power threshold.

    Leading edge  : walk LEFT until trace crosses threshold (first crossing).
    Trailing edge : walk RIGHT until `consec` consecutive columns are below
                    threshold (confirming sustained return to baseline).

    Parameters
    ----------
    thresh_db : absolute threshold in dB (e.g. noise_floor_dB + 10)
    max_half_width_us : maximum distance (µs) from peak to either edge.
        If > 0, caps the walk to prevent runaway envelopes (e.g., F143
        inter-echo signal staying above NF+5).  A surface echo cannot
        physically be wider than ~2 µs; a bed echo ~3-4 µs.
        Default 0 = no cap (backward compatible).

    Returns
    -------
    lead_us, trail_us, width_us : TWT values (µs from main bang)
    """
    thresh_y = db_to_px(thresh_db, cal)   # pixel row at threshold power
    consec   = cal["consec"]

    # Physical TWT bound for the rightward walk: cap at max_walk_us from MB.
    # Prevents the envelope extending into frame separators (all-white columns
    # look like "very high power" because argmin returns y_disp_lo there) or
    # beyond the maximum realistic two-way travel time for RIS geometry.
    max_walk_us = cal.get("max_walk_us", 17.0)
    x_right_max = min(frame_w - 1,
                      mb_x + int(max_walk_us / cal["us_per_px"]))

    # Per-echo width cap: limit walk distance from peak to prevent runaway
    # envelopes when inter-echo signal stays above threshold (F143 NF+5).
    if max_half_width_us > 0:
        max_half_px = int(max_half_width_us / cal["us_per_px"])
        x_right_max = min(x_right_max, x_peak + max_half_px)

    # Walk LEFT from peak
    lead = x_peak
    x_left_min = max(0, x_peak - 1500)
    if max_half_width_us > 0:
        max_half_px = int(max_half_width_us / cal["us_per_px"])
        x_left_min = max(x_left_min, x_peak - max_half_px)
    for x in range(x_peak, x_left_min, -1):
        if trace_y_s[x] >= thresh_y:   # trace at or below threshold
            lead = x
            break
    else:
        lead = x_left_min

    # Walk RIGHT — require `consec` consecutive below-threshold columns
    trail = x_peak
    count = 0
    for x in range(x_peak, x_right_max):
        if trace_y_s[x] >= thresh_y:
            count += 1
            if count >= consec:
                trail = x - consec
                break
        else:
            count = 0
    else:
        trail = x_right_max

    lead_us  = float(px_to_us(lead,  mb_x, cal))
    trail_us = float(px_to_us(trail, mb_x, cal))
    return lead_us, trail_us, float(trail_us - lead_us)


# -- Peakiness -----------------------------------------------------------------

def compute_peakiness(trace_y_s: np.ndarray,
                      x_peak: int,
                      lead_px: int,
                      trail_px: int,
                      cal: dict) -> float:
    """
    Peakiness = peak power (linear) / mean power (linear) within ±10 dB window.

    > 1 : sharply peaked (specular reflector / liquid water)
    ~ 1 : flat-topped (rough / noise-dominated)
    """
    x0 = max(0, lead_px)
    x1 = min(len(trace_y_s) - 1, trail_px) + 1
    if x1 <= x0 + 2:
        return np.nan

    powers_dB  = np.array([float(px_to_db(trace_y_s[x], cal)) for x in range(x0, x1)])
    powers_lin = 10.0 ** (powers_dB / 10.0)
    peak_lin   = 10.0 ** (float(px_to_db(int(round(trace_y_s[x_peak])), cal)) / 10.0)

    mean_lin = float(np.mean(powers_lin))
    if mean_lin <= 0:
        return np.nan
    return float(peak_lin / mean_lin)


# -- Per-echo full metrics ------------------------------------------------------

def compute_echo_metrics(trace_y_s: np.ndarray,
                         x_peak: int,
                         noise_floor_dB: float,
                         mb_x: int,
                         frame_w: int,
                         cal: dict,
                         trace_y: np.ndarray | None = None) -> EchoResult:
    """
    Compute complete waveform metrics for one detected echo peak.

    Parameters
    ----------
    trace_y : optional raw (pre-smooth) trace from extract_trace.
        When supplied, peak power is read from the raw argmin within ±2 px
        of x_peak rather than from the smoothed trace.  The smoothed trace
        (trace_y_s) is still used for all envelope-walking operations.
        Without trace_y, peak power falls back to trace_y_s[x_peak].
    """
    # Peak power: use the raw trace (or a local window around x_peak) so that
    # Gaussian smoothing does not bias the peak value downward.
    # Window ±15 px: the smoothed trace can shift the peak column by several
    # pixels (asymmetric echoes), so a narrow ±2 px window may miss the true
    # minimum.  ±15 px covers the peak of any real echo while remaining small
    # enough to avoid picking up adjacent features.
    if trace_y is not None:
        lo  = max(0, x_peak - 15)
        hi  = min(len(trace_y), x_peak + 16)
        peak_y = int(np.min(trace_y[lo:hi]))
    else:
        peak_y = int(round(trace_y_s[x_peak]))
    peak_twt   = float(px_to_us(x_peak, mb_x, cal))
    peak_power = float(px_to_db(peak_y, cal))
    peak_snr   = peak_power - noise_floor_dB

    result = EchoResult(
        peak_x=x_peak, peak_y=peak_y,
        peak_twt_us=peak_twt,
        peak_power_dB=peak_power,
        peak_snr_dB=peak_snr,
    )

    # Envelope at +10 dB above measured noise floor
    thresh_10 = noise_floor_dB + 10.0
    if peak_power > thresh_10:
        l10, t10, w10 = walk_envelope(
            trace_y_s, x_peak, thresh_10, mb_x, frame_w, cal)
        result.lead_10_us  = l10
        result.trail_10_us = t10
        result.width_10_us = w10
        result.leading_rise_us = peak_twt - l10

        lead_10_px  = us_to_px(l10, mb_x, cal)
        trail_10_px = us_to_px(t10, mb_x, cal)
        result.peakiness = compute_peakiness(
            trace_y_s, x_peak, lead_10_px, trail_10_px, cal)

        trailing_w = t10 - peak_twt
        leading_w  = peak_twt - l10
        if leading_w > 1e-6:
            result.asymmetry = trailing_w / leading_w

    # Envelope at +5 dB above measured noise floor
    thresh_5 = noise_floor_dB + 5.0
    if peak_power > thresh_5:
        l5, t5, w5 = walk_envelope(
            trace_y_s, x_peak, thresh_5, mb_x, frame_w, cal)
        result.lead_5_us  = l5
        result.trail_5_us = t5
        result.width_5_us = w5
        if not np.isnan(result.trail_10_us):
            result.trailing_tail_us = t5 - result.trail_10_us

    # -- First-break onset: walk LEFT from peak to NF + onset_db ---------------
    # The onset (first break) is where the signal first rises above the noise
    # floor.  This gives the shortest two-way travel time (nadir reflection),
    # which is more accurate for ice thickness than the peak TWT — especially
    # over rough beds where off-nadir scattering can shift the peak later.
    #
    # Threshold: NF + 3 dB (just above noise; robust against noise fluctuations
    # while catching the true first arrival).  Walk left from the peak using the
    # smoothed trace; the onset is the last column above threshold before the
    # signal drops into the noise.
    ONSET_ABOVE_NF_DB = 3.0
    onset_thresh_dB = noise_floor_dB + ONSET_ABOVE_NF_DB
    if peak_power > onset_thresh_dB:
        onset_thresh_y = db_to_px(onset_thresh_dB, cal)
        onset_x = x_peak
        for x in range(x_peak, max(mb_x, 0), -1):
            if trace_y_s[x] >= onset_thresh_y:   # at or below threshold
                onset_x = x + 1 if x < x_peak else x_peak
                break
        else:
            onset_x = max(mb_x + 1, 0)
        # Clamp to valid range
        onset_x = max(mb_x + 1, min(onset_x, x_peak))
        result.onset_twt_us = float(px_to_us(onset_x, mb_x, cal))
        result.onset_power_dB = float(px_to_db(int(round(trace_y_s[onset_x])), cal))

    return result


# -- Local column profile extraction -------------------------------------------

def extract_local_profile(frame: np.ndarray,
                          echo_x: int,
                          mb_x: int,
                          noise_floor_dB: float,
                          cal: dict,
                          half_window: int = 80,
                          guide_y: float | None = None,
                          guide_tol: int = 150,
                          max_half_width_us: float = 4.0,
                          peak_y_override: float | None = None) -> EchoResult:
    """
    Extract calibrated echo metrics from a local column window around a
    known echo position, bypassing the global continuous trace entirely.

    This is immune to the global trace failure mode where competing dark
    features (graticule lines, reference lines) pull the argmin away from
    the CRT signal.  Within a narrow local window centered on the echo,
    the echo peak is the dominant dark feature.

    Parameters
    ----------
    frame : (H, W) normalized float array [0, 1]
    echo_x : column position of the echo (frame-relative)
    mb_x : main bang column (frame-relative)
    noise_floor_dB : estimated per-frame noise floor (dB)
    cal : calibration dict (y_ref_px, db_per_px, us_per_px, y_disp_lo, y_disp_hi)
    half_window : columns left/right of echo_x to include (default 80)
    guide_y : optional Y-row hint from the global trace at echo_x.
        When provided, the local argmin is constrained to within +-guide_tol
        rows of this value, preventing dark artifacts far from the CRT trace
        from pulling the peak estimate away.
    guide_tol : half-width of the constraint window around guide_y (default 150 px)

    Returns
    -------
    EchoResult with peak power, SNR, envelope widths, peakiness, etc.
    """
    H, W   = frame.shape
    y_lo   = cal["y_disp_lo"]
    y_hi   = min(cal["y_disp_hi"], H)

    # Column bounds (clamp to frame)
    x0 = max(0, echo_x - half_window)
    x1 = min(W, echo_x + half_window + 1)
    if x1 <= x0:
        return EchoResult()

    # Extract local band (display region only)
    band = frame[y_lo:y_hi, x0:x1].astype(float).copy()
    bH   = band.shape[0]

    # -- Graticule masking (same logic as extract_trace) -----------------------
    mask_half = cal.get("graticule_mask_half_px", 3)
    y_sp_px   = 10.0 / cal["db_per_px"]
    n_lo_g = int(np.floor((y_lo - cal["y_ref_px"]) / y_sp_px)) - 1
    n_hi_g = int(np.ceil( (y_hi - cal["y_ref_px"]) / y_sp_px)) + 1
    for _n in range(n_lo_g, n_hi_g + 1):
        gl_abs = cal["y_ref_px"] + _n * y_sp_px
        gl_rel = int(round(gl_abs - y_lo))
        if 0 <= gl_rel < bH:
            r0 = max(0, gl_rel - mask_half)
            r1 = min(bH, gl_rel + mask_half + 1)
            band[r0:r1, :] = 1.0

    # -- Determine whether the global trace is usable at this echo position ----
    # When guide_y is above noise floor, the global trace is tracking the CRT
    # signal correctly.  Use guide_y directly for peak_y (power reading) so
    # dark artifacts in the local window cannot pull the estimate away.
    # When the trace has failed (near noise floor), fall back to unconstrained
    # local argmin — the original purpose of extract_local_profile().
    _guide_active = False
    if guide_y is not None:
        guide_power = px_to_db(guide_y, cal)
        if guide_power - noise_floor_dB > 5.0:
            _guide_active = True

    # -- Per-column argmin within local window ---------------------------------
    local_trace_rel = np.argmin(band, axis=0).astype(float)  # band-relative
    local_trace_abs = local_trace_rel + y_lo                  # absolute row

    # Light median filter (window=5) to reject single-pixel film grain
    local_trace_filt = _running_median(local_trace_abs, window=5)

    # Also keep raw (unfiltered) for peak power reading
    local_trace_raw = local_trace_abs.copy()

    # -- Find peak position and Y row ------------------------------------------
    if peak_y_override is not None:
        if not np.isfinite(peak_y_override) or not 0 <= peak_y_override < H:
            raise ValueError("Peak row must be finite and inside the frame")
        peak_x_abs = echo_x
        peak_y = float(peak_y_override)
    elif _guide_active:
        # Trust the global trace: use echo_x as peak column, guide_y as peak row.
        # Read the darkest pixel in a small +-10 row window around guide_y at
        # echo_x to handle sub-pixel variations in the CRT phosphor.
        peak_x_abs = echo_x
        gy = int(round(guide_y))
        gy_lo = max(0, gy - 10)
        gy_hi = min(H, gy + 11)
        # Read a small column band around echo_x too (+-3 px)
        gx_lo = max(0, echo_x - 3)
        gx_hi = min(W, echo_x + 4)
        peak_y = int(np.min(frame[gy_lo:gy_hi, gx_lo:gx_hi] * H))
        # But frame values are [0,1] -- we need the row, not the intensity.
        # Find the darkest row near guide_y at echo_x:
        col_slice = frame[gy_lo:gy_hi, gx_lo:gx_hi]
        mean_intensity = col_slice.mean(axis=1)  # average across the few columns
        peak_y = gy_lo + int(np.argmin(mean_intensity))
    else:
        # Trace failed: use unconstrained local argmin
        peak_local_idx = int(np.argmin(local_trace_filt))
        peak_x_abs     = x0 + peak_local_idx   # frame-relative column
        # Peak power: read from raw trace in +-5 px around detected peak
        raw_lo = max(0, peak_local_idx - 5)
        raw_hi = min(len(local_trace_raw), peak_local_idx + 6)
        peak_y = int(np.min(local_trace_raw[raw_lo:raw_hi]))

    peak_twt   = float(px_to_us(peak_x_abs, mb_x, cal))
    peak_power = float(px_to_db(peak_y, cal))
    peak_snr   = peak_power - noise_floor_dB

    result = EchoResult(
        peak_x=peak_x_abs, peak_y=peak_y,
        peak_twt_us=peak_twt,
        peak_power_dB=peak_power,
        peak_snr_dB=peak_snr,
    )

    # -- Envelope walking using local filtered trace ---------------------------
    # Build a full-frame-width trace padded with NF for the envelope walker
    # (walk_envelope expects frame-width arrays and uses absolute column indices)
    nf_row = float(cal["y_ref_px"])
    full_trace = np.full(W, nf_row, dtype=float)
    full_trace[x0:x1] = local_trace_filt

    # Fine smooth within the local window for envelope walking
    local_smooth = _gauss_smooth(local_trace_filt, sigma=3.0)
    full_trace_s = np.full(W, nf_row, dtype=float)
    full_trace_s[x0:x1] = local_smooth

    # Envelope at +10 dB above noise floor
    thresh_10 = noise_floor_dB + 10.0
    if peak_power > thresh_10:
        l10, t10, w10 = walk_envelope(
            full_trace_s, peak_x_abs, thresh_10, mb_x, W, cal,
            max_half_width_us=max_half_width_us)
        result.lead_10_us  = l10
        result.trail_10_us = t10
        result.width_10_us = w10
        result.leading_rise_us = peak_twt - l10

        lead_10_px  = us_to_px(l10, mb_x, cal)
        trail_10_px = us_to_px(t10, mb_x, cal)
        result.peakiness = compute_peakiness(
            full_trace_s, peak_x_abs, lead_10_px, trail_10_px, cal)

        trailing_w = t10 - peak_twt
        leading_w  = peak_twt - l10
        if leading_w > 1e-6:
            result.asymmetry = trailing_w / leading_w

    # Envelope at +5 dB above noise floor
    thresh_5 = noise_floor_dB + 5.0
    if peak_power > thresh_5:
        l5, t5, w5 = walk_envelope(
            full_trace_s, peak_x_abs, thresh_5, mb_x, W, cal,
            max_half_width_us=max_half_width_us)
        result.lead_5_us  = l5
        result.trail_5_us = t5
        result.width_5_us = w5
        if not np.isnan(result.trail_10_us):
            result.trailing_tail_us = t5 - result.trail_10_us

    # -- First-break onset -----------------------------------------------------
    ONSET_ABOVE_NF_DB = 3.0
    onset_thresh_dB = noise_floor_dB + ONSET_ABOVE_NF_DB
    if peak_power > onset_thresh_dB:
        onset_thresh_y = db_to_px(onset_thresh_dB, cal)
        onset_x = peak_x_abs
        for x in range(peak_x_abs, max(mb_x, x0), -1):
            if full_trace_s[x] >= onset_thresh_y:
                onset_x = x + 1 if x < peak_x_abs else peak_x_abs
                break
        else:
            onset_x = max(mb_x + 1, x0)
        onset_x = max(mb_x + 1, min(onset_x, peak_x_abs))
        result.onset_twt_us = float(px_to_us(onset_x, mb_x, cal))
        result.onset_power_dB = float(px_to_db(
            int(round(full_trace_s[onset_x])), cal))

    # -- Peak-relative decay metrics ------------------------------------------
    # Walk from peak using thresholds relative to peak power (not noise floor).
    # These are SNR-independent and directly encode interface sharpness.
    peak_y_row = full_trace_s[peak_x_abs]
    db_per_px = cal["db_per_px"]

    for drop_db, attr in [(10.0, "decay_10_us"), (20.0, "decay_20_us")]:
        # Only compute if SNR is sufficient (peak must be > drop_db above NF)
        if peak_power < noise_floor_dB + drop_db + 2.0:
            continue
        thresh_y_pk = peak_y_row + drop_db / db_per_px  # lower power = higher row
        max_walk_px = int(max_half_width_us / cal["us_per_px"])
        x_limit = min(W - 1, peak_x_abs + max_walk_px)
        for x in range(peak_x_abs, x_limit):
            if full_trace_s[x] >= thresh_y_pk:
                decay_us = float(px_to_us(x, mb_x, cal)) - peak_twt
                setattr(result, attr, round(decay_us, 4))
                break

    # Rise time: walk LEFT from peak to peak-10 dB
    if peak_power >= noise_floor_dB + 12.0:
        thresh_y_rise = peak_y_row + 10.0 / db_per_px
        max_walk_px = int(max_half_width_us / cal["us_per_px"])
        x_limit_l = max(0, peak_x_abs - max_walk_px)
        for x in range(peak_x_abs, x_limit_l, -1):
            if full_trace_s[x] >= thresh_y_rise:
                rise_us = peak_twt - float(px_to_us(x, mb_x, cal))
                result.rise_10_us = round(rise_us, 4)
                break

    # Peak-relative asymmetry
    if not np.isnan(result.decay_10_us) and not np.isnan(result.rise_10_us):
        if result.rise_10_us > 1e-4:
            result.pk_asymmetry = round(result.decay_10_us / result.rise_10_us, 3)

    return result


# -- Y grid lines: extrapolation (graticule too faint to detect directly) ------

def compute_y_grid_lines(y_ref_px: float,
                          y_spacing_px: float,
                          y_disp_lo: int,
                          y_disp_hi: int,
                          n_above: int = 6,
                          n_below: int = 6,
                          ) -> list[float]:
    """
    Compute Y (horizontal) grid line positions by extrapolation.

    CRITICAL: The -60 dB reference line sits BETWEEN two major graticule
    divisions at -55 dB and -65 dB, not on a division.  The offset from
    y_ref_px to the nearest graticule line above (-55 dB) is half a major
    division ~ y_spacing_px / 2 ~ 102.5 px (validated from F125 Frame 8:
    y_ref=1507, nearest grid=1404 -> offset=103 px, from detect_gridlines3.py).

    Graticule lines: …, -45, -55, -65, -75, … dB
    NOT at:         …, -50, -60, -70, -80, … dB

    Formula:
        y_first = y_ref_px - y_spacing_px / 2    (-> -55 dB)
        y_grid[k] = y_first + k × y_spacing_px

    Returns
    -------
    Sorted list of Y row positions within [y_disp_lo, y_disp_hi].
    """
    # First graticule above the reference line (-55 dB)
    y_first = y_ref_px - y_spacing_px / 2.0
    lines = []
    for k in range(-n_above, n_below + 1):
        y = y_first + k * y_spacing_px
        if y_disp_lo <= y <= y_disp_hi:
            lines.append(float(y))
    return sorted(lines)


# -- X grid lines: ASTRA picks or signal-based fallback ------------------------

def detect_xgrid_lines(frame: np.ndarray,
                        mb_x: int,
                        x_spacing_px: float,
                        cal: dict,
                        n_major: int = 14,
                        x_band: tuple[int, int] = (1600, 1695),
                        gauss_sigma: float = 10.0,
                        half_search: int = 35,
                        inlier_thresh_px: float = 8.0,
                        guide_x_grid: list[int] | None = None,
                        d_from_mb: float | None = None,
                        ) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Use ASTRA-approved X-grid positions directly when available; otherwise
    detect X (vertical) major grid lines from a below-baseline image band.

    Critical band selection: y=[1600, 1695] sits BETWEEN the noise floor
    (y_ref ~ 1512) and the bottom major horizontal display line (y ~ 1717).
    In this region only MAJOR vertical grid lines are visible — minor tick
    notches exist only along horizontal major lines (y~1717, 1512, …) and
    do not appear inside this intermediate band.  The previous band (1550, 1800)
    included y~1717 where minor tick notches at every x_spacing/5 ~ 41 px
    scored higher than the diffuse major vertical lines, causing the phase
    search to lock onto minor tick positions (offset 1–4 minor ticks from
    the true major tick phase).

    Without ASTRA picks, the graticule is NOT anchored at mb_x. The algorithm:
      1. Finds the FIRST tick by argmax over the full first division
         [mb_x, mb_x + x_spacing_px] in the below-baseline strip.
      2. Builds all subsequent guides as first_x + k * x_spacing_px.
      3. Applies ±half_search snap to each guide (handles inter-frame drift).

    Parameters
    ----------
    frame         : (H, W) float32 frame array [0, 1]
    mb_x          : main bang column (anchor for guide positions)
    x_spacing_px  : expected major-division spacing in pixels
    cal           : calibration dict (uses y_disp_lo, y_disp_hi)
    n_major       : how many major lines to search for (rightward from mb_x)
    x_band        : (y_lo, y_hi) rows of below-baseline band
    gauss_sigma   : smoothing sigma for column mean profile
    half_search   : ±px search window around each guide position
    inlier_thresh_px : residual threshold for inlier-only linear fit
    guide_x_grid  : approved ASTRA grid coordinates, returned without adjustment

    Returns
    -------
    x_fitted     : (n_major,) array of fitted X positions (float)
    spacing_px   : spacing from inlier linear fit (px / major division)
    raw_x        : raw snapped grid detections before the linear fit
    """
    H, W = frame.shape
    if guide_x_grid:
        picked_grid = np.asarray(sorted(float(v) for v in guide_x_grid), dtype=float)
        if not np.all(np.isfinite(picked_grid)):
            raise ValueError("ASTRA X-grid picks must be finite")
        if len(picked_grid) >= 2:
            differences = np.diff(picked_grid)
            if np.any(differences <= 0):
                raise ValueError("ASTRA X-grid picks must be distinct and increasing")
            x_spacing_px = float(np.median(differences))
        if x_spacing_px <= 0:
            raise ValueError("ASTRA X-grid picks must have positive spacing")
        return picked_grid, x_spacing_px, picked_grid.copy()

    y_lo = min(x_band[0], H - 2)
    y_hi = min(x_band[1], H)

    if y_hi <= y_lo:
        guides = np.array([mb_x + k * x_spacing_px for k in range(n_major)
                           if 0 < mb_x + k * x_spacing_px < W])
        _g = guides.astype(float)
        return _g, x_spacing_px, _g.copy()

    below      = frame[y_lo:y_hi, :]
    col_mean   = np.mean(below, axis=0)
    col_smooth = _gauss_smooth(col_mean, gauss_sigma)

    # -- Find the first grid tick -------------------------------------------
    # The tick marks appear as bright columns in the below-baseline band.
    #
    # Priority A: use clicked x_grid guide positions (most accurate).
    # The user may click on any major division; back-calculate first_x from it.
    # If 2+ clicks are provided, also re-measure x_spacing_px from their spacing —
    # this corrects for per-TIFF oscilloscope time-base differences that would
    # otherwise cause progressive drift (each tick adds the per-division error).
    #
    # Priority B: grid phase search — NO reliance on mb_x.
    # Film frame widths vary significantly between frames (e.g. 2595–3250 px in
    # F127), so the CRT display starts at different frame-relative positions.
    # mb_x also jitters ±10–30 px due to detection noise.  Instead, we scan all
    # possible grid phases x0 in [0, x_spacing_px) and pick the one that maximises
    # the mean of col_smooth at {x0, x0+sp, x0+2sp, …}.  This is robust to any
    # per-frame CRT registration variation or mb_x inconsistency.
    if guide_x_grid:
        # Priority A
        x_sorted = sorted(float(v) for v in guide_x_grid)
        if len(x_sorted) >= 2:
            # Use ALL guide clicks, not only the first and last.  Clicks may skip
            # major divisions, so infer integer division indices from the incoming
            # spacing estimate and fit x = intercept + spacing * division_index.
            x0 = x_sorted[0]
            k_click = np.array([round((x - x0) / x_spacing_px) for x in x_sorted], dtype=float)
            if len(np.unique(k_click)) >= 2:
                x_spacing_px = abs(float(np.polyfit(k_click, np.array(x_sorted), 1)[0]))
        x_seed      = x_sorted[0]
        k_seed      = max(0, round((x_seed - mb_x) / x_spacing_px))
        first_guess = x_seed - k_seed * x_spacing_px
        snap_lo = max(0, int(first_guess) - 15)
        snap_hi = min(W, int(first_guess) + 16)
        if snap_hi > snap_lo:
            first_x = float(snap_lo + np.argmax(col_smooth[snap_lo:snap_hi]))
        else:
            first_x = first_guess
    elif d_from_mb is not None:
        # Priority A2: D_anchor — constant distance from mb_x to a known major tick.
        #
        # Physical basis: the oscilloscope sweep is triggered from the radar transmit
        # pulse, so all waveform content is time-locked to the graticule.  D = tick_x
        # - mb_x is constant across frames within a TIFF (same oscilloscope time base).
        #
        # Strategy: predict the anchor tick's frame-relative position as
        # anchor_pred = mb_x + d_from_mb, snap ±half_search around it to find the
        # actual tick, then back-calculate first_x = anchor_x - k_anchor * spacing.
        # The anchor lands in the right portion of the frame (past the bright left-side
        # artifact), giving a reliable snap even when major ticks are faint.
        k_anchor    = max(0, round(d_from_mb / x_spacing_px))
        anchor_pred = float(mb_x) + d_from_mb
        a_lo        = max(0, int(anchor_pred) - half_search)
        a_hi        = min(W, int(anchor_pred) + half_search + 1)
        if a_hi > a_lo:
            anchor_x = float(a_lo + np.argmax(col_smooth[a_lo:a_hi]))
        else:
            anchor_x = anchor_pred
        first_x = anchor_x - k_anchor * x_spacing_px
    else:
        # Priority B: grid phase search — right-portion only.
        #
        # The left side of every frame (~first 35% of width) contains a large
        # bright artifact in the below-baseline band (from the inter-frame
        # boundary, CRT unblank, and main-bang region bleeding downward).
        # Using the full frame causes the phase search to lock onto that bright
        # region rather than the faint but periodic graticule tick marks.
        #
        # Fix: score only positions in the right 65% of the frame (x > W*0.35).
        # This region is after the main bang and early echoes, where the trace
        # sits at the noise floor and the only periodic features are the CRT
        # graticule ticks.  Typically >=8 major ticks remain in this region —
        # sufficient for reliable phase estimation.
        #
        # Coarse sweep over all phases (~200 steps), then 0.1 px fine refinement.
        step     = max(0.5, x_spacing_px / 200.0)
        k_arr    = np.arange(n_major, dtype=float)
        W_sm     = len(col_smooth)
        x_start  = 200                         # exclude only inter-frame border (~0-200 px)
        min_ticks = 4                          # require at least this many valid ticks
        best_score = -np.inf
        best_x0    = 0.0
        for x0 in np.arange(0.0, x_spacing_px, step):
            idxs  = np.round(x0 + k_arr * x_spacing_px).astype(int)
            valid = (idxs >= x_start) & (idxs < W_sm)
            if valid.sum() >= min_ticks:
                s = col_smooth[idxs[valid]].mean()
                if s > best_score:
                    best_score = s
                    best_x0    = x0
        # Fine refinement ±2 coarse steps around best_x0
        for x0 in np.arange(max(0.0,          best_x0 - 2 * step),
                             min(x_spacing_px, best_x0 + 2 * step + 0.01), 0.1):
            idxs  = np.round(x0 + k_arr * x_spacing_px).astype(int)
            valid = (idxs >= x_start) & (idxs < W_sm)
            if valid.sum() >= min_ticks:
                s = col_smooth[idxs[valid]].mean()
                if s > best_score:
                    best_score = s
                    best_x0    = x0
        first_x = float(best_x0)

    # Guide positions anchored at first_x
    guides = []
    for k in range(n_major):
        gx = first_x + k * x_spacing_px
        if 0 < gx < W:
            guides.append(gx)
    guides = np.array(guides)

    if len(guides) < 2:
        _g = guides.astype(float)
        return _g, x_spacing_px, _g.copy()

    # Raw detection: ±half_search snap around each guide (accounts for spacing drift)
    raw_x = np.empty(len(guides), dtype=float)
    # k=0 already resolved above (first_x); still snap to refine sub-pixel position
    for i, g in enumerate(guides):
        lo = max(0, int(g) - half_search)
        hi = min(W, int(g) + half_search)
        if hi > lo:
            raw_x[i] = lo + float(np.argmax(col_smooth[lo:hi]))
        else:
            raw_x[i] = g

    # Initial linear fit across all guides
    k_all   = np.arange(len(raw_x), dtype=float)
    c_all   = np.polyfit(k_all, raw_x, 1)
    resid   = raw_x - np.polyval(c_all, k_all)

    # Inlier-only refit (validated approach)
    inliers = np.abs(resid) <= inlier_thresh_px
    if inliers.sum() >= 2:
        c_in    = np.polyfit(k_all[inliers], raw_x[inliers], 1)
        spacing = abs(float(c_in[0]))
    else:
        c_in    = c_all
        spacing = abs(float(c_all[0]))

    x_fitted = np.polyval(c_in, k_all)
    return x_fitted, spacing, raw_x


# -- Per-frame calibration ------------------------------------------------------

def detect_frame_calibration(
        frame: np.ndarray,
        default_cal: dict,
        guides: dict | None = None,
) -> dict:
    """
    Per-frame calibration using validated algorithms.

    Steps
    -----
    A. Main bang (mb_x): use an approved position when supplied; otherwise
       detect automatically, optionally with a search guide.

    B. Y reference (y_ref_px, -60 dB): pre-main-bang trace median sits at
       the system noise floor = the -60 dB reference line.

    C. X grid lines: column-mean argmax in below-baseline band y=[1550,1800].
       Grid tick marks are bright; we fit an inlier-only linear model
       (detect_gridlines3.py).

    D. Y grid lines: use ASTRA-approved coordinates when provided. Otherwise,
       extrapolate from y_ref_px + k × y_spacing_px because the horizontal
       graticule lines are too faint to detect algorithmically.

    Parameters
    ----------
    frame       : (H, W) float32 array [0, 1], already cropped to this frame
    default_cal : baseline calibration dict (DEFAULT_CAL or previous per-frame)
    guides      : optional ASTRA calibration picks or legacy guides, with keys:
                    'mb'          – int, optional main bang x (frame-rel)
                    'ref'         – int, approximate y_ref_px row
                    'x_grid'      – list of approved X-grid coordinates
                    'y_grid'      – list of approved Y-grid coordinates
                    'x_ref'       – int, approved X reference coordinate
                    'x_spacing_px'– float, X major-division spacing
                    'y_spacing_px'– float, Y major-division spacing

    Returns
    -------
    cal : updated dict with added keys:
        mb_x           – detected main bang column (frame-relative pixels)
        mb_power_dB    – power at main bang (dB)
        y_ref_px       – noise-floor row = -60 dB anchor (pixels)
        db_per_px      – Y scale (dB/px; from y_spacing_px)
        hgrid_lines    – approved Y grid rows or extrapolated fallback
        hgrid_spacing  – Y major-division spacing used (px)
        xgrid_lines    – list of X grid line cols (detected)
        x_spacing_px   – X major-division spacing (px; from inlier fit)
        cal_source_y   – 'ASTRA picks' when approved Y-grid rows are provided
    """
    guides = guides or {}
    cal    = default_cal.copy()

    # Derive spacing constants: prefer guide, then default_cal, then DEFAULT_CAL
    x_spacing_px = float(guides.get("x_spacing_px",
                         default_cal.get("x_spacing_px",
                         2.0 / default_cal["us_per_px"])))
    y_spacing_px = float(guides.get("y_spacing_px",
                         default_cal.get("y_spacing_px",
                         10.0 / default_cal["db_per_px"])))

    # -- A: Main bang ------------------------------------------------------
    trace_y, trace_y_s, _ = extract_trace(frame, cal)
    guide_mb           = guides.get("mb")

    if guide_mb is not None and guides.get("mb_is_pick"):
        mb_x = int(round(float(guide_mb)))
        if not 0 <= mb_x < frame.shape[1]:
            raise ValueError(
                f"Approved main bang at x={mb_x} is outside frame width "
                f"{frame.shape[1]}; its power cannot be measured from this frame"
            )
    else:
        mb_x = detect_mb(trace_y_s, cal, guide_x=guide_mb,
                          frame=frame)
    # Measure MB power from the UNCONSTRAINED trace (trace_y), not the
    # smoothed trace (trace_y_s).  The smoothed trace deliberately clamps
    # the MB region to the noise floor (to prevent T/R ringing from biasing
    # envelope walks), so trace_y_s[mb_x] reads near NF regardless of true
    # MB power.  This masked attenuator changes and CRT gain shifts.
    # Search a ±10 px window in trace_y for the actual MB peak (minimum Y
    # = maximum power) to handle slight positional offset.
    _mb_win = 10
    _mb_lo = max(0, mb_x - _mb_win)
    _mb_hi = min(len(trace_y), mb_x + _mb_win + 1)
    _mb_best_y = float(np.min(trace_y[_mb_lo:_mb_hi]))
    mb_power_dB        = float(px_to_db(_mb_best_y, cal))

    cal["mb_x"]        = mb_x
    cal["mb_power_dB"] = mb_power_dB

    # -- B: Y reference (noise floor = -60 dB) ----------------------------
    guide_ref = guides.get("ref")
    if guide_ref is not None:
        # Guide provided: use it directly (user confirmed the reference line)
        y_ref_px = float(guide_ref)
        if not np.isfinite(y_ref_px) or not 0 <= y_ref_px < frame.shape[0]:
            raise ValueError("ASTRA Y-reference pick falls outside the frame")
    else:
        # No guide: sample from the MIDDLE of the pre-bang region.
        # Avoid frame-boundary artifacts (cols 0–30) and the main bang's
        # leading edge (last ~50 px before mb_x).
        # Safe window: mb_x//4 -> mb_x//2, clamped away from the bang.
        pre_lo = max(30, mb_x // 4)
        pre_hi = min(max(pre_lo + 10, mb_x // 2), max(pre_lo + 5, mb_x - 50))
        if pre_hi > pre_lo + 5:
            pre_trace = trace_y_s[pre_lo:pre_hi]
            y_ref_px  = float(np.percentile(pre_trace, 75))
            # Sanity: must lie within the oscilloscope display band
            if not (cal["y_disp_lo"] < y_ref_px < cal["y_disp_hi"]):
                warnings.warn(
                    f"y_ref estimate {y_ref_px:.0f} px outside display band "
                    f"[{cal['y_disp_lo']}, {cal['y_disp_hi']}] — using default")
                y_ref_px = float(default_cal["y_ref_px"])
        else:
            y_ref_px = float(default_cal["y_ref_px"])

    cal["y_ref_px"] = y_ref_px
    cal["y_ref_db"] = -60.0

    # -- C: X grid lines (argmax in below-baseline band) ------------------
    x_fitted, x_spacing_detected, x_raw = detect_xgrid_lines(
        frame, mb_x, x_spacing_px, cal,
        guide_x_grid=guides.get("x_grid"),
        d_from_mb=guides.get("d_from_mb"))
    cal["xgrid_lines"]      = x_fitted.tolist()
    cal["xgrid_raw_lines"]  = x_raw.tolist()
    cal["x_spacing_px"]     = x_spacing_detected

    # -- D: Y grid lines --------------------------------------------------
    if guides.get("y_grid"):
        hgrid_lines = sorted(float(v) for v in guides["y_grid"])
        if (not np.all(np.isfinite(hgrid_lines))
                or any(y < 0 or y >= frame.shape[0] for y in hgrid_lines)):
            raise ValueError("ASTRA Y-grid picks fall outside the frame")
        if len(hgrid_lines) >= 2:
            differences = np.diff(hgrid_lines)
            if np.any(differences <= 0):
                raise ValueError("ASTRA Y-grid picks must be distinct and increasing")
            y_spacing_px = float(np.median(differences))
        if y_spacing_px <= 0:
            raise ValueError("ASTRA Y-grid picks must have positive spacing")
    else:
        hgrid_lines = compute_y_grid_lines(
            y_ref_px, y_spacing_px,
            cal["y_disp_lo"], cal["y_disp_hi"])
    cal["hgrid_lines"]  = hgrid_lines
    cal["hgrid_spacing"] = y_spacing_px
    cal["db_per_px"]     = 10.0 / y_spacing_px
    cal["cal_source_y"]  = "ASTRA picks" if guides.get("y_grid") else "extrapolated"
    if guides.get("mb_is_pick"):
        red_trace = astra_trace_with_fallback(
            frame, trace_y, guides.get("trace_context", "Phase 3 main bang"),
            guides.get("crop_start_y", ASTRA_DEFAULT_CROP_START_Y),
            guides.get("crop_end_y", ASTRA_DEFAULT_CROP_END_Y))
        cal["astra_trace"] = red_trace
        mb_row = astra_peak_row(red_trace, mb_x, guides.get("y_grid", []),
                                guides.get("trace_context", "Phase 3 main bang"))
        cal["mb_power_dB"] = float(px_to_db(mb_row, cal))
    if guides.get("x_ref") is not None:
        x_ref_px = int(guides["x_ref"])
        if not 0 <= x_ref_px < frame.shape[1]:
            raise ValueError("ASTRA X-reference pick falls outside the frame")
        cal["x_ref_px"] = x_ref_px

    return cal


# -- Main entry: process one TIFF ----------------------------------------------

def process_tiff(tiff_path: str | Path,
                 cal: dict | None = None,
                 frame_bounds: list[tuple[int, int]] | None = None,
                 verbose: bool = True) -> list[LyraFrame]:
    """
    Process all A-scope frames in a TIFF and extract waveform metrics.

    Parameters
    ----------
    tiff_path    : path to raw A-scope TIFF
    cal          : calibration dict (uses DEFAULT_CAL if None)
    frame_bounds : pre-computed (left, right) boundaries; auto-detected if None
    verbose      : print per-frame progress

    Returns
    -------
    list of LyraFrame, one per detected frame
    """
    if cal is None:
        cal = DEFAULT_CAL.copy()
    tiff_path = Path(tiff_path)

    if verbose:
        print(f"\nLYRA v1.0 — processing {tiff_path.name}")
        print("-" * 70)

    Image.MAX_IMAGE_PIXELS = None
    img      = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W     = img_norm.shape

    if verbose:
        print(f"  Image : {W} × {H} px")

    if frame_bounds is None:
        frame_bounds = detect_frames(img_norm)
    if verbose:
        print(f"  Frames detected : {len(frame_bounds)}")
        print()

    results: list[LyraFrame] = []

    for idx, (left, right) in enumerate(frame_bounds):
        frame   = img_norm[:, left : right + 1]
        frame_w = frame.shape[1]
        lf      = LyraFrame(
            frame_idx=idx, frame_left=left, frame_right=right, frame_w=frame_w)

        try:
            trace_y, trace_y_s, _ = extract_trace(frame, cal)
            mb_x               = detect_mb(trace_y_s, cal, frame=frame)
            lf.mb_x            = mb_x
            lf.noise_floor_dB  = estimate_noise_floor(trace_y_s, mb_x, cal)

            surface_x, bed_x = detect_echoes(
                trace_y_s, mb_x, lf.noise_floor_dB, cal)

            if surface_x is not None:
                lf.surface          = compute_echo_metrics(
                    trace_y_s, surface_x, lf.noise_floor_dB, mb_x, frame_w, cal)
                lf.surface_detected = True
            else:
                lf.quality = "no_surface"

            if bed_x is not None:
                lf.bed          = compute_echo_metrics(
                    trace_y_s, bed_x, lf.noise_floor_dB, mb_x, frame_w, cal)
                lf.bed_detected = True
            else:
                if lf.quality == "ok":
                    lf.quality = "no_bed"

        except Exception as exc:
            warnings.warn(f"Frame {idx}: {exc}")
            lf.quality = "failed"

        results.append(lf)

        if verbose:
            s = f"  [{idx:2d}] MB={lf.mb_x:4d}px  NF={lf.noise_floor_dB:+.1f} dB"
            if lf.surface_detected:
                s += (f"  | surf {lf.surface.peak_twt_us:.2f}µs"
                      f" {lf.surface.peak_power_dB:+.1f}dB"
                      f" SNR={lf.surface.peak_snr_dB:.1f}dB")
            if lf.bed_detected:
                s += (f"  | bed {lf.bed.peak_twt_us:.2f}µs"
                      f" {lf.bed.peak_power_dB:+.1f}dB"
                      f" SNR={lf.bed.peak_snr_dB:.1f}dB"
                      f"  peak={lf.bed.peakiness:.2f}"
                      f"  asym={lf.bed.asymmetry:.2f}")
            if lf.quality != "ok":
                s += f"  [{lf.quality}]"
            print(s)

    return results


# -- DataFrame serialisation ----------------------------------------------------

def to_dataframe(results: list[LyraFrame],
                 tiff_path: str | Path | None = None) -> pd.DataFrame:
    """
    Flatten list of LyraFrame into a tidy DataFrame (one row per frame).

    Column naming convention: {echo}_{metric}  (e.g. bed_peakiness)
    """
    tiff_name = Path(tiff_path).name if tiff_path else ""
    rows = []

    for lf in results:
        row: dict = dict(
            tiff             = tiff_name,
            frame_idx        = lf.frame_idx,
            frame_left       = lf.frame_left,
            frame_right      = lf.frame_right,
            frame_w          = lf.frame_w,
            mb_x             = lf.mb_x,
            noise_floor_dB   = lf.noise_floor_dB,
            surface_detected = lf.surface_detected,
            bed_detected     = lf.bed_detected,
            quality          = lf.quality,
        )

        # Echo columns with echo prefix
        for prefix, echo in [("surface", lf.surface), ("bed", lf.bed)]:
            for k, v in asdict(echo).items():
                row[f"{prefix}_{k}"] = v

        # Derived geometry (physical distances)
        if lf.surface_detected:
            row["h_air_m"] = lf.surface.peak_twt_us / 2.0 * C_AIR_M_PER_US
        else:
            row["h_air_m"] = np.nan

        if lf.surface_detected and lf.bed_detected:
            twt_ice        = lf.bed.peak_twt_us - lf.surface.peak_twt_us
            row["h_ice_m"] = twt_ice / 2.0 * C_ICE_M_PER_US
        else:
            row["h_ice_m"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# Navigation & CBD Alignment
# ===========================================================================
#
# Maps LYRA CBD numbers to geographic positions by cross-correlating
# LYRA h_ice against Bingham ICE_THICKN along the Stanford nav track.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NAV_DIR = navigation_root()
_BINGHAM_DIR = _NAV_DIR / "Bingham" / "002_nav_data_rgb_corrected"
_RIGGS_CSV = _REPO_ROOT / "Data" / "RIGGS" / "riggs_stations.csv"


def load_stanford_nav(flt: int | str) -> pd.DataFrame:
    """Load Stanford navigation CSV for a flight.

    Returns DataFrame with columns: CBD, LAT, LON, THK, SRF.
    CBD is always equal to the 0-based row index.
    THK/SRF = 9999 means missing.
    """
    flt = str(flt).lstrip("Ff")
    nav_dir = navigation_root(_NAV_DIR)
    candidates = [nav_dir / f"{flt}.csv"]
    if "LYRA_NAV_DIR" not in os.environ:
        candidates.extend([_REPO_ROOT / "Navigation_Files" / "Stanford" / f"{flt}.csv",
                           _REPO_ROOT / "Navigation_Files" / f"{flt}.csv"])
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            df.columns = df.columns.str.strip()
            return df
    raise FileNotFoundError(
        f"Stanford nav not found for F{flt}: tried "
        f"looked for {_NAV_DIR / f'{flt}.csv'}")


def load_bingham_ice_thickness(flt: int | str) -> pd.DataFrame | None:
    """Load Bingham corrected TXT and return ICE_THICKN indexed by Stanford CBD.

    Steps:
      1. Load Bingham TXT (handles F128 special column format).
      2. Filter to rows with valid ICE_THICKN > 0.
      3. Spatially match each Bingham row to the nearest Stanford nav row.
      4. Return DataFrame with columns: stanford_cbd, ice_thickn, lat, lon, match_km.

    Returns None if the Bingham file doesn't exist or has no valid ICE_THICKN.
    """
    from pyproj import Transformer

    flt_str = str(flt).lstrip("Ff")
    txt_path = _BINGHAM_DIR / f"sprinsftud_1974_{flt_str}_nav.txt"
    if not txt_path.exists():
        return None

    bingham = pd.read_csv(txt_path, sep="\t")
    bingham.columns = bingham.columns.str.strip()

    # Normalize column names (F128 uses different headers)
    if "ICE_THICKN" not in bingham.columns:
        return None

    valid = bingham["ICE_THICKN"].notna() & (bingham["ICE_THICKN"] > 0)
    bingham = bingham[valid].copy()
    if len(bingham) == 0:
        return None

    # Load Stanford nav for spatial matching
    try:
        stanford = load_stanford_nav(flt_str)
    except FileNotFoundError:
        return None

    # Project to EPSG:3031 for spatial matching
    T = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    from scipy.spatial import cKDTree

    st_x, st_y = T.transform(stanford["LON"].values, stanford["LAT"].values)
    st_xy = np.column_stack([st_x, st_y])

    bg_x, bg_y = T.transform(bingham["LONGITUDE"].values, bingham["LATITUDE"].values)
    bg_xy = np.column_stack([bg_x, bg_y])

    tree = cKDTree(st_xy)
    dists, idxs = tree.query(bg_xy)
    dists_km = dists / 1000.0

    result = pd.DataFrame({
        "stanford_cbd": stanford["CBD"].values[idxs],
        "ice_thickn":   bingham["ICE_THICKN"].values,
        "lat":          bingham["LATITUDE"].values,
        "lon":          bingham["LONGITUDE"].values,
        "match_km":     dists_km,
    })

    # Warn if any matches are far
    far = (result["match_km"] > 5.0).sum()
    if far > 0:
        warnings.warn(f"F{flt_str}: {far}/{len(result)} Bingham points > 5 km "
                      f"from nearest Stanford nav point")

    return result


@dataclass
class AlignmentResult:
    """Result of CBD-to-navigation cross-correlation alignment."""
    flight: int
    offset: int                    # nav_row = lyra_cbd + offset
    correlation: float             # peak Pearson r (best anchor or global)
    n_matched: int                 # LYRA good frames used in best match
    n_reference: int               # total Bingham ICE_THICKN values
    search_range: tuple            # (min_offset, max_offset) searched
    second_best_corr: float        # for confidence assessment
    confidence: str                # "high" / "medium" / "low"
    riggs_validation: dict | None  # RIGGS cross-check results
    n_anchor_windows: int = 0     # number of agreeing anchor windows
    method: str = "global"         # "windowed" or "global" (fallback)


# ---- Rose 1978 Navigation Fix Data (Table 2.2, Table 2.3) ----------------

ROSE_STATIONS = {
    "mcmurdo_skiway":    {"lat": -77.8750, "lon": 166.9133, "alt_m": 10},
    "mcmurdo_ice":       {"lat": -77.8817, "lon": 166.7117, "alt_m": 6},
    "byrd":              {"lat": -80.0103, "lon": -119.5021, "alt_m": 1545},
    "south_pole":        {"lat": -89.9954, "lon": 144.3411, "alt_m": 2830},
    "roosevelt_island":  {"lat": -80.1917, "lon": -161.5617, "alt_m": 80},
}

_ROSE_TABLE_PATH = _REPO_ROOT / "docs" / "ross_table2-3.xlsx"

# Stanford nav files use McMurdo Ice (not Skiway) as their McMurdo reference.
# When comparing Stanford positions to known stations, use these coordinates.
_STANFORD_NAV_STATIONS = {
    "mcmurdo_skiway": {"lat": -77.8817, "lon": 166.7117},  # Stanford uses McMurdo Ice coords
    "mcmurdo_ice":    {"lat": -77.8817, "lon": 166.7117},
    "byrd":           {"lat": -80.0103, "lon": -119.5021},
    "south_pole":     {"lat": -89.9954, "lon": 144.3411},
    "roosevelt_island": {"lat": -80.1917, "lon": -161.5617},
}


def _parse_fix_station(note: str) -> str | None:
    """Extract ROSE_STATIONS key from a Rose Table 2.3 note string.

    Only assigns station coordinates for fixes where the aircraft is AT or
    OVER a known station (overhead passes, terminals). Oblique photos are
    taken from a distance, so the aircraft position != station position.
    """
    note_lower = note.lower()
    is_oblique = "oblique" in note_lower or "oblqiue" in note_lower
    is_overhead = "overhead" in note_lower or "ovehead" in note_lower
    is_terminal = "terminal" in note_lower

    # Oblique photos: aircraft is NOT at the station
    if is_oblique:
        return None

    # Overhead passes: aircraft IS at the station
    if is_overhead:
        if "roosevelt" in note_lower:
            return "roosevelt_island"
        if "byrd" in note_lower:
            return "byrd"
        if "south pole" in note_lower:
            return "south_pole"
        return None

    # Terminal fixes
    if is_terminal:
        # Strip "originated at ..." context — describes origin, not terminal
        # e.g., "terminal (this flight originated at Pole)" = McMurdo terminal
        term_note = note_lower
        for origin_phrase in ["originated at pole", "originated at byrd",
                              "orignated at byrd", "orignated at pole"]:
            term_note = term_note.replace(origin_phrase, "")

        if "roosevelt" in term_note:
            return "roosevelt_island"
        if "byrd" in term_note:
            return "byrd"
        if "south pole" in term_note or "pole" in term_note:
            return "south_pole"
        # Bare "terminal" = McMurdo (resolved by caller)
        return None

    return None


def _parse_flight_origin(fixes: list[tuple]) -> str:
    """Determine flight origin station from fix notes."""
    for _, _, note in fixes:
        if "originated at pole" in note.lower():
            return "south_pole"
        if "originated at byrd" in note.lower() or "orignated at byrd" in note.lower():
            return "byrd"
    return "mcmurdo_skiway"


def load_rose_fixes() -> dict:
    """Load Rose 1978 Table 2.3 from the digitized Excel file.

    Returns dict mapping flight number -> list of (time_minutes, vector_km,
    description, station_key_or_None).

    station_key is a ROSE_STATIONS key for fixes with known coordinates
    (terminal/overhead), or None for oblique photos.
    """
    import pandas as pd

    if not _ROSE_TABLE_PATH.exists():
        warnings.warn(f"Rose Table 2.3 not found: {_ROSE_TABLE_PATH}")
        return {}

    df = pd.read_excel(_ROSE_TABLE_PATH, skiprows=3, header=None,
                       names=["flight", "h", "min", "vector_km", "note"])
    df["flight"] = df["flight"].ffill().astype(int)

    result = {}
    for flt, grp in df.groupby("flight"):
        fixes_raw = []
        for _, row in grp.iterrows():
            time_min = int(row["h"]) * 60 + int(row["min"])
            note = str(row["note"]).strip() if pd.notna(row["note"]) else ""
            fixes_raw.append((time_min, float(row["vector_km"]), note))

        origin = _parse_flight_origin(fixes_raw)

        fixes = []
        for time_min, vector_km, note in fixes_raw:
            station = _parse_fix_station(note)
            # Resolve bare "terminal" to flight origin/destination
            if station is None and "terminal" in note.lower():
                # Terminal = origin for originated-at flights, McMurdo otherwise
                # But terminal is at the END of the flight, so:
                # - flights originating at McMurdo terminate at McMurdo
                #   UNLESS the note says otherwise (e.g., "terminal, Byrd")
                # - flights originating at Byrd/Pole terminate at McMurdo
                #   (all non-McMurdo flights return to McMurdo)
                if origin != "mcmurdo_skiway":
                    station = "mcmurdo_skiway"
                else:
                    station = "mcmurdo_skiway"
            fixes.append((time_min, vector_km, note, station))

        result[flt] = fixes

    return result


def load_rose_flight_origins() -> dict:
    """Return dict mapping flight number -> origin station key."""
    fixes_all = load_rose_fixes()
    origins = {}
    for flt, fixes in fixes_all.items():
        raw = [(t, v, n) for t, v, n, _ in fixes]
        origins[flt] = _parse_flight_origin(raw)
    return origins


@dataclass
class NavCorrection:
    """CBD-varying navigation correction for a flight."""
    flight: int
    fix_cbds: list[int]          # CBD indices of fix points
    dx_at_fixes: list[float]     # x correction (m) in EPSG:3031 at each fix
    dy_at_fixes: list[float]     # y correction (m) in EPSG:3031 at each fix
    fix_descriptions: list[str]  # description of each fix
    assessment: str              # from diagnostics

    def correction_at_cbd(self, cbd) -> tuple:
        """Interpolate (dx, dy) correction in EPSG:3031 metres for given CBD(s).

        Returns (dx, dy) arrays in metres.  Add these to the Stanford
        EPSG:3031 coordinates to get corrected positions.
        """
        cbd = np.atleast_1d(np.asarray(cbd, dtype=float))
        dx = np.interp(cbd, self.fix_cbds, self.dx_at_fixes)
        dy = np.interp(cbd, self.fix_cbds, self.dy_at_fixes)
        return dx, dy


def diagnose_nav_fixes(flt: int | str, verbose: bool = True) -> dict:
    """Compare Stanford nav positions at Rose fix points against known positions.

    For fixes with known station coordinates (terminal fixes, overhead passes),
    computes the residual distance between the Stanford nav position and the
    true station position.  This reveals whether the Stanford nav files already
    incorporate Rose's corrections.

    Parameters
    ----------
    flt : flight number (with or without 'F' prefix)
    verbose : print results to console

    Returns
    -------
    dict with keys: flight, origin, n_nav_cbds, n_fixes, n_diagnosed,
    fixes (list of per-fix dicts), assessment.
    """
    from pyproj import Transformer

    flt_str = str(flt).lstrip("Ff")
    flt_int = int(flt_str)

    all_fixes = load_rose_fixes()
    if flt_int not in all_fixes:
        if verbose:
            print(f"F{flt_str}: no Rose fix data available")
        return {"flight": flt_int, "n_fixes": 0, "assessment": "no_data"}

    fixes = all_fixes[flt_int]
    origins = load_rose_flight_origins()
    origin = origins.get(flt_int, "mcmurdo_skiway")

    try:
        nav = load_stanford_nav(flt_str)
    except FileNotFoundError:
        if verbose:
            print(f"F{flt_str}: Stanford nav file not found")
        return {"flight": flt_int, "n_fixes": len(fixes),
                "assessment": "no_nav"}

    n_nav = len(nav)
    T = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    results = []
    for time_min, vector_km, desc, station_key in fixes:
        # Convert time to approximate CBD index (20 s per CBD = 3 CBDs/min)
        cbd_approx = int(round(time_min * 3))
        cbd_clamped = min(max(cbd_approx, 0), n_nav - 1)

        # For terminal fixes, use the actual last CBD of the nav file
        is_terminal = "terminal" in desc.lower()
        if is_terminal:
            cbd_clamped = n_nav - 1

        nav_lat = float(nav.iloc[cbd_clamped]["LAT"])
        nav_lon = float(nav.iloc[cbd_clamped]["LON"])

        fix_result = {
            "time_min": time_min,
            "vector_km": vector_km,
            "description": desc,
            "station": station_key,
            "nav_cbd": cbd_clamped,
            "nav_lat": round(nav_lat, 6),
            "nav_lon": round(nav_lon, 6),
            "true_lat": None,
            "true_lon": None,
            "residual_km": None,
        }

        if station_key is not None and station_key in _STANFORD_NAV_STATIONS:
            nx, ny = T.transform(nav_lon, nav_lat)

            # For McMurdo terminals, check both Ice and Skiway references
            if station_key in ("mcmurdo_skiway", "mcmurdo_ice"):
                ice = ROSE_STATIONS["mcmurdo_ice"]
                sky = ROSE_STATIONS["mcmurdo_skiway"]
                ix, iy = T.transform(ice["lon"], ice["lat"])
                sx, sy = T.transform(sky["lon"], sky["lat"])
                d_ice = np.sqrt((nx - ix)**2 + (ny - iy)**2) / 1000.0
                d_sky = np.sqrt((nx - sx)**2 + (ny - sy)**2) / 1000.0
                if d_ice < d_sky:
                    true = ice
                    dist_km = d_ice
                    fix_result["mcmurdo_ref"] = "mcmurdo_ice"
                else:
                    true = sky
                    dist_km = d_sky
                    fix_result["mcmurdo_ref"] = "mcmurdo_skiway"
            else:
                true = _STANFORD_NAV_STATIONS[station_key]
                tx, ty = T.transform(true["lon"], true["lat"])
                dist_km = np.sqrt((nx - tx)**2 + (ny - ty)**2) / 1000.0

            fix_result["true_lat"] = true["lat"]
            fix_result["true_lon"] = true["lon"]
            fix_result["residual_km"] = round(dist_km, 2)

        results.append(fix_result)

    # Assessment
    diagnosed = [r for r in results if r["residual_km"] is not None]
    n_diagnosed = len(diagnosed)

    assessment = "inconclusive"
    if n_diagnosed > 0:
        residuals = [r["residual_km"] for r in diagnosed]
        vectors = [r["vector_km"] for r in diagnosed]
        mean_residual = np.mean(residuals)
        mean_vector = np.mean(vectors)

        if mean_residual < 2.0:
            assessment = "likely_corrected"
        elif mean_residual > 0.5 * mean_vector:
            assessment = "likely_uncorrected"
        else:
            assessment = "partially_corrected"

    result = {
        "flight": flt_int,
        "origin": origin,
        "n_nav_cbds": n_nav,
        "n_fixes": len(fixes),
        "n_diagnosed": n_diagnosed,
        "fixes": results,
        "assessment": assessment,
    }

    if verbose:
        print(f"\nF{flt_str} Navigation Fix Diagnostics")
        print(f"  Origin: {origin} | Nav CBDs: {n_nav} | "
              f"Fixes: {len(fixes)} ({n_diagnosed} with known coords)")
        print(f"  {'Time':>6s}  {'CBD':>5s}  {'Rose km':>8s}  "
              f"{'Residual':>9s}  Description")
        print(f"  {'----':>6s}  {'---':>5s}  {'-------':>8s}  "
              f"{'--------':>9s}  -----------")
        for r in results:
            resid_str = (f"{r['residual_km']:8.2f}" if r["residual_km"] is not None
                         else "     N/A")
            mc_ref = r.get("mcmurdo_ref", "")
            mc_tag = f" [{mc_ref}]" if mc_ref else ""
            t = r["time_min"]
            print(f"  {t//60:2d}h{t%60:02d}m  {r['nav_cbd']:5d}  "
                  f"{r['vector_km']:8.1f}  {resid_str}  "
                  f"{r['description']}{mc_tag}")
        print(f"\n  Assessment: {assessment}")

    return result


def diagnose_all_nav_fixes(verbose: bool = True) -> dict:
    """Run diagnose_nav_fixes for all flights with Rose fix data.

    Returns dict mapping flight number to diagnostic result.
    """
    all_fixes = load_rose_fixes()
    all_results = {}
    for flt in sorted(all_fixes.keys()):
        try:
            r = diagnose_nav_fixes(flt, verbose=verbose)
            all_results[flt] = r
        except Exception as e:
            if verbose:
                print(f"F{flt}: ERROR - {e}")
            all_results[flt] = {"flight": flt, "assessment": "error",
                                "error": str(e)}
        if verbose:
            print()
    return all_results


def build_nav_correction(flt: int | str) -> NavCorrection | None:
    """Build a CBD-varying position correction from Rose fix diagnostics.

    Computes the error vector (true minus Stanford) at each fix point with
    known coordinates, then builds a piecewise-linear correction that can
    be interpolated at any CBD.

    Returns None if no fixes with known coordinates exist.
    """
    from pyproj import Transformer

    diag = diagnose_nav_fixes(flt, verbose=False)
    if diag.get("assessment") in ("no_data", "no_nav", "error"):
        return None

    flt_int = diag["flight"]
    origin = diag["origin"]

    T = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    nav = load_stanford_nav(flt_int)

    fix_cbds = []
    dx_list = []
    dy_list = []
    descs = []

    # Determine which McMurdo reference this flight's nav file uses
    # by checking the terminal fix from the diagnostic
    mcmurdo_ref = "mcmurdo_skiway"  # default
    for fix in diag["fixes"]:
        if fix.get("mcmurdo_ref"):
            mcmurdo_ref = fix["mcmurdo_ref"]
            break

    def _get_true_coords(station_key):
        """Get true coordinates, using the correct McMurdo reference."""
        if station_key in ("mcmurdo_skiway", "mcmurdo_ice"):
            return ROSE_STATIONS[mcmurdo_ref]
        return ROSE_STATIONS[station_key]

    # Add origin as first fix point (CBD 0 = known station)
    # Only if CBD 0 is actually near the origin station (< 10 km)
    if origin in _STANFORD_NAV_STATIONS:
        true = _get_true_coords(origin)
        nav_lat = float(nav.iloc[0]["LAT"])
        nav_lon = float(nav.iloc[0]["LON"])
        nx, ny = T.transform(nav_lon, nav_lat)
        tx, ty = T.transform(true["lon"], true["lat"])
        origin_dist_km = np.sqrt((nx - tx)**2 + (ny - ty)**2) / 1000.0
        if origin_dist_km < 10.0:
            fix_cbds.append(0)
            dx_list.append(tx - nx)
            dy_list.append(ty - ny)
            descs.append(f"origin: {origin} [{mcmurdo_ref}]")

    # Add diagnosed fixes with known station coords
    for fix in diag["fixes"]:
        if fix["residual_km"] is None or fix["station"] is None:
            continue
        cbd = fix["nav_cbd"]
        true = _get_true_coords(fix["station"])
        nx, ny = T.transform(fix["nav_lon"], fix["nav_lat"])
        tx, ty = T.transform(true["lon"], true["lat"])
        fix_cbds.append(cbd)
        dx_list.append(tx - nx)
        dy_list.append(ty - ny)
        descs.append(fix["description"])

    if len(fix_cbds) < 1:
        return None

    # Deduplicate: keep last entry for each CBD
    seen = {}
    for i, cbd in enumerate(fix_cbds):
        seen[cbd] = i
    unique_idxs = sorted(seen.values(), key=lambda i: fix_cbds[i])
    fix_cbds = [fix_cbds[i] for i in unique_idxs]
    dx_list = [dx_list[i] for i in unique_idxs]
    dy_list = [dy_list[i] for i in unique_idxs]
    descs = [descs[i] for i in unique_idxs]

    return NavCorrection(
        flight=flt_int,
        fix_cbds=fix_cbds,
        dx_at_fixes=dx_list,
        dy_at_fixes=dy_list,
        fix_descriptions=descs,
        assessment=diag["assessment"],
    )


def save_nav_correction(corr: NavCorrection, out_dir: Path) -> Path:
    """Save NavCorrection to JSON."""
    import json
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    out_path = val_dir / f"F{corr.flight}_nav_correction.json"
    d = {
        "flight": corr.flight,
        "fix_cbds": corr.fix_cbds,
        "dx_at_fixes": [round(x, 1) for x in corr.dx_at_fixes],
        "dy_at_fixes": [round(y, 1) for y in corr.dy_at_fixes],
        "fix_descriptions": corr.fix_descriptions,
        "assessment": corr.assessment,
    }
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    return out_path


def load_nav_correction(flt: int | str,
                        out_dir: Path | None = None) -> NavCorrection | None:
    """Load a previously saved NavCorrection from JSON."""
    import json
    flt_str = str(flt).lstrip("Ff")
    if out_dir is None:
        out_dir = output_root() / f"F{flt_str}"
    json_path = out_dir / "validation" / f"F{flt_str}_nav_correction.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return NavCorrection(**d)


def _xcorr_at_offset(lyra_cbd, lyra_hice, ref_lookup, offset, min_pairs=10):
    """Compute Pearson r for a single offset. Returns (r, n_pairs)."""
    shifted = lyra_cbd + offset
    pl, pr = [], []
    for i, sc in enumerate(shifted):
        if sc in ref_lookup and np.isfinite(lyra_hice[i]):
            pl.append(lyra_hice[i])
            pr.append(ref_lookup[sc])
    n = len(pl)
    if n < min_pairs:
        return -9999.0, n
    pl, pr = np.array(pl), np.array(pr)
    if np.std(pl) < 1e-6 or np.std(pr) < 1e-6:
        return 0.0, n
    return float(np.corrcoef(pl, pr)[0, 1]), n


def _windowed_xcorr(
    lyra_cbd: np.ndarray,
    lyra_hice: np.ndarray,
    ref_lookup: dict,
    search_range: tuple = (-200, 200),
    window_size: int = 40,
    window_step: int = 10,
    min_std_m: float = 30.0,
    min_pairs: int = 10,
    min_r: float = 0.7,
) -> list[dict]:
    """Find anchor windows where local h_ice cross-correlation is strong.

    Slides a window across the sorted LYRA CBD sequence. In each window,
    scans all offsets and keeps windows with high correlation and sufficient
    ice-thickness variation.

    Returns list of anchor dicts sorted by r descending.
    """
    # Sort by CBD
    order = np.argsort(lyra_cbd)
    cbd_s = lyra_cbd[order]
    hice_s = lyra_hice[order]
    n_total = len(cbd_s)

    anchors = []

    for start in range(0, n_total - min_pairs, window_step):
        end = min(start + window_size, n_total)
        w_cbd = cbd_s[start:end]
        w_hice = hice_s[start:end]

        # Skip windows with insufficient terrain variation
        valid = np.isfinite(w_hice)
        if valid.sum() < min_pairs:
            continue
        if np.std(w_hice[valid]) < min_std_m:
            continue

        # Scan offsets for this window
        best_off, best_r, best_n = 0, -9999.0, 0
        second_r = -9999.0

        for off in range(search_range[0], search_range[1] + 1):
            r, n = _xcorr_at_offset(w_cbd, w_hice, ref_lookup, off, min_pairs)
            if r > best_r:
                # Update second-best before overwriting best
                if abs(off - best_off) >= 5:
                    second_r = best_r
                best_off, best_r, best_n = off, r, n
            elif r > second_r and abs(off - best_off) >= 5:
                second_r = r

        if best_r >= min_r and best_n >= min_pairs:
            anchors.append({
                "cbd_center": int(np.median(w_cbd)),
                "offset": best_off,
                "r": round(best_r, 4),
                "n_pairs": best_n,
                "lyra_std": round(float(np.std(w_hice[valid])), 1),
                "second_r": round(second_r, 4) if second_r > -9000 else -1.0,
            })

    # Sort by correlation descending
    anchors.sort(key=lambda a: a["r"], reverse=True)
    return anchors


def align_cbd_to_nav(
    echoes_csv: Path | str | pd.DataFrame,
    flt: int | str,
    search_range: tuple = (-200, 200),
    min_matched: int = 15,
) -> AlignmentResult:
    """Find optimal CBD offset using windowed cross-correlation with track propagation.

    Uses a two-stage approach:
      1. Windowed cross-correlation: slides a 40-CBD window along the LYRA
         h_ice profile, correlating against Bingham ICE_THICKN at each offset.
         Windows with strong terrain variation (std > 30 m) and high correlation
         (r > 0.7) become "anchor windows."
      2. Offset consensus: if multiple anchors agree on the same offset (within
         +/-2), that offset is adopted with high confidence. Even on flights with
         mostly flat ice, a single terrain-rich segment can anchor the entire
         flight thanks to the confirmed 1:1 CBD-to-waypoint ratio.

    Falls back to global correlation and RIGGS tiebreaking when no anchors found.

    The offset means: Stanford nav row index = LYRA CBD + offset.

    Parameters
    ----------
    echoes_csv : path to phase 4 echoes CSV, or a pre-loaded DataFrame
    flt : flight number
    search_range : (min_offset, max_offset) to test
    min_matched : minimum number of matched pairs for global correlation

    Returns
    -------
    AlignmentResult with offset, correlation, confidence, RIGGS validation,
    anchor window count, and method used.
    """
    flt_str = str(flt).lstrip("Ff")
    flt_int = int(flt_str)

    def _empty(conf="low"):
        return AlignmentResult(
            flight=flt_int, offset=0, correlation=0.0,
            n_matched=0, n_reference=0, search_range=search_range,
            second_best_corr=0.0, confidence=conf, riggs_validation=None)

    # Load LYRA echoes
    if isinstance(echoes_csv, pd.DataFrame):
        echoes = echoes_csv
    else:
        echoes = pd.read_csv(echoes_csv)
        echoes.columns = echoes.columns.str.strip()

    good = echoes[echoes["echo_status"] == "good"].copy()
    if len(good) == 0:
        return _empty()

    lyra_cbd = good["cbd"].astype(int).values
    lyra_hice = good["h_ice_m"].values

    # Load Bingham ICE_THICKN indexed by Stanford CBD
    bingham = load_bingham_ice_thickness(flt_str)
    if bingham is None or len(bingham) == 0:
        return _empty()

    ref_lookup = bingham.groupby("stanford_cbd")["ice_thickn"].mean().to_dict()

    # ---- Stage 1: windowed cross-correlation --------------------------------
    anchors = _windowed_xcorr(lyra_cbd, lyra_hice, ref_lookup, search_range)

    # ---- Stage 2: offset consensus from anchor windows ----------------------
    consensus_offset = None
    consensus_n = 0
    consensus_r = 0.0

    if anchors:
        # Group anchors by offset (within +/-2 tolerance)
        from collections import Counter
        offset_votes: dict[int, int] = {}
        offset_best_r: dict[int, float] = {}

        for a in anchors:
            off = a["offset"]
            # Find existing cluster within +/-2
            matched_key = None
            for existing in offset_votes:
                if abs(off - existing) <= 2:
                    matched_key = existing
                    break
            if matched_key is not None:
                offset_votes[matched_key] += 1
                offset_best_r[matched_key] = max(offset_best_r[matched_key], a["r"])
            else:
                offset_votes[off] = 1
                offset_best_r[off] = a["r"]

        # Pick offset with most votes, break ties by best r
        consensus_offset = max(offset_votes,
                               key=lambda o: (offset_votes[o], offset_best_r[o]))
        consensus_n = offset_votes[consensus_offset]
        consensus_r = offset_best_r[consensus_offset]

    # ---- Stage 3: global correlation as fallback ----------------------------
    global_r, global_n = _xcorr_at_offset(
        lyra_cbd, lyra_hice, ref_lookup, 0, min_matched)
    # Also find the global best offset
    best_global_off, best_global_r, best_global_n = 0, global_r, global_n
    # Quick scan: only test offsets near anchor consensus and at 0
    scan_offsets = set(range(search_range[0], search_range[1] + 1))
    for off in scan_offsets:
        r, n = _xcorr_at_offset(lyra_cbd, lyra_hice, ref_lookup, off, min_matched)
        if r > best_global_r:
            best_global_off, best_global_r, best_global_n = off, r, n

    # Second-best global (at least 5 away from best)
    second_global_r = -1.0
    for off in sorted(scan_offsets, key=lambda o: -_xcorr_at_offset(
            lyra_cbd, lyra_hice, ref_lookup, o, min_matched)[0]):
        if abs(off - best_global_off) >= 5:
            r, _ = _xcorr_at_offset(lyra_cbd, lyra_hice, ref_lookup, off, min_matched)
            if r > -9000:
                second_global_r = r
                break

    # ---- Stage 4: terrain-weighted windowed consensus -------------------------
    # Weight anchor votes by terrain richness (lyra_std) and peak sharpness
    # (r - second_r).  Flat-ice windows have high r but tiny gap to 2nd-best,
    # meaning offset is ambiguous there.  Terrain-rich windows with a clear
    # peak are far more informative.
    terrain_consensus_off = None
    terrain_consensus_n = 0
    terrain_consensus_r = 0.0

    if anchors:
        from collections import defaultdict
        offset_weight: dict[int, float] = defaultdict(float)
        offset_count: dict[int, int] = defaultdict(int)
        offset_best_r2: dict[int, float] = defaultdict(float)

        for a in anchors:
            off = a["offset"]
            gap = a["r"] - max(a["second_r"], 0)
            weight = a["lyra_std"] * gap  # terrain richness x peak sharpness

            # Find existing cluster within +/-2
            matched_key = None
            for existing in list(offset_weight):
                if abs(off - existing) <= 2:
                    matched_key = existing
                    break
            key = matched_key if matched_key is not None else off
            offset_weight[key] += weight
            offset_count[key] += 1
            offset_best_r2[key] = max(offset_best_r2[key], a["r"])

        # Pick offset with highest total weight
        terrain_consensus_off = max(offset_weight,
                                    key=lambda o: offset_weight[o])
        terrain_consensus_n = offset_count[terrain_consensus_off]
        terrain_consensus_r = offset_best_r2[terrain_consensus_off]

    # ---- Stage 5: RIGGS-arbitrated decision ---------------------------------
    # Collect all candidate offsets and pick the one with lowest RIGGS RMSE.
    # If RIGGS is unavailable or insufficient, fall back to terrain-weighted
    # consensus, then global correlation, then offset=0.
    candidates = {0}
    if terrain_consensus_off is not None:
        candidates.add(terrain_consensus_off)
    if best_global_off != 0:
        candidates.add(best_global_off)

    # Evaluate RIGGS at each candidate
    riggs_results = {}
    for cand in candidates:
        rv = _riggs_validate(lyra_cbd, lyra_hice, flt_str, cand)
        riggs_results[cand] = rv

    # Pick candidate with lowest RIGGS RMSE (if >= 3 stations available)
    has_riggs = any(
        rv and rv.get("n_stations", 0) >= 3
        for rv in riggs_results.values()
    )

    chosen_offset = 0
    confidence = "low"
    method = "global"
    n_anchor = 0
    report_r = best_global_r
    report_n = best_global_n

    if has_riggs:
        # RIGGS arbitrates between candidates
        best_rmse = 9999.0
        for cand in candidates:
            rv = riggs_results[cand]
            rmse = (rv.get("rmse_m", 9999)
                    if rv and rv.get("n_stations", 0) >= 3 else 9999)
            if rmse < best_rmse:
                best_rmse = rmse
                chosen_offset = cand

        if chosen_offset != 0 and chosen_offset == terrain_consensus_off:
            method = "windowed"
            n_anchor = terrain_consensus_n
            report_r = terrain_consensus_r
        elif chosen_offset != 0 and chosen_offset == best_global_off:
            method = "global"
            report_r = best_global_r

        # Confidence based on RMSE and method
        if best_rmse < 100:
            confidence = "high"
        elif best_rmse < 200:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        # No RIGGS: rely on windowed consensus or global correlation
        if (terrain_consensus_off is not None
                and terrain_consensus_n >= 3
                and terrain_consensus_r >= 0.8):
            chosen_offset = terrain_consensus_off
            confidence = "high"
            method = "windowed"
            n_anchor = terrain_consensus_n
            report_r = terrain_consensus_r
        elif best_global_r > 0.8 and best_global_n >= 30:
            chosen_offset = best_global_off
            confidence = "medium"
            method = "global"
        # else: offset=0, low confidence (default)

    # ---- Stage 6: track-geometry sanity check -------------------------------
    try:
        nav = load_stanford_nav(flt_str)
        nav_cbds = lyra_cbd + chosen_offset
        valid_mask = (nav_cbds >= 0) & (nav_cbds < len(nav))
        if valid_mask.sum() >= 2:
            from pyproj import Transformer
            _T = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
            lats = nav["LAT"].values[nav_cbds[valid_mask]]
            lons = nav["LON"].values[nav_cbds[valid_mask]]
            x, y = _T.transform(lons, lats)
            dists_km = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / 1000.0
            n_jumps = (dists_km > 10.0).sum()
            if n_jumps > len(dists_km) * 0.1:
                confidence = "low"  # discontinuous track -> downgrade
    except FileNotFoundError:
        pass

    # ---- Stage 7: RIGGS validation at chosen offset -------------------------
    riggs_val = riggs_results.get(chosen_offset)
    if riggs_val is None:
        riggs_val = _riggs_validate(lyra_cbd, lyra_hice, flt_str, chosen_offset)

    return AlignmentResult(
        flight=flt_int,
        offset=chosen_offset,
        correlation=round(report_r, 4),
        n_matched=report_n if method == "global" else (
            anchors[0]["n_pairs"] if anchors else 0),
        n_reference=len(bingham),
        search_range=search_range,
        second_best_corr=round(second_global_r, 4),
        confidence=confidence,
        riggs_validation=riggs_val,
        n_anchor_windows=n_anchor,
        method=method,
    )


def _riggs_validate(lyra_cbd, lyra_hice, flt_str, offset,
                    match_radius_km=50) -> dict | None:
    """Cross-check alignment offset against RIGGS stations."""
    import re as _re
    from scipy.spatial import cKDTree
    from pyproj import Transformer

    if not _RIGGS_CSV.exists():
        return None

    riggs = pd.read_csv(_RIGGS_CSV)

    # Parse DMS coordinates
    def _dms(s):
        s = str(s).strip().strip('"').strip("'")
        m = _re.match(r'(\d+)\D+(\d+)\D+(\d+)["\']?\s*([NSEW])', s)
        if not m:
            return float("nan")
        dd = int(m.group(1)) + int(m.group(2)) / 60.0 + int(m.group(3)) / 3600.0
        if m.group(4) in ("S", "W"):
            dd = -dd
        return dd

    riggs["lat_dd"] = riggs["Latitude"].apply(_dms)
    riggs["lon_dd"] = riggs["Longitude"].apply(_dms)
    valid_riggs = riggs.dropna(subset=["lat_dd", "lon_dd"])

    # Get RIGGS h_ice (prefer seismic)
    h_col_seis = "h_i (seismics), m"
    h_col_radar = "h_i (radar), m"
    riggs_hice = []
    for _, r in valid_riggs.iterrows():
        h = r.get(h_col_seis)
        if pd.isna(h) or h == "" or (isinstance(h, str) and not h.strip()):
            h = r.get(h_col_radar)
        try:
            riggs_hice.append(float(h))
        except (TypeError, ValueError):
            riggs_hice.append(np.nan)
    valid_riggs = valid_riggs.copy()
    valid_riggs["h_ice"] = riggs_hice
    valid_riggs = valid_riggs[valid_riggs["h_ice"].notna()].copy()

    if len(valid_riggs) == 0:
        return None

    # Get LYRA positions using offset
    try:
        nav = load_stanford_nav(flt_str)
    except FileNotFoundError:
        return None

    T = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    # LYRA positions
    nav_cbds = lyra_cbd + offset
    valid_mask = (nav_cbds >= 0) & (nav_cbds < len(nav))
    if valid_mask.sum() == 0:
        return None

    lyra_lats = nav["LAT"].values[nav_cbds[valid_mask]]
    lyra_lons = nav["LON"].values[nav_cbds[valid_mask]]
    lyra_h = lyra_hice[valid_mask]
    lx, ly = T.transform(lyra_lons, lyra_lats)
    lyra_xy = np.column_stack([lx, ly])

    # RIGGS positions
    rx, ry = T.transform(valid_riggs["lon_dd"].values, valid_riggs["lat_dd"].values)
    riggs_xy = np.column_stack([rx, ry])

    # Match each RIGGS station to nearest LYRA frame
    tree = cKDTree(lyra_xy)
    dists, idxs = tree.query(riggs_xy)
    dists_km = dists / 1000.0

    matched = dists_km < match_radius_km
    if not matched.any():
        return {"n_stations": 0}

    diffs = []
    stations = []
    for i in np.where(matched)[0]:
        h_riggs = valid_riggs.iloc[i]["h_ice"]
        h_lyra = lyra_h[idxs[i]]
        if np.isfinite(h_lyra) and np.isfinite(h_riggs):
            diffs.append(h_lyra - h_riggs)
            stations.append(valid_riggs.iloc[i]["Station"])

    if not diffs:
        return {"n_stations": 0}

    diffs = np.array(diffs)
    return {
        "n_stations": len(diffs),
        "stations": stations,
        "mean_diff_m": round(float(np.mean(diffs)), 1),
        "median_diff_m": round(float(np.median(diffs)), 1),
        "rmse_m": round(float(np.sqrt(np.mean(diffs**2))), 1),
    }


def get_nav_positions(
    cbd_array,
    flt: int | str,
    offset: int = 0,
    nav_correction: NavCorrection | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up lat/lon positions for an array of CBDs with a CBD offset.

    Parameters
    ----------
    cbd_array : array-like of LYRA CBD numbers
    flt : flight number
    offset : nav_row = lyra_cbd + offset
    nav_correction : optional piecewise-linear position correction from
        Rose 1978 fix data.  When provided, corrects INS drift at each CBD.

    Returns
    -------
    lats, lons : arrays of latitude/longitude (NaN where out of range)
    """
    from pyproj import Transformer

    nav = load_stanford_nav(flt)
    cbd_array = np.asarray(cbd_array, dtype=int)
    nav_idx = cbd_array + offset

    lats = np.full(len(cbd_array), np.nan)
    lons = np.full(len(cbd_array), np.nan)

    valid = (nav_idx >= 0) & (nav_idx < len(nav))
    lats[valid] = nav["LAT"].values[nav_idx[valid]]
    lons[valid] = nav["LON"].values[nav_idx[valid]]

    # Apply Rose 1978 piecewise-linear nav correction if provided
    if nav_correction is not None and np.any(valid):
        T_fwd = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        T_inv = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)

        v_lats = lats[valid]
        v_lons = lons[valid]
        v_nav_idx = nav_idx[valid]

        # Project to EPSG:3031, apply correction, project back
        x_ps, y_ps = T_fwd.transform(v_lons, v_lats)
        dx, dy = nav_correction.correction_at_cbd(v_nav_idx)
        x_corr = x_ps + dx
        y_corr = y_ps + dy
        lon_corr, lat_corr = T_inv.transform(x_corr, y_corr)

        lats[valid] = lat_corr
        lons[valid] = lon_corr

    return lats, lons


def save_alignment(result: AlignmentResult, out_dir: Path) -> Path:
    """Save alignment result to JSON."""
    import json
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    out_path = val_dir / f"F{result.flight}_alignment.json"
    d = {
        "flight": result.flight,
        "offset": result.offset,
        "correlation": result.correlation,
        "n_matched": result.n_matched,
        "n_reference": result.n_reference,
        "search_range": list(result.search_range),
        "second_best_corr": result.second_best_corr,
        "confidence": result.confidence,
        "riggs_validation": result.riggs_validation,
        "n_anchor_windows": result.n_anchor_windows,
        "method": result.method,
    }
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    return out_path


def load_alignment(flt: int | str, out_dir: Path | None = None) -> AlignmentResult | None:
    """Load a previously computed alignment result from JSON.

    Returns None if no alignment file exists.
    """
    import json
    flt_str = str(flt).lstrip("Ff")
    if out_dir is None:
        out_dir = output_root() / f"F{flt_str}"
    json_path = out_dir / "validation" / f"F{flt_str}_alignment.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return AlignmentResult(
        flight=d["flight"],
        offset=d["offset"],
        correlation=d["correlation"],
        n_matched=d["n_matched"],
        n_reference=d["n_reference"],
        search_range=tuple(d["search_range"]),
        second_best_corr=d["second_best_corr"],
        confidence=d["confidence"],
        riggs_validation=d.get("riggs_validation"),
        n_anchor_windows=d.get("n_anchor_windows", 0),
        method=d.get("method", "global"),
    )


def enrich_echoes_with_nav(flt: int | str,
                           echoes_csv: Path | str | None = None,
                           offset: int | None = None,
                           nav_correction: NavCorrection | None = None) -> Path:
    """Add lat/lon columns to a flight's echoes CSV using the alignment offset.

    If *offset* is None, loads the alignment JSON for the flight.
    If *nav_correction* is provided, applies Rose 1978 piecewise-linear
    position correction to compensate for INS drift.
    Overwrites the echoes CSV in-place with the new columns.

    Returns the path to the updated CSV.
    """
    flt_str = str(flt).lstrip("Ff")
    if echoes_csv is None:
        echoes_csv = (output_root()
                      / f"F{flt_str}" / "phase4" / f"F{flt_str}_echoes.csv")
    echoes_csv = Path(echoes_csv)
    df = pd.read_csv(echoes_csv)
    df.columns = df.columns.str.strip()

    if offset is None:
        alignment = load_alignment(flt_str)
        if alignment is None:
            raise FileNotFoundError(
                f"No alignment JSON for F{flt_str}; provide an offset or an alignment JSON.")
        offset = alignment.offset

    cbd_valid = df["cbd"].notna()
    cbds = df.loc[cbd_valid, "cbd"].astype(int).values
    lats_v, lons_v = get_nav_positions(cbds, flt_str, offset,
                                       nav_correction=nav_correction)

    # Drop existing lat/lon columns if present (will re-insert)
    df = df.drop(columns=["lat", "lon"], errors="ignore")

    # Insert lat/lon right before mb_x
    mb_idx = df.columns.get_loc("mb_x")
    lat_col = np.full(len(df), np.nan)
    lon_col = np.full(len(df), np.nan)
    lat_col[cbd_valid.values] = np.round(lats_v, 6)
    lon_col[cbd_valid.values] = np.round(lons_v, 6)
    df.insert(mb_idx, "lon", lon_col)
    df.insert(mb_idx, "lat", lat_col)

    df.to_csv(echoes_csv, index=False)
    return echoes_csv


_EMBEDDED_SCRIPT_SOURCES: dict[str, str] = {
    'ASTRA_GridPixel_V2': r'''"""
ASTRA Scale - Axis Scale Measurement Tool

Records reference line positions and axis point spacings from TIFF A-scope images.
In LYRA mode, Capture All approves each frame's aligned X-reference line as
the main bang, together with the 10 X and 6 Y grid points.
These X positions may fall outside the frame after montage alignment.

REQUIRED PACKAGES:
- numpy, matplotlib, Pillow, scipy, pandas

INSTALLATION:
    pip install numpy matplotlib Pillow scipy pandas

USAGE:
    python ASTRA_scale.py
    Enter the directory path when prompted.
    Output CSV is saved in the specified directory, named after the folder.

OUTPUT CSV COLUMNS:
    filename      - TIFF filename
    frame         - Frame/capture number within the file
    x_ref_abs     - X reference line absolute pixel position
    x_ref_screen  - X reference line screen (view) pixel position
    x1_abs..x10_abs       - X axis points absolute pixel positions
    x1_screen..x10_screen - X axis points screen (view) pixel positions
    y_ref_abs     - Y reference line absolute pixel position
    y_ref_screen  - Y reference line screen (view) pixel position
    y1_abs..y6_abs       - Y axis points absolute pixel positions
    y1_screen..y6_screen - Y axis points screen (view) pixel positions
    crop_start_y  - First TIFF row used by the ASTRA crop and trace search
    crop_end_y    - Exclusive last TIFF row used by the ASTRA crop and trace search

AUTHOR: Angelo Tarzona, dtarzona@gatech.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
from scipy.ndimage import gaussian_filter
import tkinter as tk
import tkinter.messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
import glob
import argparse
import re
from pathlib import Path
from lyra import (
    ASTRA_DEFAULT_CROP_END_Y, ASTRA_DEFAULT_CROP_START_Y,
    astra_trace, ASTRA_X_POINTS, ASTRA_Y_POINTS,
)

CROP_START_Y = 500
CROP_END_Y = 1800

CSV_COLUMNS = (
    ['filename', 'frame', 'montage_approved', 'x_ref_abs', 'x_ref_screen']
    + [f'x{i}_{kind}' for i in range(1, ASTRA_X_POINTS + 1)
       for kind in ('abs', 'screen')]
    + ['y_ref_abs', 'y_ref_screen']
    + [f'y{i}_{kind}' for i in range(1, ASTRA_Y_POINTS + 1)
       for kind in ('abs', 'screen')]
    + ['crop_start_y', 'crop_end_y']
)

# Disable the decompression bomb check
Image.MAX_IMAGE_PIXELS = None

_parser = argparse.ArgumentParser()
_parser.add_argument("--lyra", action="store_true")
_parser.add_argument("--tiff-dir")
_parser.add_argument("--frame-index")
_parser.add_argument("--output-csv")
_parser.add_argument("--tiff-names", nargs="*")
_lyra_args, _unknown_args = _parser.parse_known_args()
LYRA_MODE = _lyra_args.lyra
LYRA_FRAME_INDEX = None
ASTRA_CANCELLED = False

# --- Get TIFF directory ---
if LYRA_MODE:
    if not (_lyra_args.tiff_dir and _lyra_args.frame_index and _lyra_args.output_csv):
        raise SystemExit(
            "ASTRA --lyra mode requires --tiff-dir, --frame-index, and --output-csv."
        )
    tiff_dir = str(Path(_lyra_args.tiff_dir).expanduser())
    LYRA_FRAME_INDEX = pd.read_csv(_lyra_args.frame_index, dtype=str)
    if _lyra_args.tiff_names:
        requested_names = set(_lyra_args.tiff_names)
    else:
        requested_names = None
else:
    tiff_dir = input("Enter the directory containing TIFF files: ").strip().strip('"').strip("'")

if not os.path.isdir(tiff_dir):
    print(f"Error: Directory not found: {tiff_dir}")
    exit(1)

# Find TIFF files
tiff_files = []
for pattern in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
    tiff_files.extend(glob.glob(os.path.join(tiff_dir, pattern)))
tiff_files = sorted(list(set(tiff_files)))
if LYRA_MODE and requested_names is not None:
    tiff_files = [f for f in tiff_files if os.path.basename(f) in requested_names]
if LYRA_MODE:
    def _first_cbd(path):
        match = re.match(r"^F\d+-C(\d+)_C\d+\.tiff$", os.path.basename(path), re.IGNORECASE)
        return int(match.group(1)) if match else 0

    tiff_files.sort(key=_first_cbd)

if not tiff_files:
    print(f"Error: No TIFF files found in: {tiff_dir}")
    exit(1)
if LYRA_MODE:
    missing_frames = [
        os.path.basename(tiff_path) for tiff_path in tiff_files
        if not (
            (LYRA_FRAME_INDEX["tiff"] == os.path.basename(tiff_path))
            & (LYRA_FRAME_INDEX["frame_type"] == "complete")
        ).any()
    ]
    if missing_frames:
        raise SystemExit(
            "LYRA frame index has no complete frames for: "
            + ", ".join(missing_frames)
        )

print(f"Found {len(tiff_files)} TIFF files in {tiff_dir}")
for i, f in enumerate(tiff_files, 1):
    print(f"  {i}. {os.path.basename(f)}")

# CSV named after the folder unless LYRA provided a centralized output path
folder_name = os.path.basename(os.path.normpath(tiff_dir))
output_csv_path = (
    str(Path(_lyra_args.output_csv).expanduser())
    if LYRA_MODE else str(
        Path(os.environ.get(
            "LYRA_OUTPUT_DIR",
            Path(__file__).resolve().parents[1] / "output",
        )) / f"{folder_name}.csv"
    )
)
Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

# --- Resume from existing CSV if present ---
current_file_index = 0
data = []
if os.path.isfile(output_csv_path):
    try:
        existing_df = pd.read_csv(output_csv_path)
        data = existing_df.to_dict('records')
        if LYRA_MODE and not _lyra_args.tiff_names:
            frame_index = LYRA_FRAME_INDEX
            for idx, tiff_path in enumerate(tiff_files):
                rows = frame_index[
                    (frame_index["tiff"] == os.path.basename(tiff_path))
                    & (frame_index["frame_type"] == "complete")
                ]
                captured = existing_df[
                    existing_df["filename"].astype(str) == os.path.basename(tiff_path)
                ]
                expected_frames = set(range(1, len(rows) + 1))
                captured_frame_values = (
                    captured["frame"] if "frame" in captured
                    else pd.Series(dtype=float)
                )
                captured_frames = set(
                    pd.to_numeric(captured_frame_values, errors="coerce")
                    .dropna().astype(int).tolist()
                )
                required = ['montage_approved', 'x_ref_abs', 'y_ref_abs']
                required += [f'x{i}_abs' for i in range(1, ASTRA_X_POINTS + 1)]
                required += [f'y{i}_abs' for i in range(1, ASTRA_Y_POINTS + 1)]
                complete = all(col in captured.columns for col in required)
                if complete:
                    complete = (
                        captured[required].notna().all(axis=1).all()
                        and captured['montage_approved'].astype(str).str.lower().eq('true').all()
                    )
                if not expected_frames.issubset(captured_frames) or not complete:
                    current_file_index = idx
                    break
            else:
                print("ASTRA grid picks already cover all detected complete frames.")
                raise SystemExit(0)
        elif LYRA_MODE and _lyra_args.tiff_names:
            current_file_index = 0
        elif 'filename' in existing_df.columns and len(existing_df) > 0:
            last_filename = existing_df['filename'].iloc[-1]
            # Find this file in our sorted list and advance past it
            tiff_basenames = [os.path.basename(f) for f in tiff_files]
            if last_filename in tiff_basenames:
                last_idx = tiff_basenames.index(last_filename)
                next_idx = last_idx + 1
                if next_idx < len(tiff_files):
                    current_file_index = next_idx
                    print(f"Resuming: CSV has data through '{last_filename}', "
                          f"starting at '{tiff_basenames[next_idx]}'")
                else:
                    current_file_index = last_idx
                    print(f"CSV already covers all files up to '{last_filename}' (last file). "
                          f"Re-opening it.")
            else:
                print(f"CSV last file '{last_filename}' not found in directory, starting from first.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"Could not read existing CSV, starting fresh: {e}")
        data = []

image_path = tiff_files[current_file_index]
current_filename = os.path.splitext(os.path.basename(image_path))[0]
crop_start_y = CROP_START_Y
crop_end_y = CROP_END_Y
frame_number = 1

# Per-image state
reference_line_y = None
reference_line_y_orig = None
setting_reference_line = False
y_axis_points = []
setting_y_axis = False
y_axis_lines = []
current_y_axis_pick = None

x_reference_line = None
x_reference_line_orig = None
setting_x_reference_line = False
x_axis_points = []
setting_x_axis = False
x_axis_lines = []
current_x_axis_pick = None
prev_x_axis_lines = []
montage_prompt_shown = False
saved_manual_division = None

zoom_width = 2300
trace_y = None


def load_image_and_reset():
    global img, cropped_img, width, height, zoom_height, zoom_width
    global reference_line_y, reference_line_y_orig, setting_reference_line
    global y_axis_points, setting_y_axis, y_axis_lines, current_y_axis_pick
    global x_reference_line, x_reference_line_orig, setting_x_reference_line
    global x_axis_points, setting_x_axis, x_axis_lines, current_x_axis_pick
    global prev_x_axis_lines
    global image_path, current_filename
    global crop_start_y, crop_end_y

    img = Image.open(image_path)
    width, height = img.size
    crop_start_y = CROP_START_Y
    crop_end_y = CROP_END_Y
    if crop_end_y > height:
        crop_end_y = height
    cropped_img = img.crop((0, crop_start_y, img.width, crop_end_y))
    zoom_width = 2300
    zoom_height = cropped_img.height

    # Reset per-image state
    reference_line_y = None
    reference_line_y_orig = None
    setting_reference_line = False
    y_axis_points = []
    setting_y_axis = False
    y_axis_lines = []
    current_y_axis_pick = None

    x_reference_line = None
    x_reference_line_orig = None
    setting_x_reference_line = False
    x_axis_points = []
    setting_x_axis = False
    x_axis_lines = []
    current_x_axis_pick = None
    prev_x_axis_lines = []

    current_filename = os.path.splitext(os.path.basename(image_path))[0]
    frame_number = 1

    try:
        x_slider_control.valmax = cropped_img.width - zoom_width
        x_slider_control.set_val(0)
        x_slider_control.ax.set_xlim(x_slider_control.valmin, x_slider_control.valmax)
    except Exception:
        pass
    process_and_trace(0, zoom_width)


def process_and_trace(left, right):
    global trace_y

    zoomed_in_img = cropped_img.crop((left, 0, right, zoom_height))
    gray_img_array = np.array(zoomed_in_img)
    smoothed_img = gaussian_filter(gray_img_array.astype(float), sigma=4)
    ax.clear()
    ax.imshow(smoothed_img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{current_filename}  ({current_file_index + 1}/{len(tiff_files)})', pad=30)

    trace_y = astra_trace(smoothed_img, 0, smoothed_img.shape[0], smoothed=True)
    ax.plot(range(len(trace_y)), trace_y, color='red', linewidth=2)

    img_w = smoothed_img.shape[1]
    img_h = smoothed_img.shape[0]

    # Y reference line
    if reference_line_y is not None:
        ax.axhline(reference_line_y, color='blue', linestyle='--', linewidth=2)
        ax.text(-40, reference_line_y, "Y Ref", color='blue', va='bottom', ha='right',
                fontsize=10, fontweight='bold')
        ax.text(img_w + 10, reference_line_y, f"{int(reference_line_y)} px", color='blue',
                va='bottom', ha='left', fontsize=10)

    # X reference line
    if x_reference_line is not None:
        ax.axvline(x_reference_line, color='purple', linestyle='--', linewidth=2)
        ax.text(x_reference_line, img_h + 55, "X Ref", color='purple', va='bottom',
                ha='center', fontsize=10, fontweight='bold', clip_on=False)
        ax.text(x_reference_line, -45, f"{int(x_reference_line)} px", color='purple',
                va='top', ha='center', fontsize=10, clip_on=False)

    # Y axis lines
    for i, y in enumerate(y_axis_lines):
        ax.axhline(y, color='green', linestyle=':', linewidth=1)
        ax.text(-40, y, f"Y{i+1}", color='green', va='bottom', ha='right',
                fontsize=9, fontweight='bold')
        ax.text(img_w + 10, y, f"{int(y)} px", color='green', va='bottom', ha='left', fontsize=9)

    # X axis lines
    for i, x in enumerate(x_axis_lines):
        ax.axvline(x, color='orange', linestyle=':', linewidth=1)
        ax.text(x, img_h + 55, f"X{i+1}", color='orange', va='bottom', ha='center',
                fontsize=9, fontweight='bold', clip_on=False)
        ax.text(x, -50, f"{int(x)} px", color='orange', va='top', ha='center',
                fontsize=9, clip_on=False)

    # Previous X axis ghost lines (shown when current x_axis_lines is empty)
    if len(x_axis_lines) == 0 and len(prev_x_axis_lines) > 0:
        for i, x in enumerate(prev_x_axis_lines):
            ax.axvline(x, color='orange', linestyle=':', linewidth=1, alpha=0.35)
            ax.text(x, img_h + 55, f"X{i+1}(prev)", color='orange', va='bottom', ha='center',
                    fontsize=8, fontstyle='italic', alpha=0.5, clip_on=False)

    # Guide lines during Y axis setup
    if setting_y_axis:
        for y in y_axis_points:
            ax.axhline(y, color='green', linestyle='--', linewidth=2, alpha=0.7)
        if current_y_axis_pick is not None:
            ax.axhline(current_y_axis_pick, color='green', linestyle='--', linewidth=2, alpha=0.7)

    # Guide lines during X axis setup
    if setting_x_axis:
        for x in x_axis_points:
            ax.axvline(x, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        if current_x_axis_pick is not None:
            ax.axvline(current_x_axis_pick, color='orange', linestyle='--', linewidth=2, alpha=0.7)

    canvas.draw()


def on_click(event):
    global reference_line_y, reference_line_y_orig, setting_reference_line
    global x_reference_line, x_reference_line_orig, setting_x_reference_line
    global setting_y_axis, y_axis_points, y_axis_lines, current_y_axis_pick
    global setting_x_axis, x_axis_points, x_axis_lines, current_x_axis_pick

    if event is None or event.inaxes is not ax:
        return

    # X reference line setup
    if setting_x_reference_line and x_reference_line is None:
        if event.xdata is not None:
            x_reference_line = int(event.xdata)
            x_reference_line_orig = x_reference_line + int(x_slider_control.val)
            print(f"X Ref set at x={x_reference_line} (orig: {x_reference_line_orig})")
        setting_x_reference_line = False
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
        return

    # Y reference line setup
    if setting_reference_line and reference_line_y is None:
        if event.ydata is not None:
            reference_line_y = int(event.ydata)
            reference_line_y_orig = reference_line_y + crop_start_y
            print(f"Y Ref set at y={reference_line_y} (orig: {reference_line_y_orig})")
        setting_reference_line = False
        process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
        return

    # Y axis points
    if setting_y_axis:
        if event.ydata is not None:
            y_axis_points.append(int(event.ydata))
            current_y_axis_pick = int(event.ydata)
            if len(y_axis_points) == ASTRA_Y_POINTS:
                y_axis_lines = sorted(y_axis_points)
                setting_y_axis = False
                current_y_axis_pick = None
                print(f"Y axis points set: {y_axis_lines}")
            process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
        return

    # X axis points
    if setting_x_axis:
        if event.xdata is not None:
            x_axis_points.append(int(event.xdata))
            current_x_axis_pick = int(event.xdata)
            if len(x_axis_points) == ASTRA_X_POINTS:
                x_axis_lines = sorted(x_axis_points)
                setting_x_axis = False
                current_x_axis_pick = None
                print(f"X axis points set: {x_axis_lines}")
            process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)
        return


def update(val):
    left = int(x_slider_control.val)
    process_and_trace(left, left + zoom_width)


def setup_reference_line():
    global setting_reference_line
    if reference_line_y is None:
        setting_reference_line = True
        print("Click on the plot to set Y reference line")
    else:
        print("Y reference line already set.")


def setup_y_axis():
    global setting_y_axis, y_axis_points, y_axis_lines
    if not setting_y_axis:
        setting_y_axis = True
        y_axis_points = []
        y_axis_lines = []
        print(f"Pick {ASTRA_Y_POINTS} points for Y axis")


def setup_x_reference_line():
    global setting_x_reference_line
    if x_reference_line is None:
        setting_x_reference_line = True
        print("Click on the plot to set X reference line")
    else:
        print("X reference line already set.")


def setup_x_axis():
    global setting_x_axis, x_axis_points, x_axis_lines
    if not setting_x_axis:
        setting_x_axis = True
        x_axis_points = []
        x_axis_lines = []
        print(f"Pick {ASTRA_X_POINTS} points for X axis")


def capture():
    global frame_number, x_axis_lines, x_axis_points, prev_x_axis_lines

    slider_offset = int(x_slider_control.val)
    row = {
        'filename': os.path.basename(image_path),
        'frame': frame_number,
        'crop_start_y': crop_start_y,
        'crop_end_y': crop_end_y,
    }

    # X reference: absolute and screen pixel
    if x_reference_line is not None:
        row['x_ref_abs'] = x_reference_line + slider_offset
        row['x_ref_screen'] = x_reference_line
    else:
        row['x_ref_abs'] = ''
        row['x_ref_screen'] = ''

    # X axis: absolute and screen pixels (fall back to previous frame if not re-picked)
    effective_x_axis = x_axis_lines if len(x_axis_lines) == ASTRA_X_POINTS else prev_x_axis_lines
    if len(effective_x_axis) == ASTRA_X_POINTS:
        sx = sorted(effective_x_axis)
        for i, x in enumerate(sx, 1):
            row[f'x{i}_abs'] = x + slider_offset
            row[f'x{i}_screen'] = x
    else:
        for i in range(1, ASTRA_X_POINTS + 1):
            row[f'x{i}_abs'] = ''
            row[f'x{i}_screen'] = ''

    # Y reference: absolute and screen pixel
    if reference_line_y is not None:
        row['y_ref_abs'] = reference_line_y + crop_start_y
        row['y_ref_screen'] = reference_line_y
    else:
        row['y_ref_abs'] = ''
        row['y_ref_screen'] = ''

    # Y axis: absolute and screen pixels
    if len(y_axis_lines) == ASTRA_Y_POINTS:
        sy = sorted(y_axis_lines)
        for i, y in enumerate(sy, 1):
            row[f'y{i}_abs'] = y + crop_start_y
            row[f'y{i}_screen'] = y
    else:
        for i in range(1, ASTRA_Y_POINTS + 1):
            row[f'y{i}_abs'] = ''
            row[f'y{i}_screen'] = ''

    data.append(row)
    print(f"Captured frame {frame_number}: {row}")
    frame_number += 1

    # Save CSV (overwrites each time so partial progress is saved)
    df = pd.DataFrame(data, columns=CSV_COLUMNS)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved: {output_csv_path}")

    # Save current X axis as previous, then clear for re-picking opportunity
    if len(effective_x_axis) == ASTRA_X_POINTS:
        prev_x_axis_lines = list(effective_x_axis)
    x_axis_lines = []
    x_axis_points = []
    process_and_trace(int(x_slider_control.val), int(x_slider_control.val) + zoom_width)


class FrameDivisionWindow:
    """Step 1: Show full cropped TIFF with slider-controlled frame divisions.
    Each frame is 1500 px wide. Click a line to remove that division."""

    FRAME_WIDTH = 2980

    def __init__(self, parent, cropped_img, zoom_height, on_done):
        self.cropped_img = cropped_img
        self.img_width = cropped_img.width
        self.zoom_height = zoom_height
        self.on_done = on_done

        # Downscale for display, use extent= so click coords are in original space
        disp_w = 1400
        self.scale_factor = disp_w / self.img_width
        disp_h = max(1, int(zoom_height * self.scale_factor))
        self.display_img = np.array(cropped_img.resize((disp_w, disp_h)))

        self.division_lines = []
        self.manual_mode = False
        max_frames = max(1, (self.img_width + self.FRAME_WIDTH - 1) // self.FRAME_WIDTH)

        self.win = tk.Toplevel(parent)
        self.win.title("Step 1: Define Frame Boundaries")

        self.status = tk.Label(self.win,
            text=f"Slider sets # of {self.FRAME_WIDTH}px frames. Click a line to remove it.",
            fg='gray')
        self.status.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(14, 3))
        self.canvas_d = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas_d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        slider_frame = tk.Frame(self.win)
        slider_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(slider_frame, text="Frames:").pack(side=tk.LEFT)
        self.frame_slider = tk.Scale(slider_frame, from_=1, to=max_frames,
                                     orient=tk.HORIZONTAL, command=self._on_slider)
        self.frame_slider.set(max_frames)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.manual_btn = tk.Button(slider_frame, text="Manual Override",
                                    command=self._toggle_manual, bg='#FF9800', fg='white')
        self.manual_btn.pack(side=tk.LEFT, padx=8)

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Next \u2192", command=self._next,
                  bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.LEFT, padx=5)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        self._compute_even(max_frames)
        if saved_manual_division is not None:
            self._toggle_manual()
            self.division_lines = [x for x in saved_manual_division if 0 < x < self.img_width]
            self.status.config(
                text="Manual frame division carried over from the previous TIFF. "
                     "Adjust if needed, then Next.", fg='blue')
        self._redraw()

    def _compute_even(self, n):
        self.division_lines = [self.FRAME_WIDTH * i for i in range(1, n)]

    def _on_slider(self, val):
        if not self.manual_mode:
            self._compute_even(int(val))
            self._redraw()

    def _toggle_manual(self):
        self.manual_mode = not self.manual_mode
        if self.manual_mode:
            self.division_lines = []
            self.frame_slider.config(state=tk.DISABLED)
            self.manual_btn.config(relief=tk.SUNKEN, bg='#E65100')
            self.status.config(
                text="Manual mode: Left-click to add a line, Right-click to remove nearest.",
                fg='blue')
        else:
            self.frame_slider.config(state=tk.NORMAL)
            self.manual_btn.config(relief=tk.RAISED, bg='#FF9800')
            self._compute_even(int(self.frame_slider.get()))
            self.status.config(
                text="Slider mode restored.", fg='gray')
        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        click_x = event.xdata

        if self.manual_mode:
            # Left-click (button 1) => add line
            if event.button == 1:
                x = int(click_x)
                if 0 < x < self.img_width:
                    self.division_lines.append(x)
                    self.division_lines.sort()
                    self.status.config(
                        text=f"Added line at x={x}. {len(self.division_lines)+1} frames.",
                        fg='green')
            # Right-click (button 3) => remove nearest line
            elif event.button == 3 and self.division_lines:
                dists = [abs(line - click_x) for line in self.division_lines]
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < self.img_width * 0.03:
                    removed = self.division_lines.pop(min_idx)
                    self.status.config(
                        text=f"Removed line at x={removed}. {len(self.division_lines)+1} frames.",
                        fg='red')
            self._redraw()
            return

        # Slider mode: click to remove nearest line
        if not self.division_lines:
            return
        dists = [abs(line - click_x) for line in self.division_lines]
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < self.img_width * 0.02:
            removed = self.division_lines.pop(min_idx)
            self.status.config(
                text=f"Removed line at x={removed}. {len(self.division_lines)+1} frames.",
                fg='red')
            self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.display_img, cmap='gray', aspect='auto',
                       extent=[0, self.img_width, self.zoom_height, 0])
        for x in self.division_lines:
            self.ax.axvline(x, color='cyan', linewidth=1.5, alpha=0.8)
        n_frames = len(self.division_lines) + 1
        mode_hint = ("LClick=add  RClick=remove" if self.manual_mode
                     else "Click a line to remove it")
        self.ax.set_title(
            f"{n_frames} frames  |  {len(self.division_lines)} division lines  |  "
            f"{mode_hint}", fontsize=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas_d.draw()

    def _cancel(self):
        plt.close(self.fig)
        self.win.destroy()

    def _next(self):
        edges = [0] + sorted(self.division_lines) + [self.img_width]
        frames = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
        if not frames:
            tk.messagebox.showwarning("No Frames", "Need at least one frame!",
                                      parent=self.win)
            return
        global saved_manual_division
        saved_manual_division = sorted(self.division_lines) if self.manual_mode else None
        plt.close(self.fig)
        self.win.destroy()
        self.on_done(frames)


class PickLeftBoundaryWindow:
    """Step 1.5: Pick the left boundary of the first frame.
    Shows the first frame region; user clicks to set where the frame actually starts."""

    def __init__(self, parent, cropped_img, frame_bounds, zoom_height, on_done):
        self.cropped_img = cropped_img
        self.frame_bounds = frame_bounds
        self.zoom_height = zoom_height
        self.on_done = on_done
        self.left_boundary = None

        start, end = frame_bounds[0]
        # Show a generous region around the first frame
        show_end = min(end + 200, cropped_img.width)
        chunk = cropped_img.crop((0, 0, show_end, zoom_height))
        self.display_img = np.array(chunk)
        self.show_end = show_end

        self.win = tk.Toplevel(parent)
        self.win.title("Step 1.5: Mark Left Boundary of First Frame")

        self.status = tk.Label(self.win,
            text="Click to mark the left boundary of the first frame, then press Next.",
            fg='blue')
        self.status.pack(pady=5)

        h, w = self.display_img.shape[:2]
        aspect = h / max(w, 1)
        fw = min(14, max(8, w / 150))
        fh = max(3, min(8, fw * aspect))
        self.fig, self.ax = plt.subplots(figsize=(fw, fh))
        self.canvas_lb = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas_lb.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Next \u2192", command=self._next,
                  bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.LEFT, padx=5)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.left_boundary = int(event.xdata)
        self.status.config(
            text=f"Left boundary set at x={self.left_boundary} px. Click again to adjust.",
            fg='green')
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.display_img, cmap='gray', aspect='auto',
                       extent=[0, self.show_end, self.zoom_height, 0])
        # Show existing division lines for context
        for start, end in self.frame_bounds:
            if 0 < start < self.show_end:
                self.ax.axvline(start, color='cyan', linewidth=1, alpha=0.4)
        if self.left_boundary is not None:
            self.ax.axvline(self.left_boundary, color='lime', linewidth=2.5, linestyle='-')
            self.ax.text(self.left_boundary + 5, self.zoom_height * 0.05,
                         f"Left = {self.left_boundary}", color='lime',
                         fontsize=10, fontweight='bold')
        self.ax.set_title("Click to set the left boundary of the first frame", fontsize=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas_lb.draw()

    def _cancel(self):
        plt.close(self.fig)
        self.win.destroy()

    def _next(self):
        if self.left_boundary is None:
            tkinter.messagebox.showwarning("Missing",
                "Click on the image to mark the left boundary first.",
                parent=self.win)
            return
        # Adjust the first frame's left edge
        new_bounds = list(self.frame_bounds)
        old_start, old_end = new_bounds[0]
        new_bounds[0] = (self.left_boundary, old_end)
        plt.close(self.fig)
        self.win.destroy()
        self.on_done(new_bounds)


class PickRefsWindow:
    """Step 2: Pick X/Y ref lines and axis points on the first frame."""

    def __init__(self, parent, cropped_img, frame_bounds, zoom_height, on_done):
        self.cropped_img = cropped_img
        self.frame_bounds = frame_bounds
        self.zoom_height = zoom_height
        self.on_done = on_done

        start, end = frame_bounds[0]
        chunk = cropped_img.crop((start, 0, end, zoom_height))
        gray = np.array(chunk)
        self.frame_img = gaussian_filter(gray.astype(float), sigma=4)

        # Compute trace
        self.trace_y = astra_trace(self.frame_img, 0, self.frame_img.shape[0], smoothed=True)

        self.x_ref = None
        self.y_ref = None
        self.x_axis = []
        self.y_axis = []
        self.mode = None
        self.picks = []

        self.win = tk.Toplevel(parent)
        self.win.title("Step 2: Pick References on First Frame")

        top = tk.Frame(self.win)
        top.pack(fill=tk.X, padx=5, pady=5)
        self.status = tk.Label(top, text="Set all references then click Next", fg='gray')
        self.status.pack(side=tk.TOP, pady=(0, 4))

        tk.Button(top, text="Set X Ref / Main Bang",
                  command=lambda: self._set_mode('x_ref')).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Set Y Ref",
                  command=lambda: self._set_mode('y_ref')).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text=f"Set X Axis ({ASTRA_X_POINTS} pts)",
                  command=lambda: self._set_mode('x_axis')).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text=f"Set Y Axis ({ASTRA_Y_POINTS} pts)",
                  command=lambda: self._set_mode('y_axis')).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Next \u2192", command=self._next,
                  bg='#4CAF50', fg='white').pack(side=tk.RIGHT, padx=3)
        tk.Button(top, text="Cancel",
                  command=self._cancel).pack(side=tk.RIGHT, padx=3)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        h, w = self.frame_img.shape[:2]
        aspect = h / max(w, 1)
        fw = min(12, max(6, w / 200))
        fh = min(8, fw * aspect)
        self.fig, self.ax = plt.subplots(figsize=(fw, fh))
        self.canvas_p = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas_p.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._redraw()

    def _set_mode(self, m):
        self.mode = m
        self.picks = []
        msgs = {
            'x_ref': 'Click to set X Reference / Main Bang line',
            'y_ref': 'Click to set Y Reference line',
            'x_axis': f'Pick X axis points (0/{ASTRA_X_POINTS})',
            'y_axis': f'Pick Y axis points (0/{ASTRA_Y_POINTS})',
        }
        self.status.config(text=msgs[m], fg='blue')

    def _on_click(self, event):
        if self.mode is None or event.inaxes != self.ax:
            return
        if self.mode == 'x_ref':
            self.x_ref = int(event.xdata)
            self.mode = None
            self.status.config(text=f"X Ref = {self.x_ref} px", fg='purple')
        elif self.mode == 'y_ref':
            self.y_ref = int(event.ydata)
            self.mode = None
            self.status.config(text=f"Y Ref = {self.y_ref} px", fg='blue')
        elif self.mode == 'x_axis':
            self.picks.append(int(event.xdata))
            n = len(self.picks)
            if n < ASTRA_X_POINTS:
                self.status.config(text=f"Pick X axis points ({n}/{ASTRA_X_POINTS})", fg='orange')
            else:
                self.x_axis = sorted(self.picks)
                self.picks = []
                self.mode = None
                self.status.config(text=f"X axis: {self.x_axis}", fg='orange')
        elif self.mode == 'y_axis':
            self.picks.append(int(event.ydata))
            n = len(self.picks)
            if n < ASTRA_Y_POINTS:
                self.status.config(text=f"Pick Y axis points ({n}/{ASTRA_Y_POINTS})", fg='green')
            else:
                self.y_axis = sorted(self.picks)
                self.picks = []
                self.mode = None
                self.status.config(text=f"Y axis: {self.y_axis}", fg='green')
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.frame_img, cmap='gray')
        self.ax.plot(range(len(self.trace_y)), self.trace_y, color='red', linewidth=1.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("First Frame \u2014 Pick References", fontsize=10)
        if self.y_ref is not None:
            self.ax.axhline(self.y_ref, color='blue', linestyle='--', linewidth=2)
        if self.x_ref is not None:
            self.ax.axvline(self.x_ref, color='purple', linestyle='--', linewidth=2)
        for y in self.y_axis:
            self.ax.axhline(y, color='green', linestyle=':', linewidth=1)
        for x in self.x_axis:
            self.ax.axvline(x, color='orange', linestyle=':', linewidth=1)
        for p in self.picks:
            if self.mode == 'x_axis':
                self.ax.axvline(p, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
            elif self.mode == 'y_axis':
                self.ax.axhline(p, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
        self.fig.tight_layout()
        self.canvas_p.draw()

    def _next(self):
        missing = []
        if self.x_ref is None:
            missing.append("X Ref")
        if self.y_ref is None:
            missing.append("Y Ref")
        if len(self.x_axis) != ASTRA_X_POINTS or len(set(self.x_axis)) != ASTRA_X_POINTS:
            missing.append(f"X Axis ({ASTRA_X_POINTS} pts)")
        if len(self.y_axis) != ASTRA_Y_POINTS or len(set(self.y_axis)) != ASTRA_Y_POINTS:
            missing.append(f"Y Axis ({ASTRA_Y_POINTS} pts)")
        if missing:
            tk.messagebox.showwarning("Missing",
                "Still need:\n\u2022 " + "\n\u2022 ".join(missing), parent=self.win)
            return
        plt.close(self.fig)
        self.win.destroy()
        self.on_done(self.x_ref, self.y_ref, self.x_axis, self.y_axis)

    def _cancel(self):
        global ASTRA_CANCELLED
        plt.close(self.fig)
        self.win.destroy()
        if LYRA_MODE:
            ASTRA_CANCELLED = True
            root.quit()


class FrameXAxisPickerWindow:
    """Manual per-frame X-axis picker for a montage frame."""

    def __init__(self, parent, frame_img, x_ref_line, x_axis_positions,
                 current_custom, on_done):
        self.frame_img = frame_img
        self.x_ref_line = x_ref_line
        self.x_axis_positions = sorted(x_axis_positions)
        self.custom = sorted(current_custom) if current_custom else []
        self.on_done = on_done
        self.mode = None
        self.picks = []

        self.win = tk.Toplevel(parent)
        self.win.title("Manual X-Axis Picker")

        top = tk.Frame(self.win)
        top.pack(fill=tk.X, padx=5, pady=5)
        self.status = tk.Label(top,
            text=f"Click Pick X Axis, then select {ASTRA_X_POINTS} X axis positions on the frame.", fg='gray')
        self.status.pack(side=tk.LEFT, pady=(0, 4))

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_frame, text="Pick X Axis", command=lambda: self._set_mode('x_axis')).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Clear Picks", command=self._clear).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Done", command=self._done,
                  bg='#4CAF50', fg='white').pack(side=tk.RIGHT, padx=3)
        tk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=3)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        h, w = self.frame_img.shape[:2]
        aspect = h / max(w, 1)
        fw = min(12, max(6, w / 200))
        fh = min(8, fw * aspect)
        self.fig, self.ax = plt.subplots(figsize=(fw, fh))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._set_mode('x_axis')
        self._redraw()

    def _set_mode(self, mode):
        self.mode = mode
        self.picks = []
        if mode == 'x_axis':
            self.status.config(text=f"Pick X axis points ({len(self.picks)}/{ASTRA_X_POINTS})", fg='magenta')

    def _on_click(self, event):
        if self.mode != 'x_axis' or event.inaxes != self.ax or event.xdata is None:
            return
        self.picks.append(int(event.xdata))
        n = len(self.picks)
        if n < ASTRA_X_POINTS:
            self.status.config(text=f"Pick X axis points ({n}/{ASTRA_X_POINTS})", fg='magenta')
        else:
            self.custom = sorted(self.picks)
            self.picks = []
            self.mode = None
            self.status.config(text=f"Manual X axis saved ({len(self.custom)} points)", fg='purple')
        self._redraw()

    def _clear(self):
        self.custom = []
        self.picks = []
        self.mode = None
        self.status.config(text="Cleared manual X axis picks.", fg='gray')
        self._redraw()

    def _done(self):
        if self.custom and len(self.custom) != ASTRA_X_POINTS:
            tk.messagebox.showwarning("Incomplete",
                f"Need exactly {ASTRA_X_POINTS} X axis points or clear the picks.", parent=self.win)
            return
        plt.close(self.fig)
        self.win.destroy()
        self.on_done(self.custom if len(self.custom) == ASTRA_X_POINTS else None)

    def _cancel(self):
        plt.close(self.fig)
        self.win.destroy()

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.frame_img, cmap='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Manual X axis picker", fontsize=10)
        self.ax.axvline(self.x_ref_line, color='purple', linewidth=1.5,
                        linestyle='--', alpha=0.8)
        for x in self.x_axis_positions:
            self.ax.axvline(x, color='orange', linestyle=':', linewidth=1, alpha=0.4)
        if self.custom and len(self.custom) == ASTRA_X_POINTS:
            for x in self.custom:
                self.ax.axvline(x, color='magenta', linestyle='--', linewidth=2.5, alpha=0.8)
        for p in self.picks:
            self.ax.axvline(p, color='magenta', linestyle='--', linewidth=1.5, alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()


class MontageReviewWindow:
    """Step 3: Review all frames with per-frame X adjustment sliders under each thumbnail.
    Double-click a thumbnail to zoom in; scroll wheel zooms the whole montage."""

    def __init__(self, parent, cropped_img, frame_bounds, zoom_height,
                 x_ref, y_ref, x_axis, y_axis, img_path, crop_start_y, on_capture):
        self.cropped_img = cropped_img
        self.frame_bounds = frame_bounds
        self.zoom_height = zoom_height
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.x_axis = list(x_axis)
        self.y_axis = list(y_axis)
        self.img_path = img_path
        self.crop_start_y = crop_start_y
        self.on_capture = on_capture

        self.n = len(frame_bounds)
        self.x_offsets = [0] * self.n
        self.custom_x_axes = [None] * self.n
        self.x_correction = None

        self.frame_images = []
        for start, end in frame_bounds:
            chunk = cropped_img.crop((start, 0, end, zoom_height))
            self.frame_images.append(gaussian_filter(np.asarray(chunk, dtype=float), sigma=4))

        # Compute grid layout â€” fixed 4 columns, sized to fill the screen
        screen_w = parent.winfo_screenwidth()
        screen_h = parent.winfo_screenheight()
        self.cols = min(self.n, 4)
        self.rows = max(1, (self.n + self.cols - 1) // self.cols)
        # Size each thumbnail to fill available width, with reasonable height
        avail_w = screen_w - 80  # scrollbar + padding
        self.fig_w = max(2, avail_w / self.cols / 100)  # inches at 100 dpi
        self.fig_h = max(1.5, self.fig_w * 0.7)  # ~70% aspect ratio

        self.win = tk.Toplevel(parent)
        self.win.title(f"Step 3: Montage Review \u2014 {os.path.basename(img_path)}")
        self.win.geometry(f"{screen_w - 40}x{screen_h - 80}+20+20")

        top = tk.Frame(self.win)
        top.pack(fill=tk.X, padx=5, pady=5)
        self.status = tk.Label(top,
            text="Review the purple X reference / main bang and grids in each frame, then Capture All. "
                 "Double-click a frame to re-pick its X axis lines, right-click to zoom in.", fg='gray')
        self.status.pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Capture All", command=self._capture,
                  bg='#4CAF50', fg='white').pack(side=tk.RIGHT, padx=5)
        tk.Button(top, text="Cancel",
                  command=self._on_close).pack(side=tk.RIGHT, padx=5)
        tk.Button(top, text="Reset X Correction",
                  command=self._reset_x_correction).pack(side=tk.RIGHT, padx=5)

        # Scrollable area for frames + sliders
        scroll_outer = tk.Frame(self.win)
        scroll_outer.pack(fill=tk.BOTH, expand=True)
        self.scroll_canvas = tk.Canvas(scroll_outer)
        vsb = tk.Scrollbar(scroll_outer, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        hsb = tk.Scrollbar(scroll_outer, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        scroll_inner = tk.Frame(self.scroll_canvas)
        scroll_inner.bind("<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.create_window((0, 0), window=scroll_inner, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse wheel scrolling (unbind on window close to avoid stale references)
        self._mw_binding = self.scroll_canvas.bind_all("<MouseWheel>",
            lambda e: self.scroll_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Build grid: each cell = figure + slider
        self.figs = []
        self.axes_list = []
        self.canvases = []
        self.sliders = []
        self.snap_buttons = []
        self.snap_idx = None
        self.snapped = set()

        for i in range(self.n):
            r = i // self.cols
            c = i % self.cols
            cell = tk.Frame(scroll_inner, bd=1, relief=tk.GROOVE)
            cell.grid(row=r, column=c, padx=2, pady=2)

            fig_i, ax_i = plt.subplots(figsize=(self.fig_w, self.fig_h))
            canvas_i = FigureCanvasTkAgg(fig_i, master=cell)
            canvas_i.get_tk_widget().pack()
            fig_i.canvas.mpl_connect('button_press_event',
                lambda event, idx=i: self._on_canvas_event(event, idx))
            self.figs.append(fig_i)
            self.axes_list.append(ax_i)
            self.canvases.append(canvas_i)

            s = tk.Scale(cell, from_=-2000, to=2000, orient=tk.HORIZONTAL,
                         length=int(self.fig_w * 80), resolution=1,
                         label=f"F{i+1} X offset",
                         command=lambda v, idx=i: self._adj(idx, int(float(v))))
            s.set(0)
            s.pack(pady=(0, 2))
            self.sliders.append(s)

            b = tk.Button(cell, text="Snap", command=lambda idx=i: self._arm_snap(idx))
            b.pack(pady=(0, 3))
            self.snap_buttons.append(b)

        n_aligned = sum(1 for i in range(1, self.n) if self._auto_snap(i))
        self.status.config(
            text=f"Auto-aligned {n_aligned}/{max(0, self.n - 1)} frames to the F1 grid. "
                 "Review grid alignment, then Capture All.",
            fg='blue')
        self._redraw_all()

    def _on_close(self):
        global ASTRA_CANCELLED
        self.scroll_canvas.unbind_all("<MouseWheel>")
        for f in self.figs:
            plt.close(f)
        self.win.destroy()
        if LYRA_MODE:
            ASTRA_CANCELLED = True
            root.quit()

    def _arm_snap(self, idx):
        """Frames after F1: auto-align the grid by matching the previous frame's
        grid region (X ref line through last X axis line) in this frame.
        F1 (or a failed match): click where the X ref line should go."""
        if idx > 0 and self._auto_snap(idx):
            return
        if self.snap_idx is not None:
            self.snap_buttons[self.snap_idx].config(relief=tk.RAISED, bg='SystemButtonFace')
        self.snap_idx = idx
        self.snap_buttons[idx].config(relief=tk.SUNKEN, bg='#FFD54F')
        self.status.config(
            text=f"Frame {idx+1}: click where the X reference line should go.", fg='purple')

    def _trace(self, i):
        """Per-column y of the first dark pixel (the echo trace), as in ASTRA.py."""
        img = self.frame_images[i]
        return astra_trace(img, 0, img.shape[0], smoothed=True)

    def _auto_snap(self, idx):
        """ASTRA.py-style alignment: every frame is aligned directly to the reference
        frame F1 (where the X ref and X axis were picked), not chained. F1's trace from
        just before the X ref line through the grid is slid across this frame's trace
        near the same spot; the grid follows the best match."""
        ref_tr, cur_tr = self._trace(0), self._trace(idx)
        a = max(0, self.x_ref - 50)
        b = min(len(ref_tr), self.x_ref + 700)
        length = b - a
        if length < 50 or len(cur_tr) <= length:
            return False
        search = int(max(50, 0.5 * len(cur_tr)))
        c0 = max(0, a - search)
        c1 = min(len(cur_tr), b + search)
        if c1 - c0 <= length:
            return False
        cand = np.lib.stride_tricks.sliding_window_view(cur_tr[c0:c1], length)
        differences = np.abs(cand - ref_tr[a:b])
        valid = np.isfinite(differences)
        counts = valid.sum(axis=1)
        if not np.any(counts >= max(1, length // 2)):
            return False
        scores = np.full(len(cand), np.inf)
        eligible = counts >= max(1, length // 2)
        scores[eligible] = np.nansum(differences[eligible], axis=1) / counts[eligible]
        shift = int(np.argmin(scores))
        off = max(-2000, min(2000, c0 + shift - a))
        self.sliders[idx].set(off)
        self.snapped.add(idx)
        self.status.config(
            text=f"Frame {idx+1} aligned to F1 grid via trace match: offset {off:+d}.",
            fg='blue')
        return True
    def _snap(self, idx, x_click):
        """Shift the X ref line (and the whole X grid with it) to the clicked x.
        The image stays fixed."""
        x_feat = int(round(x_click))
        raw = x_feat - self.x_ref
        off = max(-2000, min(2000, raw))
        self.sliders[idx].set(off)  # triggers _adj -> redraw
        self.snapped.add(idx)
        self.status.config(
            text=f"Frame {idx+1} snapped: X ref -> x={x_feat}, grid shifted {off:+d} px "
                 f"from the F1 grid; recorded on Capture All."
                 + (" [clamped]" if off != raw else ""), fg='blue')

    def _on_canvas_event(self, event, idx):
        if (event.button == 1 and self.snap_idx == idx and event.inaxes is not None
                and event.xdata is not None and not event.dblclick):
            self.snap_buttons[idx].config(relief=tk.RAISED, bg='SystemButtonFace')
            self.snap_idx = None
            self._snap(idx, event.xdata)
            return
        if event.button == 3:
            self._show_zoom(idx)
            return
        if event.dblclick and event.button == 1:
            self._open_manual_x_picker(idx)

    def _reset_x_correction(self):
        self.x_correction = None
        self.status.config(text="X correction reset; next double-click re-picks it.", fg='blue')

    def _open_manual_x_picker(self, idx):
        off = self.x_offsets[idx]
        if self.x_correction is not None:
            # Reuse the per-line correction from the first repicked frame
            picks = [x + off + d for x, d in zip(sorted(self.x_axis), self.x_correction)]
            self._set_custom_x_axis(idx, picks, from_template=True)
            return
        x_ref_line = self.x_ref + off
        x_axis_positions = [x + off for x in self.x_axis]
        FrameXAxisPickerWindow(self.win, self.frame_images[idx], x_ref_line,
                               x_axis_positions, self.custom_x_axes[idx],
                               on_done=lambda picks, idx=idx, off=off:
                                   self._first_pick_done(idx, off, picks))

    def _first_pick_done(self, idx, off, picks):
        if picks is not None and len(picks) == len(self.x_axis):
            self.x_correction = [p - (x + off) for p, x in zip(sorted(picks), sorted(self.x_axis))]
        self._set_custom_x_axis(idx, picks)

    def _set_custom_x_axis(self, idx, picks, from_template=False):
        self.custom_x_axes[idx] = sorted(picks) if picks is not None else None
        if from_template:
            self.status.config(text=f"Frame {idx+1}: applied X correction from first repick.", fg='blue')
            self._redraw_frame(idx)
            return
        self.status.config(text=f"Frame {idx+1} custom X axis updated.", fg='blue')
        self._redraw_frame(idx)

    def _show_zoom(self, idx):
        """Pop up a larger zoomable view of a single frame."""
        zoom_win = tk.Toplevel(self.win)
        start, end = self.frame_bounds[idx]
        off = self.x_offsets[idx]
        zoom_win.title(f"Zoom: F{idx+1} [{start}:{end}] adj:{off:+d}")

        fig_z, ax_z = plt.subplots(figsize=(10, 7))
        canvas_z = FigureCanvasTkAgg(fig_z, master=zoom_win)
        canvas_z.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ax_z.imshow(self.frame_images[idx], cmap='gray', aspect='auto')
        ax_z.plot(self._trace(idx), color='red', linewidth=1.0, alpha=0.8)
        ax_z.set_title(f"F{idx+1} [{start}:{end}] adj:{off:+d}", fontsize=10)
        ax_z.axhline(self.y_ref, color='blue', linewidth=1.5, linestyle='--', alpha=0.8)
        ax_z.axvline(self.x_ref + off, color='purple', linewidth=1.5,
                     linestyle='--', alpha=0.8)
        for y in self.y_axis:
            ax_z.axhline(y, color='green', linewidth=1, linestyle=':', alpha=0.6)
        for x in self.x_axis:
            ax_z.axvline(x + off, color='orange', linewidth=1, linestyle=':', alpha=0.6)
        if self.custom_x_axes[idx] and len(self.custom_x_axes[idx]) == ASTRA_X_POINTS:
            for x in self.custom_x_axes[idx]:
                ax_z.axvline(x, color='magenta', linewidth=2, linestyle='--', alpha=0.8)
        ax_z.set_xlim(-0.5, self.frame_images[idx].shape[1] - 0.5)

        # Enable matplotlib's built-in zoom/pan toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas_z, zoom_win)
        toolbar.update()

        fig_z.tight_layout()
        canvas_z.draw()
        tk.Button(zoom_win, text="Close", command=zoom_win.destroy).pack(pady=5)

    def _adj(self, idx, val):
        self.x_offsets[idx] = val
        self._redraw_frame(idx)

    def _redraw_frame(self, i):
        a = self.axes_list[i]
        a.clear()
        a.imshow(self.frame_images[i], cmap='gray', aspect='auto')
        a.plot(self._trace(i), color='red', linewidth=0.7, alpha=0.8)
        start, end = self.frame_bounds[i]
        off = self.x_offsets[i]
        a.set_title(f"F{i+1} [{start}:{end}] adj:{off:+d}", fontsize=7)
        a.set_xticks([])
        a.set_yticks([])
        a.axhline(self.y_ref, color='blue', linewidth=3.2, linestyle='--', alpha=0.8)
        a.axvline(self.x_ref + off, color='purple', linewidth=3.2,
                  linestyle='--', alpha=0.8)
        for y in self.y_axis:
            a.axhline(y, color='green', linewidth=3.2, linestyle=':', alpha=0.7)
        for x in self.x_axis:
            a.axvline(x + off, color='orange', linewidth=3.2, linestyle=':', alpha=0.7)
        if self.custom_x_axes[i] and len(self.custom_x_axes[i]) == ASTRA_X_POINTS:
            for x in self.custom_x_axes[i]:
                a.axvline(x, color='magenta', linewidth=2.2, linestyle='--', alpha=0.85)
        a.set_xlim(-0.5, self.frame_images[i].shape[1] - 0.5)
        self.figs[i].tight_layout()
        self.canvases[i].draw_idle()

    def _redraw_all(self):
        for i in range(self.n):
            self._redraw_frame(i)

    def _capture(self):
        for i, _ in enumerate(self.frame_bounds):
            grid = self.custom_x_axes[i] or [x + self.x_offsets[i] for x in self.x_axis]
            if len(grid) != ASTRA_X_POINTS or len(set(grid)) != ASTRA_X_POINTS:
                tkinter.messagebox.showwarning(
                    "Invalid frame picks",
                    f"Frame {i + 1}: {ASTRA_X_POINTS} distinct X grid points are required.",
                    parent=self.win,
                )
                return
        self.on_capture(self.frame_bounds, self.x_ref, self.y_ref,
                        self.x_axis, self.y_axis, self.x_offsets,
                        self.custom_x_axes)
        self.scroll_canvas.unbind_all("<MouseWheel>")
        for f in self.figs:
            plt.close(f)
        self.win.destroy()


def start_montage_workflow():
    """Launch montage: divide frames -> pick left boundary -> pick refs -> review & capture."""
    if LYRA_MODE:
        rows = LYRA_FRAME_INDEX[
            (LYRA_FRAME_INDEX["tiff"] == os.path.basename(image_path))
            & (LYRA_FRAME_INDEX["frame_type"] == "complete")
        ].sort_values("frame_idx")
        if rows.empty:
            global ASTRA_CANCELLED
            ASTRA_CANCELLED = True
            tkinter.messagebox.showerror(
                "No detected frames",
                f"LYRA found no complete frames for {os.path.basename(image_path)}.",
                parent=root,
            )
            root.quit()
            return
        frame_bounds = [
            (int(row["left_px"]), int(row["right_px"]) + 1)
            for _, row in rows.iterrows()
        ]
        _montage_step2(frame_bounds)
        return
    FrameDivisionWindow(root, cropped_img, zoom_height,
                        on_done=_montage_step1_5)


def _montage_step1_5(frame_bounds):
    PickLeftBoundaryWindow(root, cropped_img, frame_bounds, zoom_height,
                           on_done=_montage_step2)


def _montage_step2(frame_bounds):
    PickRefsWindow(root, cropped_img, frame_bounds, zoom_height,
                   on_done=lambda xr, yr, xa, ya:
                       _montage_step3(frame_bounds, xr, yr, xa, ya))


def _montage_step3(frame_bounds, x_ref, y_ref, x_axis, y_axis):
    MontageReviewWindow(root, cropped_img, frame_bounds, zoom_height,
                        x_ref, y_ref, x_axis, y_axis,
                        image_path, crop_start_y, on_capture=do_montage_capture)


def do_montage_capture(frame_bounds, m_x_ref, m_y_ref, m_x_axis, m_y_axis,
                       x_offsets, custom_x_axes):
    """Save all frames from montage workflow."""
    global frame_number, data, x_axis_lines, x_axis_points, prev_x_axis_lines
    global x_reference_line, reference_line_y, y_axis_lines, montage_prompt_shown
    global ASTRA_CANCELLED

    if montage_prompt_shown:
        print("Warning: montage capture re-entered; ignoring duplicate call")
        return
    montage_prompt_shown = True

    x_reference_line = m_x_ref
    reference_line_y = m_y_ref
    x_axis_lines = list(m_x_axis)
    y_axis_lines = list(m_y_axis)
    filename = os.path.basename(image_path)
    data = [row for row in data if row.get('filename') != filename]

    sorted_x = sorted(m_x_axis)
    sorted_y = sorted(m_y_axis)

    for i, (start, end) in enumerate(frame_bounds):
        off = x_offsets[i]
        row = {
            'filename': filename,
            'frame': frame_number,
            'montage_approved': True,
            'crop_start_y': crop_start_y,
            'crop_end_y': crop_end_y,
        }
        row['x_ref_abs'] = m_x_ref + off + start
        row['x_ref_screen'] = m_x_ref + off

        if custom_x_axes and custom_x_axes[i] and len(custom_x_axes[i]) == ASTRA_X_POINTS:
            for j, x in enumerate(sorted(custom_x_axes[i]), 1):
                row[f'x{j}_abs'] = x + start
                row[f'x{j}_screen'] = x
        else:
            for j, x in enumerate(sorted_x, 1):
                row[f'x{j}_abs'] = x + off + start
                row[f'x{j}_screen'] = x + off

        row['y_ref_abs'] = m_y_ref + crop_start_y
        row['y_ref_screen'] = m_y_ref
        for j, y in enumerate(sorted_y, 1):
            row[f'y{j}_abs'] = y + crop_start_y
            row[f'y{j}_screen'] = y
        data.append(row)
        print(f"  Frame {frame_number}: [{start}:{end}] x_adj={off}")
        frame_number += 1

    df = pd.DataFrame(data, columns=CSV_COLUMNS)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Montage captured: {len(frame_bounds)} frames. CSV: {output_csv_path}")

    prev_x_axis_lines = list(m_x_axis)
    x_axis_lines = []
    x_axis_points = []

    # Ask whether to continue to the next TIFF or quit
    if current_file_index + 1 < len(tiff_files):
        next_name = os.path.basename(tiff_files[current_file_index + 1])
        cont = tkinter.messagebox.askyesno(
            "Continue?",
            f"Captured {len(frame_bounds)} frames.\nCSV saved: {output_csv_path}\n\n"
            f"Continue to next file ({next_name})?")
        if cont:
            done_current_file()
            root.after(100, start_montage_workflow)
            montage_prompt_shown = False
        else:
            print("Done. CSV saved.")
            montage_prompt_shown = False
            if LYRA_MODE:
                ASTRA_CANCELLED = True
            root.quit()
    else:
        montage_prompt_shown = False
        tkinter.messagebox.showinfo(
            "Done",
            f"All {len(tiff_files)} TIFF files processed.\nCSV saved: {output_csv_path}")
        root.quit()


def done_current_file():
    global current_file_index, image_path, current_filename, frame_number

    # Move to next TIFF
    current_file_index += 1
    if current_file_index >= len(tiff_files):
        print("All TIFF files processed.")
        root.quit()
        return

    image_path = tiff_files[current_file_index]
    current_filename = os.path.splitext(os.path.basename(image_path))[0]
    frame_number = 1
    print(f"Loading: {os.path.basename(image_path)}")
    load_image_and_reset()


# --- Build GUI ---
root = tk.Tk()
root.title("ASTRA Scale Measurement")

control_frame = tk.Frame(root)
control_frame.pack(side=tk.LEFT, padx=20)

plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.RIGHT, padx=20)

fig, ax = plt.subplots(figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

load_image_and_reset()
process_and_trace(0, zoom_width)

# Slider
ax_slider = fig.add_axes([0.1, 0.01, 0.6, 0.03], facecolor='lightgoldenrodyellow')
x_slider_control = Slider(ax_slider, 'X Position', 0, cropped_img.width - zoom_width, valinit=0, valstep=1)
x_slider_control.on_changed(update)

# Navigation buttons
ax_jump = fig.add_axes([0.80, 0.05, 0.11, 0.04])
btn_jump = Button(ax_jump, 'Move 1000 px')
btn_jump.on_clicked(lambda e: x_slider_control.set_val(
    min(cropped_img.width - zoom_width, x_slider_control.val + 1000)))

ax_left = fig.add_axes([0.80, 0.01, 0.04, 0.03])
btn_left = Button(ax_left, '\u2190')
btn_left.on_clicked(lambda e: x_slider_control.set_val(max(0, x_slider_control.val - 10)))

ax_right = fig.add_axes([0.87, 0.01, 0.04, 0.03])
btn_right = Button(ax_right, '\u2192')
btn_right.on_clicked(lambda e: x_slider_control.set_val(
    min(cropped_img.width - zoom_width, x_slider_control.val + 10)))

# Control buttons
tk.Button(control_frame, text="Set Up X Axis Reference Line", command=setup_x_reference_line).pack(pady=2)
tk.Button(control_frame, text=f"Set Up X Axis ({ASTRA_X_POINTS} points)", command=setup_x_axis).pack(pady=2)
tk.Button(control_frame, text="Set Up Y Axis Reference Line", command=setup_reference_line).pack(pady=2)
tk.Button(control_frame, text=f"Set Up Y Axis ({ASTRA_Y_POINTS} points)", command=setup_y_axis).pack(pady=2)
tk.Label(control_frame, text="").pack()
tk.Button(control_frame, text="Capture", command=capture).pack(pady=5)

tk.Label(control_frame, text="--- Montage ---", font=('Arial', 9, 'bold')).pack(pady=(10, 2))
tk.Button(control_frame, text="Montage Workflow", command=start_montage_workflow,
         bg='#4CAF50', fg='white').pack(pady=5)
tk.Button(control_frame, text="Done (Next File)", command=done_current_file).pack(pady=5)
tk.Button(control_frame, text="Quit", command=lambda: (root.quit(), root.destroy())).pack(pady=5)

fig.canvas.mpl_connect('button_press_event', on_click)

# Hide the main window â€” montage workflow uses its own popups
root.withdraw()

# Auto-launch montage workflow on startup
root.after(100, start_montage_workflow)

root.mainloop()
if LYRA_MODE and ASTRA_CANCELLED:
    raise SystemExit("ASTRA grid picking was cancelled or could not start.")
''',
    'build_digit_templates': r'''"""
build_digit_templates.py — LYRA digit template builder
=======================================================
Uses user-provided ground-truth CBD numbers to extract per-digit pixel
templates from the 7-segment display font printed on the film, then saves
them for use by the NCC-based digit recognizer on all future TIFFs.

Ground truth source: user-confirmed for 40_0008400_0008424-reel_begin_end.tiff
  Flight 125, frames 1–12, CBDs 0458–0469 (sequential, all 10 digits covered)

Recognition strategy (two-stage):
  Stage 1 — NCC matching: match each digit blob to the nearest template
  Stage 2 — Sequential constraint: |CBD[n+1] - CBD[n]| = 1 always; use
             majority-vote anchor to correct any wrong NCC reads

Outputs
-------
  digit_templates.npy          — dict {digit_char: (H, W) float32 array}
  output/F125/templates_validation.png  — contact sheet showing
      each saved template and recognition result on all frames

Run the embedded utility:
    python lyra.py --tool build_digit_templates
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT    = Path(__file__).resolve().parents[1]
TIFF    = ROOT / "tiff/125/40_0008400_0008424-reel_begin_end.tiff"
TMPL    = Path(__file__).resolve().parent / "digit_templates.npy"

sys.path.insert(0, str(ROOT / "codes"))
from lyra import detect_frames, display_path, output_root, _connected_components_1d, _gauss_smooth

OUT_DIR = output_root(ROOT / "output") / "F125"

# -- Ground truth from user --------------------------------------------------
FLIGHT = 125
KNOWN_CBDS = {
    1: 458,  2: 459,  3: 460,  4: 461,  5: 462,  6: 463,
    7: 464,  8: 465,  9: 466, 10: 467, 11: 468, 12: 469,
}

# -- Constants for text region -----------------------------------------------
TEXT_Y0_FRAC = 0.803    # top of CBD text row (fraction of frame height)
TEXT_Y1_FRAC = 0.865    # bottom of CBD text row
TEXT_X0_FRAC = 0.25     # left crop (fraction of frame width)
TEXT_X1_FRAC = 0.75     # right crop
SCALE_X      = 5        # upscale factor horizontal
SCALE_Y      = 10       # upscale factor vertical (makes thin text taller)
THRESH_FRAC  = 0.55     # binarization threshold (fraction of intensity range)
TMPL_W       = 80       # standard template width (px)
TMPL_H       = 160      # standard template height (px)
MIN_BLOB_COL = 15       # min column width to count as a digit blob

# Spatial filter: real digit blobs are confined to 28–72% of the upscaled
# binary band width. Noise blobs from film damage land at ~8% or ~88% —
# filtering these out before blob counting removes the root cause of failures.
DIGIT_X_MIN_FRAC = 0.25
DIGIT_X_MAX_FRAC = 0.78


def get_binary_text(frame_crop, H, fw):
    """Return upscaled binary image of the CBD text region."""
    y0 = int(H * TEXT_Y0_FRAC)
    y1 = int(H * TEXT_Y1_FRAC)
    x0 = int(fw * TEXT_X0_FRAC)
    x1 = int(fw * TEXT_X1_FRAC)
    band  = frame_crop[y0:y1, x0:x1]
    big   = np.array(Image.fromarray(band).resize(
        (band.shape[1] * SCALE_X, band.shape[0] * SCALE_Y), Image.LANCZOS))
    lo, hi = int(big.min()), int(big.max())
    thresh = lo + (hi - lo) * THRESH_FRAC
    return (big < thresh).astype(np.float32)


def find_digit_col_ranges(binary):
    """
    Find x-ranges of individual digit blobs via column projection.
    Returns list of (x_start, x_end) sorted left-to-right.

    Spatial filter: only blobs whose centre falls in [DIGIT_X_MIN_FRAC,
    DIGIT_X_MAX_FRAC] of the band width are kept. This eliminates noise marks
    at the film edges that have no CBD content.

    Narrow-blob merge: digits like "4" have a thin horizontal bar that causes
    the column projection to dip mid-digit, splitting it into two narrow blobs.
    Adjacent blobs that are each <100 px wide and together <250 px are merged.
    """
    bw       = binary.shape[1]
    x_min    = DIGIT_X_MIN_FRAC * bw
    x_max    = DIGIT_X_MAX_FRAC * bw

    col_proj  = binary.sum(axis=0)
    smoothed  = _gauss_smooth(col_proj, sigma=3)
    threshold = smoothed.max() * 0.08
    is_ink    = smoothed > threshold
    runs      = _connected_components_1d(is_ink)
    blobs     = [(s, e) for s, e in runs if (e - s + 1) >= MIN_BLOB_COL]

    # Spatial filter: keep only blobs whose centre is in the digit region
    blobs = [(s, e) for s, e in blobs if x_min <= (s + e) / 2 <= x_max]

    # Merge adjacent narrow blobs that are parts of one split digit (e.g. "4").
    # A normal digit is ~150-200 px wide; split halves are <100 px each.
    merged = True
    while merged:
        merged = False
        new_blobs = []
        i = 0
        while i < len(blobs):
            if i + 1 < len(blobs):
                s1, e1 = blobs[i]
                s2, e2 = blobs[i + 1]
                w1, w2 = e1 - s1 + 1, e2 - s2 + 1
                gap = s2 - e1
                combined = e2 - s1 + 1
                if w1 < 100 and w2 < 100 and combined < 250 and gap < 100:
                    new_blobs.append((s1, e2))
                    i += 2
                    merged = True
                    continue
            new_blobs.append(blobs[i])
            i += 1
        blobs = new_blobs

    return blobs


def extract_templates(img_norm, frames, H):
    """
    Extract and average per-digit templates from known frames.
    Returns dict {digit_char: (TMPL_H, TMPL_W) float32 array}
    """
    accum  = {d: [] for d in '0123456789'}

    for frame_idx, cbd in KNOWN_CBDS.items():
        full_str = f"{FLIGHT:03d}{cbd:04d}"   # e.g. "1250465"
        l, r = frames[frame_idx]
        crop = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
        fw   = crop.shape[1]

        binary = get_binary_text(crop, H, fw)
        blobs  = find_digit_col_ranges(binary)

        # We expect 7 blobs (one per character in "1250XXX")
        if len(blobs) < 5 or len(blobs) > 10:
            print(f"  Frame {frame_idx}: {len(blobs)} blobs — skipping "
                  f"(expected ~7 for '{full_str}')")
            continue

        blobs = adjust_blobs_to_7(blobs)

        print(f"  Frame {frame_idx} (CBD {cbd:04d}): '{full_str}'  "
              f"{len(blobs)} blobs -> match")

        for (s, e), digit_char in zip(blobs, full_str):
            pad = 4
            crop_d = binary[:, max(0, s-pad): e+pad+1]
            if crop_d.shape[1] < 5 or crop_d.shape[0] < 5:
                continue
            resized = np.array(Image.fromarray(
                (crop_d * 255).astype(np.uint8)).resize(
                (TMPL_W, TMPL_H), Image.LANCZOS), dtype=np.float32) / 255.0
            accum[digit_char].append(resized)

    # Average multiple examples of each digit
    templates = {}
    for d, crops in accum.items():
        if crops:
            templates[d] = np.mean(crops, axis=0).astype(np.float32)
            print(f"  Digit '{d}': {len(crops)} examples averaged")
        else:
            print(f"  Digit '{d}': NO examples — missing from this TIFF!")
    return templates


def adjust_blobs_to_7(blobs):
    """
    Adjust a list of digit blob column ranges to exactly 7 entries.

    If > 7: merge the pair with the smallest gap (repeatedly).
    If < 7 (but >= 5): split the widest blob at its midpoint (once).
    If < 5 or > 10: return unchanged (caller should handle gracefully).

    Returns the adjusted list of (start, end) tuples.
    """
    if len(blobs) < 5 or len(blobs) > 10:
        return blobs
    blobs = list(blobs)
    while len(blobs) > 7:
        gaps = [(blobs[k+1][0] - blobs[k][1]) for k in range(len(blobs)-1)]
        m = int(np.argmin(gaps))
        merged = (blobs[m][0], blobs[m+1][1])
        blobs = blobs[:m] + [merged] + blobs[m+2:]
    if len(blobs) < 7:
        ws = [e - s for s, e in blobs]
        m = int(np.argmax(ws))
        s, e = blobs[m]
        mid = (s + e) // 2
        blobs = blobs[:m] + [(s, mid), (mid+1, e)] + blobs[m+1:]
    return blobs


def ncc_match(query, template):
    """Normalized cross-correlation between two equal-sized float arrays."""
    q = query   - query.mean()
    t = template - template.mean()
    denom = (np.linalg.norm(q) * np.linalg.norm(t)) + 1e-9
    return float(np.sum(q * t) / denom)


def recognize_frame(frame_crop, H, fw, templates):
    """
    Recognize the 7-digit number from a frame using NCC template matching.
    Returns (recognized_string, blobs) or (None, blobs) if recognition fails.
    """
    binary = get_binary_text(frame_crop, H, fw)
    blobs  = find_digit_col_ranges(binary)   # already spatially filtered

    if len(blobs) < 5:
        return None, blobs

    blobs = adjust_blobs_to_7(blobs)

    recognized = ""
    for s, e in blobs:
        pad    = 4
        crop_d = binary[:, max(0, s-pad): e+pad+1]
        if crop_d.shape[1] < 3:
            recognized += "?"
            continue
        query = np.array(Image.fromarray(
            (crop_d * 255).astype(np.uint8)).resize(
            (TMPL_W, TMPL_H), Image.LANCZOS), dtype=np.float32) / 255.0

        scores = {d: ncc_match(query, t) for d, t in templates.items()}
        best   = max(scores, key=scores.get)
        recognized += best

    return recognized, blobs


def apply_sequential_constraint(frame_indices, raw_reads, flight=None,
                                confidences=None):
    """
    Use the physical constraint |CBD[n+1] - CBD[n]| = 1 to validate and
    correct NCC recognition results.

    Parameters
    ----------
    frame_indices : list[int]
        Frame index (position in TIFF) for each complete frame, in order.
    raw_reads : list[str or None]
        NCC-recognized 7-char string (e.g. "1250465") per complete frame.
        None if recognition returned no result.
    flight : int or None
        If given, force the flight number prefix (e.g. 125) instead of
        majority-voting across OCR reads.  This eliminates a common failure
        mode where the flight digits are misread (e.g. "025" instead of "125"),
        which prevents correct CBD anchoring.
    confidences : list[float] or None
        Per-frame confidence scores (0.0-1.0).  When provided, anchor
        agreement is weighted by confidence instead of binary counting.
        Frames with confidence < 0.3 are excluded as candidate anchors.
        When None (default), all valid anchors are weighted equally (1.0).

    Returns
    -------
    corrected : list[str]
        Corrected 7-char strings. Frames that couldn't be corrected keep
        their raw read (or None).
    n_anchors : int
        Number of raw reads that agreed with the best-fit sequence.
    """
    # Parse CBD (last 4 digits) from each raw read — ignore the flight prefix
    # so that a misread flight number doesn't prevent anchoring.
    raw_cbds  = []
    flt_votes = {}
    for read in raw_reads:
        if read and len(read) >= 7:
            try:
                raw_cbds.append(int(read[-4:]))
                prefix = read[:3]
                if prefix.isdigit():
                    flt_votes[prefix] = flt_votes.get(prefix, 0) + 1
            except ValueError:
                raw_cbds.append(None)
        else:
            raw_cbds.append(None)

    # Flight string: use forced value if given, otherwise majority vote
    if flight is not None:
        flt_str = f"{flight:03d}"
    else:
        flt_str = max(flt_votes, key=flt_votes.get) if flt_votes else f"{FLIGHT:03d}"

    # valid = list of (list_pos, frame_idx, cbd_int)
    # When confidences are provided, filter out low-confidence anchors
    conf_threshold = 0.3
    valid = []
    for k, (fi, cbd) in enumerate(zip(frame_indices, raw_cbds)):
        if cbd is None:
            continue
        if confidences is not None and confidences[k] < conf_threshold:
            continue
        valid.append((k, fi, cbd))

    if len(valid) < 2:
        return raw_reads, len(valid)

    # Try both directions (+1 ascending, -1 descending); find best anchor
    # When confidences are provided, weight each agreeing anchor by its
    # confidence instead of counting each as 1.
    best_score        = 0
    best_anchor_fi    = frame_indices[0]
    best_anchor_cbd   = raw_cbds[0] if raw_cbds[0] is not None else 0
    best_direction    = 1

    for direction in (+1, -1):
        for _, anchor_fi, anchor_cbd in valid:
            score = sum(
                (confidences[k] if confidences is not None else 1)
                for k, fi, cbd in valid
                if cbd == anchor_cbd + direction * (fi - anchor_fi)
            )
            if score > best_score:
                best_score      = score
                best_anchor_fi  = anchor_fi
                best_anchor_cbd = anchor_cbd
                best_direction  = direction

    # Count integer anchors for reporting (how many frames agreed)
    n_anchors = sum(
        1 for _, fi, cbd in valid
        if cbd == best_anchor_cbd + best_direction * (fi - best_anchor_fi)
    )

    # Assign corrected CBD to every complete frame
    corrected = []
    for fi in frame_indices:
        expected_cbd = best_anchor_cbd + best_direction * (fi - best_anchor_fi)
        corrected.append(f"{flt_str}{expected_cbd:04d}")

    return corrected, n_anchors


# -- Main --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nLoading {TIFF.name} ...")
    Image.MAX_IMAGE_PIXELS = None
    img      = np.array(Image.open(TIFF), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W     = img_norm.shape
    frames   = detect_frames(img_norm)
    widths   = [r - l for l, r in frames]
    med_w    = float(np.median(widths))
    print(f"  {len(frames)} frames detected\n")

    # -- Step 1: build templates -------------------------------------------
    print("Building digit templates from known frames ...")
    templates = extract_templates(img_norm, frames, H)
    np.save(TMPL, templates)
    print(f"\n  Templates saved -> {TMPL.relative_to(ROOT)}")
    print(f"  Digits with templates: {sorted(templates.keys())}\n")

    # -- Step 2: raw NCC recognition on all complete frames ----------------
    print("Stage 1 — NCC recognition (raw) ...")
    complete_frame_indices = []
    raw_reads_list         = []

    for i, (l, r) in enumerate(frames):
        crop  = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
        fw    = crop.shape[1]
        ptype = "partial" if fw < 0.75 * med_w else "complete"
        if ptype == "complete":
            recog, _ = recognize_frame(crop, H, fw, templates)
            complete_frame_indices.append(i)
            raw_reads_list.append(recog)

    # -- Step 3: sequential constraint correction --------------------------
    print("Stage 2 — Sequential constraint correction ...")
    corrected_list, n_anchors = apply_sequential_constraint(
        complete_frame_indices, raw_reads_list
    )
    print(f"  Anchor agreement: {n_anchors}/{len(complete_frame_indices)} "
          f"raw reads consistent with best-fit sequence\n")

    # -- Step 4: report final results --------------------------------------
    print(f"  {'Fr':>3}  {'Raw NCC':>10}  {'Corrected':>10}  "
          f"{'Expected':>10}  {'W':>5}  {'Type':>8}  Status")
    print("  " + "-" * 72)

    comp_iter = iter(zip(complete_frame_indices, raw_reads_list, corrected_list))

    for i, (l, r) in enumerate(frames):
        fw    = r - l
        ptype = "partial" if fw < 0.75 * med_w else "complete"

        if ptype == "partial":
            print(f"  {i:>3}  {'—':>10}  {'—':>10}  {'—':>10}  "
                  f"{fw:>5}  {ptype:>8}")
            continue

        fi, raw, corr = next(comp_iter)
        expected_cbd = KNOWN_CBDS.get(i)
        expected_str = f"{FLIGHT:03d}{expected_cbd:04d}" if expected_cbd else "?"
        correct      = (corr == expected_str) if expected_cbd else None
        status       = "[OK]" if correct is True else ("[X]" if correct is False else "—")
        print(f"  {i:>3}  {str(raw):>10}  {str(corr):>10}  "
              f"{expected_str:>10}  {fw:>5}  {ptype:>8}  {status}")

    # -- Step 5: final metadata summary -----------------------------------
    print("\nFrame metadata summary (LYRA output):")
    print(f"  {'Fr':>3}  {'Type':>8}  {'CBD':>8}")
    print("  " + "-" * 25)

    comp_iter2 = iter(zip(complete_frame_indices, corrected_list))
    for i in range(len(frames)):
        fw    = frames[i][1] - frames[i][0]
        ptype = "partial" if fw < 0.75 * med_w else "complete"
        if ptype == "partial":
            print(f"  {i:>3}  {ptype:>8}  {'—':>8}")
        else:
            fi, corr = next(comp_iter2)
            cbd_str  = corr[3:] if corr and len(corr) >= 7 else "?"
            print(f"  {i:>3}  {ptype:>8}  {cbd_str:>8}")

    # -- Step 6: template contact sheet figure ----------------------------
    digits_sorted = sorted(templates.keys())
    nd   = len(digits_sorted)
    ncol = 5
    nrow = int(np.ceil(nd / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 3.5))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten()

    for k, d in enumerate(digits_sorted):
        ax = axes_flat[k]
        ax.imshow(templates[d], cmap="gray", vmin=0, vmax=1, aspect="auto")
        n_ex = len([f for f in KNOWN_CBDS
                    if d in f"{FLIGHT:03d}{KNOWN_CBDS[f]:04d}"])
        ax.set_title(f"Digit '{d}'  ({n_ex} ex)", fontsize=10, fontweight="bold")
        ax.axis("off")

    for ax in axes_flat[nd:]:
        ax.set_visible(False)

    fig.suptitle(
        "LYRA digit templates — 7-segment font\n"
        f"Source: {TIFF.name}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "templates_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Template figure -> {display_path(fig_path, ROOT)}")
    print("\nDone.")
''',
    'build_pdf': r'''#!/usr/bin/env python3
"""
LYRA A-scope Data Product PDF generator.

Produces a CReSIS OPR-style PDF for each TIFF, with:
  - Page 1: Cover map (flight track, TIFF segment, start point)
  - Pages 2+: One per complete frame (location map + calibrated A-scope)

Usage:
    python codes/build_pdf.py <tiff> [--output <folder>] [--nav <folder>]

Requires phases 1-4 complete for the TIFF.
"""

import json
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# -- Resolve project root and add tools/ to path ------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "LYRA"))
sys.path.insert(0, str(ROOT / "tools"))

from lyra import (display_path, output_root, parse_tiff_filename, tiff_id, resolve_tiff_arg, load_stanford_nav,
                   get_nav_positions, load_alignment)
from plot_flight_tracks import (load_basemap_ps, _to_ps, _graticule_line,
                                 _meridian_line)

# -- Parse args ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Build a PDF data product for one TIFF")
parser.add_argument("tiff", help="TIFF path or FLIGHT/TIFF_ID shorthand")
parser.add_argument("--output", type=Path, default=None,
                    help="Output root; an F<flight> folder is created inside it")
parser.add_argument("--nav", type=Path, default=None,
                    help="Folder containing the flight navigation CSV")
args = parser.parse_args()
if args.output is not None:
    os.environ["LYRA_OUTPUT_DIR"] = str(args.output.expanduser().resolve())
if args.nav is not None:
    os.environ["LYRA_NAV_DIR"] = str(args.nav.expanduser().resolve())

TIFF = resolve_tiff_arg(args.tiff, ROOT)
_metadata = parse_tiff_filename(TIFF)
FLT  = _metadata[0] if _metadata else int(TIFF.parent.name)
TID  = tiff_id(TIFF)

OUT_DIR    = output_root(ROOT / "output") / f"F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PDF_DIR    = OUT_DIR / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# -- Load data ----------------------------------------------------------------
frame_idx_df = pd.read_csv(PHASE1_DIR / f"F{FLT}_frame_index.csv")
cal_df       = pd.read_csv(OUT_DIR / "phase3" / f"F{FLT}_cal.csv")
echoes_df    = pd.read_csv(OUT_DIR / "phase4" / f"F{FLT}_echoes.csv")

# Cal picks JSON (for X-grid phase / D_anchor)
cal_picks_path = PHASE1_DIR / f"F{FLT}_cal_picks.json"
cal_picks = json.load(open(cal_picks_path)) if cal_picks_path.exists() else {}

# Filter to this TIFF
tiff_name = TIFF.name
fi_tiff   = frame_idx_df[frame_idx_df["tiff"] == tiff_name].copy()
cal_tiff  = cal_df[cal_df["tiff"] == tiff_name].copy()
echo_tiff = echoes_df[echoes_df["tiff"] == tiff_name].copy()

# Navigation
nav = load_stanford_nav(FLT)
alignment = load_alignment(FLT, OUT_DIR)
nav_offset = alignment.offset if alignment else 0

# Full flight track for map background
all_cbds = echoes_df["cbd"].dropna().values.astype(int)
flight_lats, flight_lons = get_nav_positions(all_cbds, FLT, offset=nav_offset)
flight_x, flight_y = _to_ps(flight_lons, flight_lats)

# This TIFF's CBDs
tiff_lats = echo_tiff["lat"].values
tiff_lons = echo_tiff["lon"].values
tiff_x, tiff_y = _to_ps(tiff_lons, tiff_lats)

# Load basemap (land + ice shelves)
basemap = load_basemap_ps()
ice_shelf_path = ROOT / "Data" / "ne_10m_antarctic_ice_shelves_polys.shp"
ice_shelves = None
if ice_shelf_path.exists():
    ice_shelves = gpd.read_file(ice_shelf_path).to_crs(epsg=3031)

# Load TIFF image
Image.MAX_IMAGE_PIXELS = None
img_raw = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-9)
IMG_H, IMG_W = img_norm.shape

# Vertical crop: remove sprocket holes but keep CBD number text (~row 2000-2100)
# Top: film border ends ~row 140, CRT display starts ~150
# Bottom: CRT display ends ~row 2160, sprocket holes start ~row 2200
CRT_TOP    = 150
CRT_BOTTOM = 2170
CRT_H      = CRT_BOTTOM - CRT_TOP

# Compute D_anchor (X-grid phase) from cal_picks
# D_anchor = distance from mb_x to the first graticule line to its right
tiff_d_anchor = None
for fkey, picks in cal_picks.items():
    if ("mb" in picks and picks["mb"] is not None
            and "x_grid" in picks and picks["x_grid"]
            and len(picks["x_grid"]) >= 1):
        tiff_d_anchor = float(min(picks["x_grid"])) - float(picks["mb"])
        break

print(f"\nLYRA PDF -- F{FLT} TIFF {TID}")
print(f"  Frames: {len(fi_tiff)} total, {len(echo_tiff)} with echoes")
print(f"  D_anchor: {tiff_d_anchor:.1f} px" if tiff_d_anchor else "  D_anchor: (none)")
if ice_shelves is not None:
    print(f"  Ice shelves: {len(ice_shelves)} polygons loaded")

# -- Style constants ----------------------------------------------------------
Y_REF_DB     = -60.0   # dB at y_ref_px (noise floor reference)
CLR_SURFACE  = "#2166ac"
CLR_BED      = "#b2182b"
CLR_MB       = "#4daf4a"
CLR_GRID     = "#666666"
CLR_TRACK    = "#aaaaaa"
CLR_TIFF_SEG = "#4393c3"
CLR_FRAME    = "#d6604d"
CLR_SHELF    = "#d4e6f1"    # light blue for ice shelves
CLR_LAND     = "#e8e8e8"
CLR_OCEAN    = "#f7fbff"    # very faint blue for ocean
STATUS_CLR   = {"good": "#1b9e77", "weak_bed": "#d95f02",
                "no_bed": "#d62728", "no_surface": "#7570b3"}
FONT_LABEL   = 8
FONT_TITLE   = 10
FONT_SMALL   = 7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": FONT_LABEL,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# -- Helper: draw polar stereographic map ------------------------------------
def draw_map(ax, center_x, center_y, radius_m=100_000,
             show_flight=True, show_tiff=True, highlight_xy=None):
    """Draw a polar stereographic map centered on (center_x, center_y)."""
    ax.set_facecolor(CLR_OCEAN)
    ax.set_xlim(center_x - radius_m, center_x + radius_m)
    ax.set_ylim(center_y - radius_m, center_y + radius_m)

    # Ice shelves (below land)
    if ice_shelves is not None:
        ice_shelves.plot(ax=ax, color=CLR_SHELF, edgecolor="#b0c4de",
                         linewidth=0.3, zorder=1)

    # Land basemap
    basemap.plot(ax=ax, color=CLR_LAND, edgecolor="#bbbbbb", linewidth=0.3,
                 zorder=2)

    # Graticule
    for lat in range(-90, -60, 2):
        gx, gy = _graticule_line(np.linspace(0, 360, 500), lat)
        ax.plot(gx, gy, color="#cccccc", lw=0.25, zorder=3)
    for lon in range(0, 360, 15):
        mx, my = _meridian_line(np.linspace(-90, -60, 200), lon)
        ax.plot(mx, my, color="#cccccc", lw=0.25, zorder=3)

    # Full flight track
    if show_flight:
        valid = np.isfinite(flight_x) & np.isfinite(flight_y)
        ax.plot(flight_x[valid], flight_y[valid], color=CLR_TRACK,
                lw=0.8, zorder=4, alpha=0.6)

    # This TIFF's segment
    if show_tiff:
        valid_t = np.isfinite(tiff_x) & np.isfinite(tiff_y)
        ax.plot(tiff_x[valid_t], tiff_y[valid_t], color=CLR_TIFF_SEG,
                lw=2.0, zorder=5)

    # Highlight point (current frame)
    if highlight_xy is not None:
        hx, hy = highlight_xy
        ax.plot(hx, hy, "o", color=CLR_FRAME, ms=6, zorder=7,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Easting (km)", fontsize=FONT_SMALL)
    ax.set_ylabel("Northing (km)", fontsize=FONT_SMALL)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.tick_params(labelsize=FONT_SMALL)


# -- Helper: draw calibrated A-scope frame -----------------------------------
def draw_ascope(ax, frame_row, cal_row, echo_row):
    """Draw calibrated A-scope CRT image with grid overlays and echo markers."""
    left_px  = int(frame_row["left_px"])
    right_px = int(frame_row["right_px"])
    mb_x     = int(float(cal_row["mb_x"]))
    y_ref_px = float(cal_row["y_ref_px"])
    db_px    = float(cal_row["db_per_px"])
    us_px    = float(cal_row["us_per_px"])
    x_sp     = float(cal_row["x_spacing_px"])
    hgrid_sp = float(cal_row["hgrid_spacing"])

    frame_w  = right_px - left_px

    # -- Horizontal trim: center the CRT signal region --
    # Signal extends from MB to approximately MB + 17 us (max sweep)
    signal_end_px = mb_x + int(17.0 / us_px)
    signal_end_px = min(signal_end_px, frame_w)
    signal_span   = signal_end_px - mb_x

    # Add padding: 15% of signal span on left of MB, 10% on right
    pad_left  = max(50, int(signal_span * 0.15))
    pad_right = max(50, int(signal_span * 0.10))
    trim_left  = max(0, mb_x - pad_left)
    trim_right = min(frame_w, signal_end_px + pad_right)

    # Extract cropped frame (vertical: CRT region only)
    frame_crop = img_norm[CRT_TOP:CRT_BOTTOM,
                          left_px + trim_left : left_px + trim_right]

    # Display image
    ax.imshow(frame_crop, cmap="gray", aspect="auto", origin="upper",
              extent=[trim_left, trim_right, CRT_BOTTOM, CRT_TOP])

    # -- X grid lines: aligned to oscilloscope graticule --
    # Use D_anchor to find true graticule phase
    if tiff_d_anchor is not None:
        # First grid line right of MB
        first_xg = mb_x + tiff_d_anchor
        # Extrapolate in both directions
        xg = first_xg
        while xg > trim_left:
            xg -= x_sp
        xg += x_sp  # first visible grid line
        while xg < trim_right:
            if xg >= trim_left:
                ax.axvline(xg, color=CLR_GRID, lw=0.5, ls=":", alpha=0.45)
            xg += x_sp
    else:
        # Fallback: step from MB
        xg = mb_x
        while xg < trim_right:
            if xg >= trim_left:
                ax.axvline(xg, color=CLR_GRID, lw=0.5, ls=":", alpha=0.45)
            xg += x_sp

    # -- Y grid lines: anchored to y_ref_px, step by hgrid_spacing --
    yg = y_ref_px
    while yg > CRT_TOP:
        yg -= hgrid_sp
    while yg < CRT_BOTTOM:
        if yg >= CRT_TOP:
            ax.axhline(yg, color=CLR_GRID, lw=0.5, ls=":", alpha=0.45)
        yg += hgrid_sp

    # Label style: white background box so text is readable over the image/line
    _label_box = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85)

    # -- Main bang marker --
    if trim_left <= mb_x <= trim_right:
        ax.axvline(mb_x, color=CLR_MB, lw=0.8, ls="--", alpha=0.6)
        ax.text(mb_x, CRT_TOP + 5, " MB ", fontsize=8, color=CLR_MB,
                fontweight="bold", ha="center", va="top", bbox=_label_box)

    # -- Echo markers --
    status = str(echo_row["echo_status"]) if echo_row is not None else "no_data"

    if echo_row is not None:
        # Surface
        srf_twt = echo_row.get("surface_twt_us")
        if pd.notna(srf_twt):
            srf_x = mb_x + float(srf_twt) / us_px
            if trim_left <= srf_x <= trim_right:
                ax.axvline(srf_x, color=CLR_SURFACE, lw=1.2, ls="-", alpha=0.7)
                ax.text(srf_x, CRT_TOP + 5, " S ", fontsize=9,
                        color=CLR_SURFACE, fontweight="bold",
                        ha="center", va="top", bbox=_label_box)

        # Bed
        bed_twt = echo_row.get("bed_twt_us")
        if pd.notna(bed_twt):
            bed_x = mb_x + float(bed_twt) / us_px
            if trim_left <= bed_x <= trim_right:
                ax.axvline(bed_x, color=CLR_BED, lw=1.2, ls="-", alpha=0.7)
                ax.text(bed_x, CRT_TOP + 5, " B ", fontsize=9,
                        color=CLR_BED, fontweight="bold",
                        ha="center", va="top", bbox=_label_box)

    # -- Axes --
    ax.set_xlim(trim_left, trim_right)
    ax.set_ylim(CRT_BOTTOM, CRT_TOP)

    # Primary X: column px
    ax.set_xlabel("Column (px, frame-relative)", fontsize=FONT_SMALL)
    ax.tick_params(labelsize=FONT_SMALL)

    # Secondary X: travel time from MB
    ax2_x = ax.twiny()
    ax2_x.set_xlim((trim_left - mb_x) * us_px, (trim_right - mb_x) * us_px)
    ax2_x.set_xlabel("Two-way travel time from MB (us)", fontsize=FONT_SMALL)
    ax2_x.tick_params(labelsize=FONT_SMALL)
    ax2_x.spines["top"].set_linewidth(0.4)

    # Primary Y: row px
    ax.set_ylabel("Row (px)", fontsize=FONT_SMALL)

    # Secondary Y: power dB
    ax2_y = ax.twinx()
    db_top    = Y_REF_DB + (y_ref_px - CRT_TOP) * db_px
    db_bottom = Y_REF_DB + (y_ref_px - CRT_BOTTOM) * db_px
    ax2_y.set_ylim(db_bottom, db_top)
    ax2_y.set_ylabel("Received power (dB)", fontsize=FONT_SMALL)
    ax2_y.tick_params(labelsize=FONT_SMALL)
    ax2_y.spines["right"].set_linewidth(0.4)

    # Spine visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -- Quality badge --
    badge_clr = STATUS_CLR.get(status, "#999999")
    badge_txt = status.replace("_", " ")
    ax.text(0.98, 0.97, badge_txt, transform=ax.transAxes,
            fontsize=FONT_SMALL, fontweight="bold", color="white",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_clr,
                      edgecolor="none", alpha=0.85))

    return status


# -- Build PDF ----------------------------------------------------------------
pdf_path = PDF_DIR / f"F{FLT}_{TID}_ascope.pdf"
print(f"  Output: {display_path(pdf_path, ROOT)}")

with PdfPages(str(pdf_path)) as pdf:
    # ========== PAGE 1: COVER MAP ============================================
    fig_cover = plt.figure(figsize=(11, 8.5))

    fig_cover.suptitle(
        f"SPRI/TUD/NSF 60 MHz Airborne Radar  --  Flight F{FLT}, TIFF {TID}",
        fontsize=12, fontweight="bold", y=0.97)

    # Right panel: zoomed map centered on this TIFF's segment
    ax_main = fig_cover.add_axes([0.35, 0.08, 0.60, 0.82])
    valid_t = np.isfinite(tiff_x) & np.isfinite(tiff_y)
    if valid_t.any():
        cx = np.nanmean(tiff_x[valid_t])
        cy = np.nanmean(tiff_y[valid_t])
        dx = np.nanmax(tiff_x[valid_t]) - np.nanmin(tiff_x[valid_t])
        dy = np.nanmax(tiff_y[valid_t]) - np.nanmin(tiff_y[valid_t])
        radius = max(dx, dy) * 0.7 + 50_000
    else:
        cx, cy, radius = 0, 0, 500_000

    draw_map(ax_main, cx, cy, radius_m=radius)
    # Start marker
    if valid_t.any():
        i0 = np.where(valid_t)[0][0]
        ax_main.plot(tiff_x[i0], tiff_y[i0], "s", color="#1b9e77", ms=8,
                     zorder=6, markeredgecolor="white", markeredgewidth=0.8,
                     label="Start")
        ax_main.legend(loc="lower right", fontsize=FONT_SMALL, framealpha=0.9)

    # Left panel: Antarctica inset
    ax_inset = fig_cover.add_axes([0.04, 0.45, 0.28, 0.45])
    ax_inset.set_facecolor(CLR_OCEAN)
    if ice_shelves is not None:
        ice_shelves.plot(ax=ax_inset, color=CLR_SHELF, edgecolor="none",
                         linewidth=0)
    basemap.plot(ax=ax_inset, color=CLR_LAND, edgecolor="#aaaaaa", linewidth=0.3)
    valid_f = np.isfinite(flight_x) & np.isfinite(flight_y)
    ax_inset.plot(flight_x[valid_f], flight_y[valid_f], color=CLR_TRACK,
                  lw=0.5, alpha=0.5)
    ax_inset.plot(tiff_x[valid_t], tiff_y[valid_t], color=CLR_TIFF_SEG, lw=1.5)
    ax_inset.set_xlim(-3e6, 3e6)
    ax_inset.set_ylim(-3e6, 3e6)
    ax_inset.set_aspect("equal")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title("Antarctica", fontsize=FONT_SMALL)
    for sp in ax_inset.spines.values():
        sp.set_linewidth(0.4)

    # Metadata text block
    n_complete = len(fi_tiff[fi_tiff["frame_type"] == "complete"])
    n_good     = len(echo_tiff[echo_tiff["echo_status"] == "good"])
    cbd_min    = echo_tiff["cbd"].min() if len(echo_tiff) else "?"
    cbd_max    = echo_tiff["cbd"].max() if len(echo_tiff) else "?"

    meta_lines = [
        f"Flight: F{FLT}",
        f"TIFF: {tiff_name}",
        f"TIFF ID: {TID}",
        f"CBD range: {cbd_min} -- {cbd_max}",
        f"Frames: {n_complete} complete, {n_good} good",
        f"Projection: EPSG:3031 (NSIDC Polar Stereo.)",
        f"Radar: 60 MHz, 1 kW, 125 ns pulse",
        f"",
        f"LYRA A-scope Data Product",
        f"FrozenLegacies Project",
    ]
    fig_cover.text(0.04, 0.35, "\n".join(meta_lines),
                   fontsize=FONT_LABEL, verticalalignment="top",
                   fontfamily="sans-serif", linespacing=1.6)

    pdf.savefig(fig_cover, dpi=150)
    plt.close(fig_cover)
    print(f"  Cover page done")

    # ========== PAGES 2+: ONE PER COMPLETE FRAME =============================
    complete_fi = fi_tiff[fi_tiff["frame_type"] == "complete"].copy()
    n_pages = len(complete_fi)

    for page_i, (_, fi_row) in enumerate(complete_fi.iterrows()):
        fidx = int(fi_row["frame_idx"])
        cbd  = fi_row["cbd"]
        cbd_str = f"{int(float(cbd)):04d}" if pd.notna(cbd) else "????"

        # Match cal and echo rows
        cal_match = cal_tiff[cal_tiff["frame_idx"] == fidx]
        if len(cal_match) == 0:
            continue
        cal_row = cal_match.iloc[0]

        # Skip excluded frames
        if pd.notna(cal_row.get("exclude_reason")) and str(cal_row.get("exclude_reason")).strip():
            continue

        echo_match = echo_tiff[echo_tiff["frame_idx"] == fidx]
        echo_row = echo_match.iloc[0] if len(echo_match) else None

        # Get position
        lat = float(echo_row["lat"]) if echo_row is not None and pd.notna(echo_row.get("lat")) else None
        lon = float(echo_row["lon"]) if echo_row is not None and pd.notna(echo_row.get("lon")) else None

        # Create figure
        fig = plt.figure(figsize=(11, 8.5))

        status_str = str(echo_row["echo_status"]) if echo_row is not None else "no data"
        fig.suptitle(
            f"F{FLT} / TIFF {TID} / CBD {cbd_str}   ({page_i+1}/{n_pages})",
            fontsize=FONT_TITLE, fontweight="bold", y=0.97)

        # -- Left panel: location map (30% width) --
        ax_map = fig.add_axes([0.04, 0.12, 0.26, 0.78])

        if lat is not None and lon is not None:
            fx, fy = _to_ps(np.array([lon]), np.array([lat]))
            draw_map(ax_map, fx[0], fy[0], radius_m=80_000,
                     highlight_xy=(fx[0], fy[0]))
        else:
            ax_map.text(0.5, 0.5, "No position", transform=ax_map.transAxes,
                        ha="center", va="center", fontsize=FONT_LABEL,
                        color="#999999")
            ax_map.set_xticks([])
            ax_map.set_yticks([])

        # Frame metadata as title (above map)
        meta = f"CBD {cbd_str}"
        if echo_row is not None:
            h_air = echo_row.get("h_air_m")
            h_ice = echo_row.get("h_ice_m")
            if pd.notna(h_air):
                meta += f"  |  h_air {h_air:.0f} m"
            if pd.notna(h_ice):
                meta += f"  |  h_ice {h_ice:.0f} m"
        ax_map.set_title(meta, fontsize=FONT_SMALL, pad=6)

        # Lat/lon below map (inside axes area to avoid overlap)
        if lat is not None and lon is not None:
            coord_str = (f"{abs(lat):.3f}{'S' if lat < 0 else 'N'}, "
                         f"{abs(lon):.3f}{'E' if lon >= 0 else 'W'}")
            ax_map.text(0.5, 0.02, coord_str, transform=ax_map.transAxes,
                        fontsize=6, ha="center", va="bottom", color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.8))

        # -- Right panel: calibrated A-scope (62% width) --
        ax_asc = fig.add_axes([0.38, 0.12, 0.58, 0.76])
        draw_ascope(ax_asc, fi_row, cal_row, echo_row)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        if (page_i + 1) % 50 == 0 or page_i == 0:
            print(f"    Page {page_i+1}/{n_pages}: CBD {cbd_str} [{status_str}]")

    print(f"  {n_pages} frame pages written")

print(f"\nDone. PDF: {pdf_path}")
''',
    'calibrate': r'''# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""
calibrate.py — LYRA Phase 3: Per-frame calibration
========================================================
For each complete frame in a TIFF, calibrates using:

  • Main bang column  (mb_x)      -> ASTRA-approved X anchor: t = 0 µs
  • Reference line row (y_ref_px) -> Y anchor: power = -60 dB
  • X grid lines                  -> ASTRA-approved pixel coordinates
  • Y grid lines                  -> ASTRA-approved pixel coordinates

Calibration values are NEVER assumed constant across frames.
Each frame gets its own (mb_x, y_ref_px, x_spacing_px, y_spacing_px).

ASTRA-approved grid and reference coordinates from
F{FLT}_cal_picks.json are used directly. The montage-approved X reference
is the main bang for each frame.

Quality gate
------------
Frames whose mb_power_dB deviates more than 8 dB below the TIFF median
are reported without changing the approved position.

Frames marked "exclude": true in the guide picks JSON are skipped entirely
(film defect, tilted graticule, or other unrecoverable artifact).

Usage
-----
Run from repo root:

    python codes/calibrate.py "path\\to\\F103-C1024_C1036.tiff"

Outputs
-------
Per-flight calibration CSV (updated incrementally):
    output/F{FLT}/phase3/F{FLT}_cal.csv
      Columns: flight, tiff_id, frame_idx, cbd, file_id,
               mb_x, mb_power_dB, y_ref_px, db_per_px, us_per_px,
               x_spacing_px, hgrid_spacing, cal_source_y,
               mb_detection_source, astra_pick_signature

Per-frame diagnostic figure:
    output/F{FLT}/phase3/F{FLT}_{file_id}_cal.png
      Red   line              -> ASTRA trace
      Red   dashed vertical   -> approved main bang (t = 0 µs)
      Cyan  dashed horizontal -> reference line (-60 dB)
      Gold  dashed horizontal -> approved Y grid lines
      Blue  dashed vertical   -> approved X grid lines
      White dotted            -> user guide picks (if provided)
"""

from pathlib import Path
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from lyra import (
    ASTRA_DEFAULT_CROP_END_Y, ASTRA_DEFAULT_CROP_START_Y,
    ASTRA_X_POINTS, ASTRA_Y_POINTS, astra_pick_signature,
    DEFAULT_CAL, detect_frames, extract_trace,
    detect_mb, detect_frame_calibration, px_to_db,
    _gauss_smooth, ensure_canonical_name, output_root, parse_tiff_filename,
    resolve_tiff_arg, tiff_id,
)

# -- Resolve TIFF path ----------------------------------------------------------
if len(sys.argv) > 1:
    TIFF = resolve_tiff_arg(sys.argv[1], ROOT)
else:
    sys.exit("ERROR: TIFF path is required. Usage: python calibrate.py <TIFF_PATH>")

# Rename to canonical format if needed (e.g. F141 files renamed from reel convention)
TIFF = ensure_canonical_name(TIFF)

_metadata = parse_tiff_filename(TIFF)
try:
    FLT = _metadata[0] if _metadata else int(TIFF.parent.name.lstrip("Ff"))
except ValueError:
    FLT = int(os.environ.get("LYRA_FLIGHT", "0"))
if FLT <= 0:
    sys.exit("ERROR: Cannot infer flight; use FLT/TIFF_ID (e.g. 127/4800) or tiff/127/.")


TIFF_ID   = tiff_id(TIFF)
OUT_DIR   = output_root(ROOT / "output") / f"F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PHASE3_DIR = OUT_DIR / "phase3"
PHASE3_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV    = PHASE1_DIR / f"F{FLT}_frame_index.csv"
CAL_CSV      = PHASE3_DIR / f"F{FLT}_cal.csv"
PICKS_JSON   = PHASE1_DIR / f"F{FLT}_cal_picks.json"
DERIVED_JSON = PHASE1_DIR / f"F{FLT}_cal_derived.json"

# -- Load frame index -----------------------------------------------------------
if not INDEX_CSV.exists():
    sys.exit(f"ERROR: frame index not found at {INDEX_CSV}\n"
             "Run phase 1 (detect_frames.py) first.")

index     = pd.read_csv(INDEX_CSV, dtype=str)
tiff_rows = index[index["tiff"] == TIFF.name].copy()

if len(tiff_rows) == 0:
    sys.exit(f"ERROR: {TIFF.name} not found in frame index.\n"
             "Run phase 1 (detect_frames.py) for this TIFF first.")

print(f"\nLYRA Phase 3 — {TIFF.name}")
print(f"  Flight : F{FLT}  |  tiff_id : {TIFF_ID}")
print(f"  Frames in index: {len(tiff_rows)}  "
      f"(complete: {(tiff_rows.frame_type == 'complete').sum()})\n")

# -- Load guidance picks (imported from ASTRA by the flight runner) ------------
raw_picks:   dict = {}
derived_cal: dict = {}

if PICKS_JSON.exists():
    with open(PICKS_JSON) as f:
        raw_picks = json.load(f)
    for _pick in raw_picks.values():
        _pick.pop("mb", None)
        _pick.pop("mb_is_pick", None)
    print(f"  Guide picks   : {len(raw_picks)} frames  ({PICKS_JSON.name})")

if DERIVED_JSON.exists():
    with open(DERIVED_JSON) as f:
        derived_cal = json.load(f)
    print(f"  Derived cal   : x_spacing={derived_cal.get('x_spacing_px','?')} px  "
          f"y_spacing={derived_cal.get('y_spacing_px','?')} px")

if not raw_picks:
    print("  [INFO] No guide picks file found.")
print()

# -- Measure x_spacing_px from guide picks if not already in derived_cal -------
# derived_cal["x_spacing_px"] propagates (via _build_guides) to EVERY frame.
# Without this, all frames use DEFAULT_CAL's 205.54 px (measured on F125), which
# causes progressive drift on any TIFF with a different oscilloscope time base.
if not derived_cal.get("x_spacing_px") and raw_picks:
    _default_sp = 2.0 / DEFAULT_CAL["us_per_px"]   # F125 baseline spacing
    for _picks in raw_picks.values():
        _xg = _picks.get("x_grid")
        if _xg and len(_xg) >= 4:
            _xs = np.array(sorted(float(v) for v in _xg), dtype=float)
            _kk = np.array([round((x - _xs[0]) / _default_sp) for x in _xs], dtype=float)
            if len(np.unique(_kk)) < 2:
                continue
            derived_cal["x_spacing_px"] = abs(float(np.polyfit(_kk, _xs, 1)[0]))
            print(f"  x_spacing measured from guide picks : "
                  f"{derived_cal['x_spacing_px']:.2f} px/div  "
                  f"(DEFAULT_CAL was {_default_sp:.2f})")
            break

from collections import Counter as _Counter


def _norm_cbd(val):
    """Normalize CBD value to int (strips .0 from float CBDs)."""
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return None
    return int(float(val))


_tiff_cbds = [str(_norm_cbd(r["cbd"])) for _, r in tiff_rows.iterrows()
              if r["frame_type"] == "complete" and _norm_cbd(r["cbd"]) is not None]
_dup_cbds = {c for c, n in _Counter(_tiff_cbds).items() if n > 1}


def _cal_fkey(cbd, frame_idx):
    """Build pick-lookup key, disambiguating duplicate CBDs with frame_idx."""
    if cbd and str(cbd) in _dup_cbds:
        return f"CBD{cbd}_f{frame_idx}"
    key = f"CBD{cbd}" if cbd else f"fr{frame_idx}"
    if key not in raw_picks and cbd is not None:
        padded = f"CBD{int(cbd):04d}"
        if padded in raw_picks:
            return padded
    return key


missing_astra_picks = []
for _, pick_row in tiff_rows.iterrows():
    if pick_row["frame_type"] != "complete":
        continue
    pick_key = _cal_fkey(_norm_cbd(pick_row["cbd"]), pick_row["frame_idx"])
    frame_picks = raw_picks.get(pick_key, {})
    if (frame_picks.get("montage_approved") is not True
            or frame_picks.get("x_ref") is None or frame_picks.get("ref") is None
            or not isinstance(frame_picks.get("x_grid"), list)
            or len(frame_picks["x_grid"]) != ASTRA_X_POINTS
            or not isinstance(frame_picks.get("y_grid"), list)
            or len(frame_picks["y_grid"]) != ASTRA_Y_POINTS):
        missing_astra_picks.append(pick_key)
if missing_astra_picks:
    preview = ", ".join(missing_astra_picks[:10])
    if len(missing_astra_picks) > 10:
        preview += f", and {len(missing_astra_picks) - 10} more"
    sys.exit(
        f"ERROR: {TIFF.name} is missing approved ASTRA picks for: {preview}.\n"
        "Run the LYRA flight workflow from phase 2 to capture the grids "
        "and reference lines before calibration."
    )

# -- Identify TIFF-level reference anchor --------------------------------------
# If any complete frame in this TIFF already has picks, propagate only its REF
# to frames without individual picks. The REF (noise floor row) is stable
# across frames and can be safely reused from truncated or full-width frames alike.
# Each frame's main bang is its approved X reference and is never propagated.
#
# Any complete frame with an ASTRA reference pick can serve as the reference
# anchor, including frames at truncated TIFF boundaries.
tiff_anchor: dict = {}
for _, anchor_row in tiff_rows.iterrows():
    if anchor_row["frame_type"] != "complete":
        continue
    # REF is stable and can come from ANY frame (including truncated start/end).
    # Width check is intentionally not applied here.
    cbd_a  = _norm_cbd(anchor_row["cbd"])
    fkey_a = _cal_fkey(cbd_a, anchor_row["frame_idx"])
    if fkey_a in raw_picks:
        p_a = raw_picks[fkey_a]
        if p_a.get("ref") is not None:
            tiff_anchor["ref"] = int(p_a["ref"])
    if tiff_anchor:
        print(f"  TIFF anchor   : {fkey_a}  ref={tiff_anchor['ref']}  "
              f"(reference propagated to frames without individual picks; "
              f"main bang is the approved X reference)")
        break


# -- Enforce minimum reference pick for y_ref calibration ----------------------
# The -60 dB reference row (y_ref_px) cannot be reliably auto-detected.
# A user pick for at least one complete frame in THIS TIFF is required so
# that y_ref_px propagates accurately to all unguided frames.
# Without this, the pre-bang 75th-percentile fallback produces erratic
# y_ref values (observed range: 836–1590 px vs correct ~1509 px).
if not tiff_anchor:
    tiff_cbds = [
        str(_norm_cbd(r["cbd"])) for _, r in tiff_rows.iterrows()
        if r["frame_type"] == "complete" and _norm_cbd(r["cbd"]) is not None
    ]
    cbd_range = (
        f"CBD{tiff_cbds[0]}–CBD{tiff_cbds[-1]}" if len(tiff_cbds) >= 2
        else f"CBD{tiff_cbds[0]}" if tiff_cbds
        else f"frames in {TIFF.name}"
    )
    picks_info = (
        f"Guide picks loaded cover: {sorted(raw_picks.keys())}"
        if raw_picks
        else "No guide picks file found."
    )
    sys.exit(
        f"\nERROR: No reference-line pick found for TIFF {TIFF_ID} ({cbd_range}).\n"
        f"  {picks_info}\n\n"
        f"  The -60 dB reference row (y_ref_px) cannot be reliably detected\n"
        f"  without an ASTRA reference-line pick for this TIFF.\n\n"
        f"  Fix:\n"
        f"    python LYRA.py --tiff \"{TIFF.parent}\" --resume-from 2\n\n"
        f"  In ASTRA, pick the references and grids on a complete frame from "
        f"{cbd_range}.\n"
        f"  LYRA will continue into calibration after the picks are saved.\n"
    )


def _build_guides(fkey: str) -> dict:
    """Assemble guide dict: per-frame picks -> tiff anchor ref -> global defaults.

    MB (main bang column): use each frame's approved X-reference position.

    REF (y_ref_px = -60 dB row): use per-frame pick, then tiff anchor, then
    default.  The noise floor row is stable within a TIFF, so propagation is safe.
    """
    g: dict = {}
    g["trace_context"] = f"{TIFF.name} {fkey} Phase 3"
    # Per-frame ASTRA picks carry the crop used to create their Y coordinates.
    per_frame = raw_picks.get(fkey, {})
    crop_start = per_frame.get("crop_start_y")
    crop_end = per_frame.get("crop_end_y")
    if crop_start is None and crop_end is None:
        g["crop_start_y"] = ASTRA_DEFAULT_CROP_START_Y
        g["crop_end_y"] = ASTRA_DEFAULT_CROP_END_Y
    elif crop_start is None or crop_end is None:
        raise ValueError(f"{fkey}: ASTRA picks must include both crop bounds")
    else:
        if (not float(crop_start).is_integer()
                or not float(crop_end).is_integer()
                or float(crop_start) < 0
                or float(crop_end) <= float(crop_start)):
            raise ValueError(f"{fkey}: invalid ASTRA crop bounds {crop_start}, {crop_end}")
        g["crop_start_y"] = int(crop_start)
        g["crop_end_y"] = int(crop_end)
    # Global spacing and x-offset from derived calibration
    if derived_cal.get("x_spacing_px"):
        g["x_spacing_px"] = float(derived_cal["x_spacing_px"])
    if derived_cal.get("y_spacing_px"):
        g["y_spacing_px"] = float(derived_cal["y_spacing_px"])
    # Per-frame ASTRA picks: use reference and approved grids.
    if per_frame.get("ref") is not None:
        g["ref"] = int(per_frame["ref"])
    if per_frame.get("x_ref") is not None:
        g["x_ref"] = int(per_frame["x_ref"])
        g["mb"] = g["x_ref"]
        g["mb_is_pick"] = True
    if per_frame.get("x_grid"):
        g["x_grid"] = [int(v) for v in per_frame["x_grid"]]
    if per_frame.get("y_grid"):
        g["y_grid"] = [int(v) for v in per_frame["y_grid"]]
    # Tiff anchor: propagate ref only if not already set from per-frame pick
    if "ref" not in g and tiff_anchor.get("ref") is not None:
        g["ref"] = int(tiff_anchor["ref"])
    return g


# -- Load TIFF image ------------------------------------------------------------
print("Loading TIFF ...")
Image.MAX_IMAGE_PIXELS = None
img      = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
H, W     = img_norm.shape
print(f"  Image  : {W} x {H} px")

# -- Pass 1: calibrate all frames -----------------------------------------------
# Collect (frame_idx -> data dict) for quality gate and figure generation.
# Frames marked exclude=true in guide picks are flagged and skipped.

cal_rows    = []
_frame_data = {}   # frame_idx -> dict(frame, cal, guides, cbd, fkey, file_id)

print(f"  {'Fr':>3}  {'CBD':>6}  {'mb_x':>6}  {'mb_dB':>7}  "
      f"{'y_ref':>6}  {'db/px':>7}  {'x_sp':>6}  {'Ygrid':>5}  ASTRA?")
print("  " + "-" * 72)

for _, row in tiff_rows.iterrows():
    frame_idx  = int(row["frame_idx"])
    frame_type = row["frame_type"]
    cbd        = _norm_cbd(row["cbd"])
    left_px    = int(row["left_px"])
    right_px   = int(row["right_px"])

    file_id = f"{TIFF_ID}_CBD{cbd}" if cbd else f"{TIFF_ID}_fr{frame_idx:02d}"
    if cbd and str(cbd) in _dup_cbds:
        file_id = f"{TIFF_ID}_CBD{cbd}_f{frame_idx}"
    fkey    = _cal_fkey(cbd, frame_idx)

    if frame_type == "partial":
        print(f"  {frame_idx:>3}  {'—':>6}  {'—':>6}  {'—':>7}  "
              f"{'—':>6}  {'—':>7}  {'—':>6}  {'—':>5}  (partial — skip)")
        continue

    # Check for explicit exclusion in guide picks
    if raw_picks.get(fkey, {}).get("exclude"):
        reason = raw_picks[fkey].get("exclude_reason", "excluded in guide picks")
        print(f"  {frame_idx:>3}  {str(cbd):>6}  {'—':>6}  {'—':>7}  "
              f"{'—':>6}  {'—':>7}  {'—':>6}  {'—':>5}  (EXCLUDED: {reason})")
        # Delete any stale figure left by a previous run before exclusion was added
        stale_png = PHASE3_DIR / f"F{FLT}_{file_id}_cal.png"
        if stale_png.exists():
            stale_png.unlink()
            print(f"        > Deleted stale figure: {stale_png.name}")
        # Write excluded row so step3 knows why this frame was skipped
        cal_rows.append(dict(
            flight        = FLT,
            tiff          = TIFF.name,
            tiff_id       = TIFF_ID,
            frame_idx     = frame_idx,
            cbd           = cbd or "",
            file_id       = file_id,
            mb_x          = "",
            mb_power_dB   = "",
            y_ref_px      = "",
            db_per_px     = "",
            us_per_px     = "",
            x_spacing_px  = "",
            hgrid_spacing = "",
            astra_x_ref_px = "",
            astra_x_grid_px = "",
            astra_y_grid_px = "",
            crop_start_y = "",
            crop_end_y = "",
            cal_pick_source = "",
            mb_detection_source = "",
            astra_pick_signature = "",
            cal_source_y  = "",
            had_guide     = "",
            exclude_reason = reason,
        ))
        continue

    # Crop this frame (numpy view — no extra memory)
    frame  = img_norm[:, left_px : right_px + 1]
    guides = _build_guides(fkey)

    # Per-frame calibration using the ASTRA-approved picks
    cal = detect_frame_calibration(frame, DEFAULT_CAL, guides=guides)

    mb_x         = cal["mb_x"]
    mb_power_dB  = cal["mb_power_dB"]
    y_ref_px     = cal["y_ref_px"]
    db_per_px    = cal["db_per_px"]
    x_spacing_px = cal["x_spacing_px"]
    hgrid_sp     = cal["hgrid_spacing"]
    hgrid_lines  = cal["hgrid_lines"]
    n_hgrid      = len(hgrid_lines)
    has_guide    = "[OK]" if guides.get("ref") is not None else "—"

    print(f"  {frame_idx:>3}  {str(cbd):>6}  {mb_x:>6}  {mb_power_dB:>+7.1f}  "
          f"{y_ref_px:>6.0f}  {db_per_px:>7.5f}  {x_spacing_px:>6.1f}  "
          f"{n_hgrid:>5}  {has_guide}")

    _frame_data[frame_idx] = dict(
        frame=frame, cal=cal, guides=guides,
        cbd=cbd, fkey=fkey, file_id=file_id,
    )
    cal_rows.append(dict(
        flight        = FLT,
        tiff          = TIFF.name,
        tiff_id       = TIFF_ID,
        frame_idx     = frame_idx,
        cbd           = cbd or "",
        file_id       = file_id,
        mb_x          = mb_x,
        mb_power_dB   = round(mb_power_dB, 2),
        y_ref_px      = round(y_ref_px, 1),
        db_per_px     = round(db_per_px, 6),
        us_per_px     = round(2.0 / x_spacing_px, 8) if x_spacing_px > 0 else round(DEFAULT_CAL["us_per_px"], 8),
        x_spacing_px  = round(x_spacing_px, 2),
        hgrid_spacing = round(hgrid_sp, 1) if hgrid_sp else "",
        astra_x_ref_px = guides.get("x_ref", ""),
        astra_x_grid_px = json.dumps(guides.get("x_grid", [])),
        astra_y_grid_px = json.dumps(guides.get("y_grid", [])),
        crop_start_y = guides["crop_start_y"],
        crop_end_y = guides["crop_end_y"],
        cal_pick_source = "ASTRA" if (
            guides.get("x_ref") is not None
            and guides.get("ref") is not None
            and len(guides.get("x_grid", [])) == ASTRA_X_POINTS
            and len(guides.get("y_grid", [])) == ASTRA_Y_POINTS
        ) else "automatic",
        mb_detection_source = "ASTRA montage",
        astra_pick_signature = astra_pick_signature(raw_picks[fkey]),
        cal_source_y  = cal["cal_source_y"],
        had_guide     = bool(guides.get("ref") is not None
                             and guides.get("x_ref") is not None
                             and guides.get("x_grid") and guides.get("y_grid")),
        exclude_reason = "",
    ))

# -- Quality gate: report main-bang outlier frames ----------------------------
# If mb_power_dB is more than 8 dB below the TIFF median, the main bang was
# likely mis-detected (T/R transient, noise peak, faint film).
if cal_rows:
    valid_mb = [r["mb_power_dB"] for r in cal_rows
                if isinstance(r["mb_power_dB"], (int, float)) and r["mb_power_dB"] > -50.0]
    if valid_mb:
        tiff_med_mb = float(np.median(valid_mb))
        bad_thresh  = tiff_med_mb - 8.0
        bad_idxs    = [(i, r) for i, r in enumerate(cal_rows)
                       if isinstance(r["mb_power_dB"], (int, float))
                       and r["mb_power_dB"] < bad_thresh]
        if bad_idxs:
            print(f"\n  Quality gate  : median mb={tiff_med_mb:.1f} dB  "
                  f"threshold={bad_thresh:.1f} dB")
            for _, r in bad_idxs:
                print(f"    WARNING: {r['file_id']}: mb_power {r['mb_power_dB']:+.1f} dB; "
                      f"keeping approved mb_x={r['mb_x']}.")

# -- Generate figures (after quality gate, using final calibrations) ------------
for frame_idx in sorted(_frame_data):
    d            = _frame_data[frame_idx]
    frame        = d["frame"]
    cal          = d["cal"]
    guides       = d["guides"]
    file_id      = d["file_id"]
    mb_x         = cal["mb_x"]
    y_ref_px     = cal["y_ref_px"]
    db_per_px    = cal["db_per_px"]
    x_spacing_px = cal["x_spacing_px"]
    hgrid_lines  = cal["hgrid_lines"]
    xgrid_lines  = cal.get("xgrid_lines", [])
    xgrid_raw    = cal.get("xgrid_raw_lines", [])

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("white")
    fw = frame.shape[1]
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
              extent=[0, fw, frame.shape[0], 0])
    if "astra_trace" in cal:
        ax.plot(np.arange(fw), cal["astra_trace"], color="red",
                linewidth=0.8, alpha=0.8, label="ASTRA trace")

    # Y grid lines (gold dashed) — ASTRA-approved positions
    db_labels = {}
    for yg in hgrid_lines:
        db_at_line = -60.0 + (y_ref_px - yg) * db_per_px
        ax.axhline(yg, color="gold", lw=0.8, ls="--", alpha=0.7)
        db_labels[yg] = db_at_line

    # Reference line (cyan thick) — y_ref_px = -60 dB
    ax.axhline(y_ref_px, color="cyan", lw=1.5, ls="--", alpha=0.9,
               label=f"Ref -60 dB  y={y_ref_px:.0f}")

    # X grid diagnostic: ASTRA-approved coordinates used by LYRA.
    _grid_label_done = False
    for xg in xgrid_lines:
        if 0 < xg < fw:
            ax.axvline(xg, color="steelblue", lw=0.8, ls="--", alpha=0.75,
                       label="ASTRA-approved grid"
                       if not _grid_label_done else None)
            _grid_label_done = True

    # Main bang (red thick) — approved ASTRA X reference.
    ax.axvline(mb_x, color="red", lw=1.5, ls="--", alpha=0.9,
               label=f"MB  t=0  x={mb_x}")

    # Show approved ASTRA reference and X-reference picks.
    if guides.get("ref") is not None:
        ax.axhline(guides["ref"], color="white", lw=0.8, ls=":",
                   alpha=0.7, label=f"guide_ref y={guides['ref']}")
    if guides.get("x_ref") is not None:
        ax.axvline(guides["x_ref"], color="purple", lw=0.8, ls=":",
                   alpha=0.7, label=f"ASTRA x_ref x={guides['x_ref']}")

    ax.set_ylim(DEFAULT_CAL["y_disp_hi"] + 50, DEFAULT_CAL["y_disp_lo"] - 50)
    ax.set_xlim(0, fw)
    ax.set_xlabel("Column (px, frame-relative)", fontsize=8)
    ax.set_ylabel("Row (px)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    src_lbl = "ASTRA picks" if guides.get("ref") is not None else "automatic"
    ax.set_title(
        f"LYRA Step 2 — F{FLT} {file_id}  |  "
        f"mb_x={mb_x}  y_ref={y_ref_px:.0f} (-60 dB)  "
        f"db/px={db_per_px:.5f}  x_sp={x_spacing_px:.1f} px  [{src_lbl}]",
        fontsize=8, loc="left",
    )
    ax.legend(fontsize=7, loc="upper right",
              framealpha=0.6, facecolor="black", labelcolor="white")

    # dB axis on the right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    yticks = [y for y in db_labels
              if DEFAULT_CAL["y_disp_lo"] - 50 < y < DEFAULT_CAL["y_disp_hi"] + 50]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{db_labels[y]:+.0f} dB" for y in yticks], fontsize=7)
    ax2.tick_params(right=True, labelright=True)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    fig_path = PHASE3_DIR / f"F{FLT}_{file_id}_cal.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# -- Update calibration CSV -----------------------------------------------------
if cal_rows:
    new_df = pd.DataFrame(cal_rows)
    if CAL_CSV.exists():
        existing = pd.read_csv(CAL_CSV, dtype=str)
        existing = existing[existing["tiff"] != TIFF.name]
        merged   = pd.concat([existing, new_df.astype(str)], ignore_index=True)
    else:
        merged = new_df.astype(str)
    merged.to_csv(CAL_CSV, index=False)
    print(f"\n  Calibration CSV -> {CAL_CSV}")
    print(f"  Phase 3 figures -> {PHASE3_DIR}")
    print(f"  Total cal rows  : {len(merged)}")
    n_guided = sum(1 for r in cal_rows if r["had_guide"])
    print(f"  Frames with guide: {n_guided}/{len(cal_rows)}")

# -- Generate calibration review JSON ------------------------------------------
# Reads the full flight cal CSV and flags CHECK TIFFs / frames.
# Updated after every TIFF run so the user always has a current overview.
REVIEW_JSON = PHASE3_DIR / f"F{FLT}_cal_review.json"

if CAL_CSV.exists():
    _review_df = pd.read_csv(CAL_CSV)
    _review_df.columns = _review_df.columns.str.strip()

    # Filter to calibrated (non-excluded) frames with valid x_spacing
    _excl = _review_df["exclude_reason"].fillna("").astype(str).str.strip()
    _cal_df = _review_df[_excl == ""].copy()
    _cal_df["x_spacing_px"] = pd.to_numeric(_cal_df["x_spacing_px"], errors="coerce")
    _cal_df["mb_power_dB"] = pd.to_numeric(_cal_df["mb_power_dB"], errors="coerce")
    _cal_df = _cal_df.dropna(subset=["x_spacing_px"])

    _review: dict = {
        "flight": FLT,
        "total_frames": len(_review_df),
        "calibrated_frames": len(_cal_df),
        "excluded_frames": int((_excl != "").sum()),
    }

    # Flight-wide x_spacing median (reference for outlier detection)
    _flt_xs_med = float(_cal_df["x_spacing_px"].median()) if len(_cal_df) > 0 else 205.0

    _check_tiffs: list[dict] = []
    _check_frames: list[dict] = []

    for _tid in sorted(_cal_df["tiff_id"].unique()):
        _sub = _cal_df[_cal_df["tiff_id"] == _tid]
        _xs_med = float(_sub["x_spacing_px"].median())
        _xs_std = float(_sub["x_spacing_px"].std()) if len(_sub) > 1 else 0.0

        _reason = ""
        if _xs_std > 10:
            _reason = f"x_spacing varies > 10 px (std={_xs_std:.1f})"
        elif abs(_xs_med - _flt_xs_med) > 15:
            _reason = (f"x_spacing median ({_xs_med:.1f}) deviates "
                       f"> 15 px from flight ({_flt_xs_med:.1f})")

        if _reason:
            # Flag individual frames with x_spacing outliers
            _bad = _sub[
                (_sub["x_spacing_px"] - _flt_xs_med).abs() > 15
            ]
            _bad_cbds = sorted(_bad["cbd"].dropna().astype(int).tolist())
            _bad_frames = []
            for _, _br in _bad.iterrows():
                _bad_frames.append({
                    "tiff_id": str(int(_br["tiff_id"])),
                    "cbd": str(int(_br["cbd"])) if pd.notna(_br["cbd"]) and str(_br["cbd"]).strip() else "",
                    "file_id": str(_br["file_id"]),
                    "x_spacing_px": round(float(_br["x_spacing_px"]), 1),
                })
            _check_tiffs.append({
                "tiff_id": str(int(_tid)),
                "reason": _reason,
                "x_spacing_median": round(_xs_med, 1),
                "x_spacing_std": round(_xs_std, 1),
                "n_frames": len(_sub),
                "n_bad": len(_bad),
                "bad_cbds": [f"{c:04d}" for c in _bad_cbds],
            })
            _check_frames.extend(_bad_frames)

    _n_good_tiffs = int(_cal_df["tiff_id"].nunique()) - len(_check_tiffs)
    _review["summary"] = {
        "good_tiffs": _n_good_tiffs,
        "check_tiffs": len(_check_tiffs),
        "total_tiffs": int(_cal_df["tiff_id"].nunique()),
        "flight_x_spacing_median": round(_flt_xs_med, 1),
    }
    _review["check_tiffs"] = _check_tiffs
    _review["check_frames"] = _check_frames

    with open(REVIEW_JSON, "w") as _f:
        json.dump(_review, _f, indent=2)
    print(f"  Review JSON   -> {REVIEW_JSON}")
    if _check_tiffs:
        print(f"    {len(_check_tiffs)} CHECK TIFFs, "
              f"{len(_check_frames)} flagged frames")
    else:
        print(f"    All {_n_good_tiffs} TIFFs GOOD")

print("\nDone.")
''',
    'detect_frames': r'''"""
detect_frames.py — LYRA Phase 1: Frame Detection + CBD Recognition
========================================================================
For any A-scope TIFF:
  1. Detects frame boundaries (complete vs partial)
  2. Assigns CBD numbers to each complete frame via one of three methods:
     - manual: reads from human-verified ASTRA CSV (ground truth)
     - segment: structural 7-segment OCR + sequential constraint (default)
     - ncc: NCC template matching + sequential constraint
  3. Updates the per-flight master frame index (F{FLT}_frame_index.csv)
  4. Outputs a contact sheet and (for OCR methods) a diagnostic figure

Usage
-----
Run from the project root, passing the TIFF path as an argument:

    python codes/detect_frames.py "path\\to\\F103-C1024_C1036.tiff"

Options:
    --method manual    Use human-verified CBDs from ASTRA CSV (ground truth)
    --method segment   Structural 7-segment OCR (default, 89% raw -> 100% corrected)
    --method ncc       NCC template matching (79% raw, requires digit_templates.npy)
    --override FR:CBD  Fix specific frames after OCR, e.g. --override 10:444 12:446
    --cbd-start N      Override the filename start and assign sequentially from N

Outputs
-------
Master index (per flight, updated incrementally):
    output/F{FLT}/phase1/F{FLT}_frame_index.csv
      Columns: flight, tiff, tiff_id, frame_idx, frame_type, cbd,
               left_px, right_px, width_px, ocr_method, ocr_raw

TIFF-level figures (named by tiff_id, never overwrite another TIFF's outputs):
    output/F{FLT}/phase1/F{FLT}_{tiff_id}_contact.png
    output/F{FLT}/phase1/ocr_diag/F{FLT}_{tiff_id}_ocr_diag.png  (OCR methods only)

Per-frame outputs in later steps use:
    CBD known   -> F{FLT}_{tiff_id}_CBD{cbd}_{desc}.ext
    Partial     -> F{FLT}_{tiff_id}_fr{frame_idx:02d}_{desc}.ext
"""

from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Figure typography -----------------------------------------------------------
# Use Segoe UI throughout all LYRA figures generated by this script.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from lyra import (
    detect_frames, display_path, ensure_canonical_name, output_root, parse_tiff_filename,
    resolve_tiff_arg, tiff_id,
)
from build_digit_templates import (
    recognize_frame, apply_sequential_constraint,
    get_binary_text, find_digit_col_ranges, adjust_blobs_to_7,
)

TMPL_PATH = Path(__file__).resolve().parent / "digit_templates.npy"

# -- Resolve TIFF path ----------------------------------------------------------
import argparse
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("tiff", nargs="?", default=None)
_parser.add_argument("--cbd-start", type=int, default=None,
                     help="Override OCR: assign CBDs sequentially starting from this value")
_parser.add_argument("--method", choices=["manual", "segment", "ncc", "ml", "ensemble"],
                     default="segment",
                     help="CBD assignment method: 'manual' (ASTRA CSV ground truth), "
                          "'segment' (structural 7-segment OCR, default), "
                          "'ncc' (NCC template matching), "
                          "'ml' (random forest classifier), or "
                          "'ensemble' (segment + ML fusion)")
_parser.add_argument("--override", nargs="*", default=None, metavar="FR:CBD",
                     help="Manually override specific frame CBDs, e.g. "
                          "--override 10:444 11:445 12:446. "
                          "Implies --method manual if ASTRA data exists.")
_parser.add_argument("--redetect", action="store_true", default=False,
                     help="Re-detect frame boundaries while preserving existing "
                          "CBD assignments from frame_index.csv. Matches new frames "
                          "to old frames by spatial overlap. Skips OCR entirely.")
_args, _ = _parser.parse_known_args()

# Parse --override into dict {frame_idx: 4-digit CBD string}
FRAME_OVERRIDES: dict[int, str] = {}
if _args.override:
    for token in _args.override:
        if ":" not in token:
            sys.exit(f"ERROR: --override expects FR:CBD format, got '{token}'")
        fr_str, cbd_str = token.split(":", 1)
        try:
            fr_idx = int(fr_str)
            cbd_val = int(cbd_str)
        except ValueError:
            sys.exit(f"ERROR: --override expects FR:CBD as integers, got '{token}'")
        FRAME_OVERRIDES[fr_idx] = f"{cbd_val:04d}"

if _args.tiff:
    TIFF = resolve_tiff_arg(_args.tiff, ROOT)
else:
    sys.exit("ERROR: TIFF path is required. Usage: python detect_frames.py <TIFF_PATH>")

# Rename to canonical format if needed (e.g. F141 files renamed from reel convention)
TIFF = ensure_canonical_name(TIFF)

# Infer flight number from the current filename, then the parent directory.
_metadata = parse_tiff_filename(TIFF)
CBD_START_OVERRIDE: int | None = (
    _args.cbd_start if _args.cbd_start is not None
    else (_metadata[1] if _metadata else None)
)
try:
    FLT = _metadata[0] if _metadata else int(TIFF.parent.name.lstrip("Ff"))
except ValueError:
    FLT = int(os.environ.get("LYRA_FLIGHT", "0"))
if FLT <= 0:
    sys.exit("ERROR: Cannot infer flight; use FLT/TIFF_ID (e.g. 127/4800) or tiff/127/.")

TIFF_ID   = tiff_id(TIFF)
OUT_DIR   = output_root(ROOT / "output") / f"F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PHASE1_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CSV = PHASE1_DIR / f"F{FLT}_frame_index.csv"

# -- Method-specific setup -----------------------------------------------------
REDETECT = _args.redetect
OCR_METHOD = _args.method
templates  = None
astra_cbds = None   # list of CBD ints for this TIFF (manual method only)

if REDETECT:
    # --redetect mode: skip all OCR setup, we'll map CBDs from existing CSV
    pass
elif OCR_METHOD == "manual":
    # Read human-verified CBDs from ASTRA combined picks CSV.
    # ASTRA 'filename' column uses stems without .tiff extension.
    ASTRA_CSV = ROOT / f"picks/{FLT}/{FLT}_CombinedASTRAPicks.csv"
    _tiff_stem = TIFF.stem
    _astra_ok = False
    if ASTRA_CSV.exists():
        _astra_df = pd.read_csv(ASTRA_CSV)
        _tiff_rows = _astra_df[_astra_df["filename"] == _tiff_stem]
        if len(_tiff_rows) > 0:
            astra_cbds = [int(x) for x in _tiff_rows["CBD"].dropna().tolist()]
            _astra_ok = True
    if not _astra_ok:
        if FRAME_OVERRIDES or CBD_START_OVERRIDE is not None:
            # User provided overrides — fall back to segment OCR for non-overridden frames
            print(f"  [info] No ASTRA picks for this TIFF; falling back to segment OCR "
                  f"(overrides will still apply)")
            OCR_METHOD = "segment"
            from segment_ocr import recognize_frame_structural, SEGMENT_THRESHOLD
        else:
            sys.exit(f"ERROR: TIFF '{_tiff_stem}' has no ASTRA picks.\n"
                     f"Use --method segment, --override FR:CBD, or --cbd-start N instead.")

elif OCR_METHOD == "ncc":
    if not TMPL_PATH.exists():
        sys.exit(f"ERROR: templates not found at {TMPL_PATH}\n"
                 "Run: python lyra.py --tool build_digit_templates")
    templates = np.load(TMPL_PATH, allow_pickle=True).item()

elif OCR_METHOD in ("segment", "ensemble"):
    from segment_ocr import recognize_frame_structural, SEGMENT_THRESHOLD

# -- Load image -----------------------------------------------------------------
print(f"\nLYRA Step 1 — {TIFF.name}")
_mode_str = "redetect (preserve CBDs)" if REDETECT else OCR_METHOD
print(f"  Flight  : F{FLT}  |  tiff_id : {TIFF_ID}  |  mode : {_mode_str}")
Image.MAX_IMAGE_PIXELS = None
img      = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
H, W     = img_norm.shape
print(f"  Image   : {W} × {H} px")

# -- Detect frames --------------------------------------------------------------
frames = detect_frames(img_norm)
n      = len(frames)
widths = [r - l for l, r in frames]
med_w  = float(np.median(widths))
print(f"  Frames  : {n} detected  (median width {int(med_w)} px)\n")

frame_types = ["complete" if (r - l) >= 0.75 * med_w else "partial"
               for l, r in frames]

# -- Post-hoc merge of adjacent partial frames ---------------------------------
# A mid-frame bright strip (splice, film damage) can split one physical A-scope
# into two adjacent "partial" frames.  Merge them when:
#   - both neighbours are classified "partial"
#   - their combined span (left_1 to right_2) is 0.50–1.50 × median width
# This is purely post-processing — does not change gap detection behaviour.
merged = True
while merged:
    merged = False
    for i in range(len(frames) - 1):
        if frame_types[i] == "partial" and frame_types[i + 1] == "partial":
            combined_w = frames[i + 1][1] - frames[i][0]
            if 0.50 * med_w <= combined_w <= 1.50 * med_w:
                new_frame = (frames[i][0], frames[i + 1][1])
                print(f"  [merge] fr{i} ({frames[i][1]-frames[i][0]} px) + "
                      f"fr{i+1} ({frames[i+1][1]-frames[i+1][0]} px) -> "
                      f"merged ({combined_w} px)")
                frames[i] = new_frame
                del frames[i + 1]
                # Reclassify the merged frame
                frame_types[i] = ("complete"
                                  if combined_w >= 0.75 * med_w else "partial")
                del frame_types[i + 1]
                n = len(frames)
                merged = True
                break   # restart scan after mutation

# -- Redetect mode: preserve CBDs from existing frame_index.csv -----------------
if REDETECT:
    if not INDEX_CSV.exists():
        sys.exit(f"ERROR: --redetect requires existing {INDEX_CSV.name} but none found.")
    old_df = pd.read_csv(INDEX_CSV, dtype=str)
    old_tiff = old_df[old_df["tiff"] == TIFF.name].copy()
    if len(old_tiff) == 0:
        sys.exit(f"ERROR: --redetect found no rows for {TIFF.name} in {INDEX_CSV.name}.")

    # Build old frame list: (left_px, right_px, cbd, ocr_method, ocr_raw)
    old_frames = []
    for _, row in old_tiff.iterrows():
        old_frames.append((
            int(row["left_px"]), int(row["right_px"]),
            row.get("cbd", ""), row.get("ocr_method", ""),
            row.get("ocr_raw", ""),
        ))

    # Match new frames to old frames by spatial overlap (IoU).
    # A new frame matches the old frame with highest overlap.
    cbd_by_frame     = {}
    ocr_raw_by_frame = {}
    method_used      = "redetect"
    n_matched = 0
    for ni, (nl, nr) in enumerate(frames):
        best_iou = 0.0
        best_old = None
        for oi, (ol, oright, ocbd, ometh, oraw) in enumerate(old_frames):
            overlap = max(0, min(nr, oright) - max(nl, ol))
            union   = max(nr, oright) - min(nl, ol)
            iou     = overlap / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_old = (ocbd, ometh, oraw)
        if best_old and best_iou > 0.3:
            ocbd, ometh, oraw = best_old
            cbd_by_frame[ni]     = ocbd if (ocbd and ocbd != "nan") else None
            ocr_raw_by_frame[ni] = f"(redetect:{oraw})" if oraw else "(redetect)"
            if ocbd:
                n_matched += 1
        else:
            cbd_by_frame[ni]     = None
            ocr_raw_by_frame[ni] = "(redetect:unmatched)"

    n_old_complete = sum(1 for _, _, c, _, _ in old_frames if c and c != "nan")
    print(f"  Redetect: {len(old_frames)} old frames -> {n} new frames")
    print(f"  CBD mapping: {n_matched}/{n_old_complete} CBDs preserved "
          f"(IoU > 0.3)")
    if n_matched < n_old_complete:
        # Identify lost CBDs
        old_cbds = {str(c) for _, _, c, _, _ in old_frames
                    if c and str(c) not in ("nan", "", "None")}
        new_cbds = {str(v) for v in cbd_by_frame.values()
                    if v and str(v) not in ("nan", "", "None")}
        lost = sorted(old_cbds - new_cbds)
        if lost:
            print(f"  Removed empty-gap frames (no CRT signal): CBD {', '.join(lost)}")

    # Set variables that downstream code expects
    complete_indices = [i for i, t in enumerate(frame_types) if t == "complete"]
    n_anchors = n_matched
    FRAME_OVERRIDES = {}  # no overrides in redetect mode

else:
    pass  # fall through to normal OCR path below

# -- Stage 1: CBD recognition on complete frames --------------------------------
if not REDETECT:
    complete_indices = [i for i, t in enumerate(frame_types) if t == "complete"]
    raw_reads_list   = []
    diag_blobs       = {}   # frame_idx -> blob list (for OCR diagnostic figure)
    diag_confs       = {}   # frame_idx -> per-digit Hamming distances
    method_used      = "override"   # updated below if OCR or manual runs

if not REDETECT and CBD_START_OVERRIDE is not None:
    # Last-resort override — assign CBDs sequentially from user-specified value.
    print(f"  cbd-start override: CBDs assigned {CBD_START_OVERRIDE} + sequential "
          f"(OCR skipped)")
    corrected_list = [
        f"{FLT:03d}{CBD_START_OVERRIDE + k:04d}"
        for k in range(len(complete_indices))
    ]
    raw_reads_list = ["(override)"] * len(complete_indices)
    n_anchors      = len(complete_indices)

elif not REDETECT and OCR_METHOD == "manual":
    # Use human-verified CBDs from ASTRA CSV — no OCR needed.
    method_used = "manual"
    n_astra    = len(astra_cbds)
    n_complete = len(complete_indices)
    if n_astra != n_complete:
        print(f"  WARNING: ASTRA CSV has {n_astra} CBDs but detected {n_complete} "
              f"complete frames — assigning min({n_astra},{n_complete}) in order")
    n_assign = min(n_astra, n_complete)
    corrected_list = [
        f"{FLT:03d}{astra_cbds[k]:04d}" for k in range(n_assign)
    ]
    # If more complete frames than ASTRA rows, pad with None placeholders
    corrected_list += [None] * (n_complete - n_assign)
    raw_reads_list  = [f"(ASTRA:{astra_cbds[k]})" if k < n_astra else "(no ASTRA)"
                       for k in range(n_complete)]
    n_anchors       = n_assign
    print(f"  ASTRA manual: {n_assign} CBDs assigned from "
          f"{FLT}_CombinedASTRAPicks.csv")

elif not REDETECT:
    # OCR path: segment, ml, ensemble, or NCC
    method_used = OCR_METHOD
    ml_model = None
    conf_list = []  # per-frame confidence (ensemble only)
    if OCR_METHOD in ("ml", "ensemble"):
        from ml_ocr import load_model
        ml_model = load_model()
        if OCR_METHOD == "ml":
            from ml_ocr import recognize_frame_ml
            print("  ML model loaded")
        else:
            from ml_ocr import ensemble_select_best, recognize_frame_ml
            print("  Ensemble model loaded (segment + ML)")

    if OCR_METHOD == "ensemble":
        # Run both methods on all frames, then let sequential constraint pick
        seg_reads = []
        ml_reads = []
        for i in complete_indices:
            l, r  = frames[i]
            crop  = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
            fw    = crop.shape[1]
            seg_r, blobs, confs = recognize_frame_structural(crop, H, fw)
            ml_r, _, _ = recognize_frame_ml(crop, H, fw, ml_model)
            seg_reads.append(seg_r)
            ml_reads.append(ml_r)
            diag_blobs[i] = blobs
            diag_confs[i] = confs
        # Pick whichever method produces more sequential anchors
        raw_reads_list, conf_list, ens_label = ensemble_select_best(
            complete_indices, seg_reads, ml_reads, FLT)
        method_used = ens_label
        print(f"  Ensemble winner: {ens_label}")
    else:
        for i in complete_indices:
            l, r  = frames[i]
            crop  = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
            fw    = crop.shape[1]
            if OCR_METHOD == "ml":
                recog, blobs, confs = recognize_frame_ml(crop, H, fw, ml_model)
                diag_blobs[i] = blobs
                diag_confs[i] = confs
            elif OCR_METHOD == "segment":
                recog, blobs, confs = recognize_frame_structural(crop, H, fw)
                diag_blobs[i] = blobs
                diag_confs[i] = confs
            else:
                recog, _ = recognize_frame(crop, H, fw, templates)
            raw_reads_list.append(recog)

    # -- Stage 2: Sequential constraint correction ------------------------------
    corrected_list, n_anchors = apply_sequential_constraint(
        complete_indices, raw_reads_list, flight=FLT,
        confidences=conf_list if conf_list else None)

# -- Build per-frame metadata (skip in redetect mode — already populated) ------
if not REDETECT:
    # cbd_by_frame: frame_idx -> 4-digit CBD string or None
    # ocr_raw_by_frame: frame_idx -> raw OCR string (before sequential correction)
    cbd_by_frame     = {}
    ocr_raw_by_frame = {}
    corr_iter = iter(corrected_list)
    raw_iter2 = iter(raw_reads_list)
    for i, ftype in enumerate(frame_types):
        if ftype == "complete":
            corr = next(corr_iter)
            raw  = next(raw_iter2)
            cbd_by_frame[i]     = corr[3:] if (corr and len(corr) >= 7) else None
            ocr_raw_by_frame[i] = str(raw) if raw else ""
        else:
            cbd_by_frame[i]     = None
            ocr_raw_by_frame[i] = ""

# -- Apply per-frame overrides with sequential propagation --------------------
# Overrides act as anchor points. Between anchors (and beyond them), CBDs are
# filled sequentially with step +1 per complete frame.
#   --override 3:434 10:444  ->  fr1=432 fr2=433 fr3=434 ... fr9=440 fr10=444 fr11=445 ...
if FRAME_OVERRIDES:
    # Validate and collect anchors as (position_in_complete_list, cbd_int)
    complete_order = [i for i in range(n) if frame_types[i] == "complete"]
    pos_of = {fi: pos for pos, fi in enumerate(complete_order)}
    anchors = []  # (pos, cbd_int, frame_idx)
    for fi in sorted(FRAME_OVERRIDES):
        cbd_4 = FRAME_OVERRIDES[fi]
        if fi < 0 or fi >= n:
            print(f"  WARNING: --override frame {fi} out of range (0–{n-1}), skipping")
            continue
        if frame_types[fi] != "complete":
            # Force-promote: if user explicitly overrides a partial frame,
            # they are asserting it is a real A-scope frame.
            print(f"  [override] promoting fr{fi} from partial -> complete")
            frame_types[fi] = "complete"
            # Rebuild complete_order and pos_of after promotion
            complete_order = [i for i in range(n) if frame_types[i] == "complete"]
            pos_of = {fj: pos for pos, fj in enumerate(complete_order)}
        anchors.append((pos_of[fi], int(cbd_4), fi))

    if anchors:
        anchors.sort()
        n_complete = len(complete_order)

        # NOTE: No backward propagation from first anchor.
        # CBDs can skip numbers (gaps in sequence), so backward-filling
        # with sequential values is unreliable. Users should provide
        # explicit anchors for earlier frames if needed.

        # Forward from each anchor to the next (or end)
        for ai in range(len(anchors)):
            a_pos, a_cbd, a_fi = anchors[ai]
            end_pos = anchors[ai + 1][0] if ai + 1 < len(anchors) else n_complete
            for p in range(a_pos, end_pos):
                fi = complete_order[p]
                new_cbd = f"{a_cbd + (p - a_pos):04d}"
                old = cbd_by_frame.get(fi)
                if fi == a_fi:
                    # This is the anchor itself
                    tag = f"(override:{new_cbd})"
                else:
                    tag = f"(propagated:{new_cbd})"
                if old != new_cbd:
                    cbd_by_frame[fi] = new_cbd
                    ocr_raw_by_frame[fi] = tag
                    FRAME_OVERRIDES[fi] = new_cbd
                elif fi == a_fi:
                    # Anchor frame — mark even if CBD didn't change
                    ocr_raw_by_frame[fi] = tag
                    FRAME_OVERRIDES[fi] = new_cbd

        # Print summary
        for fi in sorted(FRAME_OVERRIDES):
            if fi in pos_of:
                cbd_4 = cbd_by_frame[fi]
                src = "anchor" if fi in {a[2] for a in anchors} else "propagated"
                print(f"  Override ({src}): Fr {fi}  -> CBD {cbd_4}")
        print()

# -- Helper: frame identifier string (used for per-frame output filenames) ------
def frame_file_id(frame_idx: int) -> str:
    """
    Return the canonical file-name stem fragment identifying this frame.
    CBD-known frames: 'CBD{cbd}'   e.g. 'CBD0458'
    Partial / unknown: '{tiff_id}_fr{frame_idx:02d}'  e.g. '8400_fr00'
    """
    cbd = cbd_by_frame.get(frame_idx)
    if cbd:
        return f"CBD{cbd}"
    return f"{TIFF_ID}_fr{frame_idx:02d}"

# -- Print metadata table -------------------------------------------------------
raw_col = "Raw OCR" if method_used in ("segment", "ncc") else "Source"
print(f"  {'Fr':>3}  {'Type':>8}  {'CBD':>8}  {'FileID':>14}  {raw_col:>10}")
print("  " + "-" * 52)
for i in range(n):
    ptype    = frame_types[i]
    cbd      = cbd_by_frame[i]
    raw_disp = ocr_raw_by_frame.get(i, "—") if ptype == "complete" else "—"
    cbd_disp = cbd if cbd else "—"
    fid      = frame_file_id(i)
    print(f"  {i:>3}  {ptype:>8}  {cbd_disp:>8}  {fid:>14}  {raw_disp:>10}")

if REDETECT:
    print(f"\n  Redetect: {n_anchors} CBDs preserved from existing frame_index.csv")
elif CBD_START_OVERRIDE is not None:
    print(f"\n  CBD assignment: sequential override, {n_anchors}/{len(complete_indices)} frames assigned")
elif method_used == "manual":
    print(f"\n  CBD assignment: ASTRA manual, {n_anchors}/{len(complete_indices)} frames assigned")
else:
    print(f"\n  Sequential anchors: {n_anchors}/{len(complete_indices)} "
          f"raw reads consistent with best-fit sequence")

# -- Update master frame index CSV ---------------------------------------------
new_rows = []
for i in range(n):
    l, r  = frames[i]
    new_rows.append(dict(
        flight     = FLT,
        tiff       = TIFF.name,
        tiff_id    = TIFF_ID,
        frame_idx  = i,
        frame_type = frame_types[i],
        cbd        = cbd_by_frame[i] if cbd_by_frame[i] else "",
        left_px    = l,
        right_px   = r,
        width_px   = r - l,
        ocr_method = method_used + "+override" if i in FRAME_OVERRIDES else method_used,
        ocr_raw    = ocr_raw_by_frame[i],
    ))

new_df = pd.DataFrame(new_rows)

# Merge with existing index (replace rows for this TIFF if already present)
if INDEX_CSV.exists():
    existing = pd.read_csv(INDEX_CSV, dtype=str)
    existing = existing[existing["tiff"] != TIFF.name]  # drop old rows for this TIFF
    merged   = pd.concat([existing, new_df.astype(str)], ignore_index=True)
else:
    merged = new_df.astype(str)

# Sort by tiff_id then frame_idx for readability
merged["_sort_tiff"] = merged["tiff_id"].astype(str).str.zfill(8)
merged["_sort_fr"]   = merged["frame_idx"].astype(str).str.zfill(4)
merged = merged.sort_values(["_sort_tiff", "_sort_fr"]).drop(
    columns=["_sort_tiff", "_sort_fr"]).reset_index(drop=True)

merged.to_csv(INDEX_CSV, index=False)
print(f"\n  Frame index -> {display_path(INDEX_CSV, ROOT)}")
print(f"  Total rows in index: {len(merged)}")

# -- Output 1: Contact sheet (named by tiff_id) ---------------------------------
# Horizontal contact sheet: all detected frames are displayed left-to-right
# in a single row. Only visualization is changed; detection/OCR are untouched.
fig_width = max(16.0, n * 2.35)
fig, axes = plt.subplots(1, n, figsize=(fig_width, 3.9), squeeze=False)
fig.patch.set_facecolor("white")
axes_flat = axes.ravel()

for i, (l, r) in enumerate(frames):
    ax = axes_flat[i]
    crop = img_norm[:, l:r + 1]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=1, aspect="auto")

    ftype = frame_types[i]
    cbd = cbd_by_frame[i]
    if ftype == "complete" and cbd:
        label = f"Frame {i}\nCBD {cbd}"
    elif ftype == "complete":
        label = f"Frame {i}\nCBD —"
    else:
        label = f"Frame {i}\n(partial)"
    fc = "salmon" if ftype == "partial" else "white"
    ax.set_title(label, fontsize=14, fontweight="bold", color="black", pad=4,
                 bbox=dict(boxstyle="round,pad=0.2", fc=fc, alpha=0.75))
    ax.axis("off")

# Contact-sheet title uses the first and last assigned CBD in this TIFF.
_contact_cbds = [
    int(cbd_by_frame[i])
    for i in range(n)
    if cbd_by_frame.get(i)
]
if _contact_cbds:
    _contact_title = (
        f"LYRA Edge Detection: F{FLT} "
        f"CBD{min(_contact_cbds):04d} - CBD{max(_contact_cbds):04d}"
    )
else:
    _contact_title = f"LYRA Edge Detection: F{FLT}"

fig.suptitle(
    _contact_title,
    fontsize=14,
    fontweight="bold",
    y=0.99,
)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.80, wspace=0.035)
cs_path = PHASE1_DIR / f"F{FLT}_{TIFF_ID}_contact.png"

fig.savefig(cs_path, dpi=150, bbox_inches="tight", facecolor="white")

plt.close(fig)
print(f"\n  Contact sheet PNG -> {display_path(cs_path, ROOT)}")

# -- Output 2: OCR diagnostic figure (only for segment/ncc methods) -----------
if method_used in ("segment", "ncc") and len(complete_indices) > 0:
    import matplotlib.patches as mpatches

    n_complete = len(complete_indices)
    fig_d, axes_d = plt.subplots(n_complete, 3,
                                  figsize=(14, n_complete * 1.8),
                                  gridspec_kw={"width_ratios": [3, 3, 2]})
    fig_d.patch.set_facecolor("white")
    if n_complete == 1:
        axes_d = axes_d[np.newaxis, :]  # ensure 2D

    corr_iter_d = iter(corrected_list)
    raw_iter_d  = iter(raw_reads_list)
    for row, fi in enumerate(complete_indices):
        l, r  = frames[fi]
        crop  = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
        fw    = crop.shape[1]
        raw_s = str(next(raw_iter_d))
        cor_s = next(corr_iter_d)
        cor_cbd = cor_s[3:] if (cor_s and len(cor_s) >= 7) else "?"

        # -- Panel 1: Raw grayscale text crop ------------------------------
        ax1 = axes_d[row, 0]
        # Extract just the text band (top ~4.5% of frame height)
        text_h = max(int(H * 0.045), 40)
        text_crop = crop[:text_h, :]
        ax1.imshow(text_crop, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax1.set_ylabel(f"Frame {fi}", fontsize=14, rotation=0, labelpad=30, va="center")
        ax1.set_xticks([]); ax1.set_yticks([])

        # -- Panel 2: Binary image with blob boundaries --------------------
        ax2 = axes_d[row, 1]
        binary = get_binary_text(crop, H, fw)
        blobs  = diag_blobs.get(fi) if OCR_METHOD == "segment" else []
        if not blobs:
            # Recompute for NCC method (or if segment didn't store them)
            blobs = find_digit_col_ranges(binary)
            blobs = adjust_blobs_to_7(blobs)

        ax2.imshow(binary, cmap="gray", vmin=0, vmax=1, aspect="auto")
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(blobs), 1)))
        bh = binary.shape[0]
        for bi, (bs, be) in enumerate(blobs):
            c = colors[bi % len(colors)]
            rect = mpatches.Rectangle((bs, 0), be - bs, bh - 1,
                                       linewidth=1.2, edgecolor=c,
                                       facecolor="none")
            ax2.add_patch(rect)
        ax2.set_xticks([]); ax2.set_yticks([])

        # -- Panel 3: Text summary ----------------------------------------
        ax3 = axes_d[row, 2]
        ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
        ax3.axis("off")

        # Build per-digit comparison string with color coding
        confs = diag_confs.get(fi, [])
        raw_disp = raw_s if raw_s else "None"
        cor_disp = cor_cbd

        # Show raw -> corrected, highlighting mismatches
        lines = [f"Raw:  {raw_disp}", f"CBD:  {cor_disp}"]
        if confs:
            ham_str = " ".join(str(h) for h in confs)
            lines.append(f"Ham:  {ham_str}")

        # Color: green if raw matches corrected, red if corrected
        if raw_s and cor_s and raw_s == cor_s:
            txt_color = "#228B22"  # green — OCR was correct
        else:
            txt_color = "#CC3333"  # red — sequential constraint corrected

        ax3.text(0.05, 0.55, "\n".join(lines), fontsize=14,
                 fontfamily="monospace", va="center",
                 color=txt_color,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                           alpha=0.8))

    # Column headers
    for ci, title in enumerate(["Grayscale text crop", "Binary + blobs", "OCR result"]):
        axes_d[0, ci].set_title(title, fontsize=14, fontweight="bold")

    method_label = OCR_METHOD
    if OCR_METHOD == "segment":
        method_label = f"segment (theta={SEGMENT_THRESHOLD})"
    fig_d.suptitle(
        f"LYRA Phase 1 OCR Diagnostic — F{FLT}  tiff_id {TIFF_ID}\n"
        f"Method: {method_label}  |  Anchors: {n_anchors}/{n_complete}",
        fontsize=14, y=1.02,
    )
    fig_d.tight_layout()
    ocr_diag_dir = PHASE1_DIR / "ocr_diag"
    ocr_diag_dir.mkdir(exist_ok=True)
    diag_path = ocr_diag_dir / f"F{FLT}_{TIFF_ID}_ocr_diag.png"
    fig_d.savefig(diag_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_d)
    print(f"  OCR diagnostic -> {display_path(diag_path, ROOT)}")

print("\nDone.")
''',
    'echoes': r'''# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""
echoes.py — LYRA Phase 4: Surface/bed echo extraction
============================================================
For each calibrated frame (from phase3 cal CSV) extract surface and bed echoes:

  • Noise floor estimation (pre-bang 75th-percentile baseline)
  • Echo peak detection  (prominence >= 5 dB, distance >= 80 px)
  • Envelope walking at +5 dB and +10 dB above noise floor
  • Waveform shape metrics: peakiness, asymmetry, trailing_tail, leading_rise
  • Derived geometry:
      h_air_m  = surface_twt / 2 × c_air   (air column height)
      h_ice_m  = (bed_twt - surface_twt) / 2 × c_ice   (ice thickness)
      h_eff_m  = h_air_m + h_ice_m / n_ice   (effective propagation distance)

Echo status:
  good       — surface + bed detected, bed SNR >= WEAK_BED_SNR_DB (5 dB)
  weak_bed   — bed detected but SNR < WEAK_BED_SNR_DB (marginal; inspect figure)
  no_bed     — surface detected, no bed echo found
  no_surface — surface not detected; geometry undefined

Per-frame calibration is taken directly from the phase3 cal CSV (mb_x, y_ref_px,
db_per_px, us_per_px). The approved X-reference remains the main bang. Surface
and bed detection uses the ASTRA red trace within the existing robust search
region; their power is read at the selected column. Detection parameters
(prominence, distance, Gaussian sigma) are taken from DEFAULT_CAL.

Usage
-----
Normal mode (extract echoes):
    python codes/echoes.py [TIFF_PATH]

Review mode (interactive override -- click on image panel):
    python codes/echoes.py TIFF --review          # bad frames only
    python codes/echoes.py TIFF --review --all     # all frames

Flight-wide mode (all TIFFs for a flight):
    python codes/echoes.py --flight 125                    # normal mode, all TIFFs
    python codes/echoes.py --flight 125 --review           # review all TIFFs
    python codes/echoes.py --flight 125 --review --all     # review all frames in all TIFFs

    Keys: [s]=surface mode  [b]=bed mode  click=set position
          [S]=suppress surface  [B]=suppress bed  (mark as not present)
          [d]=clear all overrides  [n/Right]=next  [p/Left]=prev  [q]=save & quit

Pass a TIFF path, or use --flight for the flight-wide mode.

Outputs
-------
Per-flight echo CSV (updated incrementally):
    output/F{FLT}/phase4/F{FLT}_echoes.csv
      Columns: flight, tiff, cbd, file_id, echo_status,
               mb_x, astra_x_ref_px, astra_x_grid_px, astra_y_grid_px,
               cal_pick_source, mb_detection_source, y_ref_px, db_per_px, us_per_px,
               noise_floor_dB, mb_power_dB, mb_correction_dB,
               surface_twt_us, surface_power_dB, surface_snr_dB,
               surface_width_10_us, surface_width_5_us, surface_peakiness,
               surface_asymmetry, surface_leading_rise_us, surface_trailing_tail_us,
               bed_twt_us, bed_power_dB, bed_snr_dB,
               bed_width_10_us, bed_width_5_us, bed_peakiness,
               bed_asymmetry, bed_leading_rise_us, bed_trailing_tail_us,
               h_air_m, h_ice_m, h_eff_m
      Power/SNR values are gain-corrected using the main bang as a per-frame
      reference (mb_correction_dB = flight_median_mb - frame_mb_power).

Per-frame diagnostic figure (two-panel):
    output/F{FLT}/phase4/F{FLT}_{file_id}_echoes.png
      Left  : frame image with echo markers (pixel space, same scale as phase3)
      Right : waveform in physical units (TWT µs vs power dB)
"""

from pathlib import Path
import sys
import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd

# -- Parse arguments before matplotlib (backend depends on --review) -----------
_parser = argparse.ArgumentParser(
    description="LYRA Phase 4: Surface/bed echo extraction",
    epilog="Review mode: --review shows bad frames for manual echo override",
)
_parser.add_argument("tiff", nargs="?", help="TIFF path or FLT/TIFF_ID format")
_parser.add_argument("--review", action="store_true",
                     help="Interactive review: click on image to override surface/bed picks")
_parser.add_argument("--all", action="store_true",
                     help="With --review: show all frames (default: only no_bed/no_surface/weak_bed)")
_parser.add_argument("--flight", type=int, metavar="FLT",
                     help="Process all TIFFs for a flight (e.g. --flight 125). "
                          "Works with --review to iterate through every TIFF.")
_args = _parser.parse_args()

import matplotlib
if not _args.review:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Figure typography.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from lyra import output_root

# -- Flight-wide mode: loop through all TIFFs for a flight ---------------------
# When --flight N is given, find all TIFFs and re-invoke echoes.py on each one.
# Works with --review, --all, and normal mode.
if _args.flight is not None:
    import subprocess
    flt = _args.flight
    cal_csv = output_root(ROOT / "output") / f"F{flt}/phase3/F{flt}_cal.csv"
    if not cal_csv.exists():
        sys.exit(f"ERROR: No cal CSV at {cal_csv}")
    cal_df = pd.read_csv(cal_csv, dtype=str)
    tiff_ids = sorted(cal_df["tiff_id"].unique(), key=int)
    print(f"Flight F{flt}: {len(tiff_ids)} TIFFs to process\n")
    script = str(Path(__file__).resolve())
    for i, tid in enumerate(tiff_ids, 1):
        cmd = [sys.executable, script, f"{flt}/{tid}"]
        if _args.review:
            cmd.append("--review")
        if _args.all:
            cmd.append("--all")
        print(f"{'='*60}")
        print(f"  TIFF {tid}  ({i}/{len(tiff_ids)})")
        print(f"{'='*60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: TIFF {tid} exited with code {result.returncode}")
        print()
    print(f"Flight F{flt}: all {len(tiff_ids)} TIFFs done.")
    sys.exit(0)

from lyra import (
    ASTRA_DEFAULT_CROP_END_Y, ASTRA_DEFAULT_CROP_START_Y,
    ASTRA_X_POINTS, ASTRA_Y_POINTS, astra_pick_signature, astra_peak_row, DEFAULT_CAL,
    extract_astra_science_trace, estimate_noise_floor,
    detect_echoes, compute_echo_metrics,
    extract_local_profile,
    px_to_db, px_to_us, db_to_px, us_to_px,
    C_AIR_M_PER_US, C_ICE_M_PER_US,
    ensure_canonical_name, output_root, parse_tiff_filename,
    resolve_tiff_arg, tiff_id,
)
from run_flight import _calibration_key

# -- Scientific constants -------------------------------------------------------
N_ICE           = 1.78   # refractive index of ice (Neal 1977)
WEAK_BED_SNR_DB = 5.0    # bed SNR threshold below which status = "weak_bed"

# -- Resolve TIFF path ----------------------------------------------------------
if _args.tiff:
    TIFF = resolve_tiff_arg(_args.tiff, ROOT)
else:
    sys.exit("ERROR: TIFF path or --flight is required.")

TIFF = ensure_canonical_name(TIFF)

_metadata = parse_tiff_filename(TIFF)
try:
    FLT = _metadata[0] if _metadata else int(TIFF.parent.name.lstrip("Ff"))
except ValueError:
    FLT = int(os.environ.get("LYRA_FLIGHT", "0"))
if FLT <= 0:
    sys.exit("ERROR: Cannot infer flight; use FLT/TIFF_ID (e.g. 127/4800) or tiff/127/.")


TIFF_ID   = tiff_id(TIFF)
OUT_DIR   = output_root(ROOT / "output") / f"F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PHASE3_DIR = OUT_DIR / "phase3"
PHASE4_DIR = OUT_DIR / "phase4"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV  = PHASE1_DIR / f"F{FLT}_frame_index.csv"
CAL_CSV    = PHASE3_DIR / f"F{FLT}_cal.csv"
ECHO_CSV   = PHASE4_DIR / f"F{FLT}_echoes.csv"

# -- Validate prerequisites -----------------------------------------------------
if not INDEX_CSV.exists():
    sys.exit(f"ERROR: frame index not found at {INDEX_CSV}\n"
             "Run phase 1 (detect_frames.py) first.")

if not CAL_CSV.exists():
    sys.exit(f"ERROR: phase 3 calibration CSV not found at {CAL_CSV}\n"
             "Run phase 3 (calibrate.py) first.")

# -- Load phase1 frame index ---------------------------------------------------
index     = pd.read_csv(INDEX_CSV, dtype=str)
tiff_rows = index[index["tiff"] == TIFF.name].copy()

if len(tiff_rows) == 0:
    sys.exit(f"ERROR: {TIFF.name} not found in frame index.\n"
             "Run phase 1 (detect_frames.py) for this TIFF first.")

# -- Load phase3 calibration CSV ------------------------------------------------
cal_df    = pd.read_csv(CAL_CSV, dtype=str)
tiff_cals = cal_df[cal_df["tiff"] == TIFF.name].copy()

if len(tiff_cals) == 0:
    sys.exit(f"ERROR: No phase 3 calibration rows for {TIFF.name} in {CAL_CSV}\n"
             "Run phase 3 (calibrate.py) for this TIFF first.")
if "cal_pick_source" not in tiff_cals.columns:
    sys.exit(
        f"ERROR: {TIFF.name} has legacy Phase 3 calibration without ASTRA picks.\n"
        "Re-run calibration from phase 3 before Phase 4."
    )
_active_cals = tiff_cals[
    tiff_cals["exclude_reason"].fillna("").astype(str).str.strip().isin(["", "nan"])
]
if not _active_cals["cal_pick_source"].fillna("").astype(str).eq("ASTRA").all():
    sys.exit(
        f"ERROR: {TIFF.name} has calibration rows not based on approved ASTRA picks.\n"
        "Re-run ASTRA and Phase 3 before Phase 4."
    )
if "mb_detection_source" not in tiff_cals.columns:
    sys.exit(
        f"ERROR: {TIFF.name} has legacy calibration without main-bang "
        "detection provenance. Re-run Phase 3 before Phase 4."
    )
if not _active_cals["mb_detection_source"].fillna("").astype(str).eq("ASTRA montage").all():
    sys.exit(
        f"ERROR: {TIFF.name} has calibration rows without an approved ASTRA main bang. "
        "Re-run ASTRA and Phase 3 before Phase 4."
    )
if "astra_pick_signature" not in tiff_cals.columns:
    sys.exit("ERROR: Phase 3 calibration predates approved montage picks. Re-run Phase 2 and Phase 3.")
_picks_path = PHASE1_DIR / f"F{FLT}_cal_picks.json"
if not _picks_path.exists():
    sys.exit(f"ERROR: Approved ASTRA picks are missing at {_picks_path}.")
with _picks_path.open(encoding="utf-8") as _handle:
    _current_picks = json.load(_handle)
_complete_rows = tiff_rows[tiff_rows["frame_type"] == "complete"]
_counts = pd.to_numeric(_complete_rows["cbd"], errors="coerce").dropna().astype(int).value_counts()
_duplicates = {str(cbd) for cbd, count in _counts.items() if count > 1}
for _, _cal_row in _active_cals.iterrows():
    _index_rows = _complete_rows[_complete_rows["frame_idx"] == _cal_row["frame_idx"]]
    if len(_index_rows) != 1:
        sys.exit(f"ERROR: Missing or ambiguous frame index for {TIFF.name} frame {_cal_row['frame_idx']}.")
    _key = _calibration_key(_index_rows.iloc[0], _duplicates)
    _pick = _current_picks.get(_key, {})
    _x_grid = _pick.get("x_grid")
    _y_grid = _pick.get("y_grid")
    if (_pick.get("montage_approved") is not True
            or not isinstance(_x_grid, list) or len(_x_grid) != ASTRA_X_POINTS
            or not isinstance(_y_grid, list) or len(_y_grid) != ASTRA_Y_POINTS
            or astra_pick_signature(_pick) != _cal_row["astra_pick_signature"]
            or pd.isna(_cal_row["mb_x"])
            or float(_cal_row["mb_x"]) != float(_pick.get("x_ref", -1))):
        sys.exit(f"ERROR: {TIFF.name} {_key} has stale or unapproved calibration. "
                 "Re-run ASTRA and Phase 3 before Phase 4.")

# -- Flight-wide mb_power reference for per-frame gain correction --------------
# The main bang is a fixed internal signal (T/R switch leakage) that traverses
# the full receiver chain but never leaves the aircraft.  Variation in mb_power
# across frames reflects system gain changes (attenuator setting, CRT intensity)
# rather than geophysical signal.  We use the flight-wide median as the expected
# MB power and correct each frame's echo powers accordingly.
_all_cals = cal_df.copy()
_all_cals["mb_power_dB"] = pd.to_numeric(_all_cals["mb_power_dB"], errors="coerce")
_excl_mask = _all_cals["exclude_reason"].fillna("").astype(str).str.strip().isin(["", "nan"])
_valid_mb  = _all_cals.loc[_excl_mask, "mb_power_dB"].dropna()
FLIGHT_MB_REF_DB = float(_valid_mb.median()) if len(_valid_mb) > 0 else -37.0

def _read_pick_positions(value, column: str) -> list[float]:
    if value is None or pd.isna(value) or str(value).strip() in ("", "nan"):
        return []
    positions = json.loads(str(value))
    if not isinstance(positions, list):
        raise ValueError(f"{column} in {CAL_CSV} must contain a JSON list")
    return [float(position) for position in positions]


def _read_optional_pick(value) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() in ("", "nan"):
        return None
    return float(value)


def _read_crop_bounds(row) -> tuple[int, int]:
    start = row.get("crop_start_y")
    end = row.get("crop_end_y")
    start_missing = start is None or pd.isna(start) or str(start).strip() in ("", "nan")
    end_missing = end is None or pd.isna(end) or str(end).strip() in ("", "nan")
    if start_missing and end_missing:
        return ASTRA_DEFAULT_CROP_START_Y, ASTRA_DEFAULT_CROP_END_Y
    if start_missing or end_missing:
        raise ValueError("Phase 3 calibration must contain both ASTRA crop bounds")
    start_value = float(start)
    end_value = float(end)
    if (not start_value.is_integer() or not end_value.is_integer()
            or start_value < 0 or end_value <= start_value):
        raise ValueError(f"Invalid ASTRA crop bounds: {start}, {end}")
    return int(start_value), int(end_value)


# Build frame_idx -> calibration dict mapping (skip excluded frames)
frame_cal: dict[int, dict] = {}
for _, row in tiff_cals.iterrows():
    # Skip excluded frames (empty mb_x means no calibration was performed)
    if row.get("exclude_reason", "") not in ("", "nan") and str(row.get("exclude_reason", "")) not in ("", "nan"):
        print(f"  Skipping frame {row['frame_idx']} ({row.get('cbd','?')}): {row['exclude_reason']}")
        continue
    if not row["mb_x"] or str(row["mb_x"]).strip() == "":
        continue
    fidx = int(row["frame_idx"])
    _mb_pwr = float(row["mb_power_dB"]) if row.get("mb_power_dB") else FLIGHT_MB_REF_DB
    _crop_start_y, _crop_end_y = _read_crop_bounds(row)
    frame_cal[fidx] = dict(
        mb_x        = int(float(row["mb_x"])),
        y_ref_px    = float(row["y_ref_px"]),
        db_per_px   = float(row["db_per_px"]),
        us_per_px   = float(row["us_per_px"]),
        x_ref_px    = _read_optional_pick(row.get("astra_x_ref_px")),
        x_grid_px   = _read_pick_positions(row.get("astra_x_grid_px"), "astra_x_grid_px"),
        y_grid_px   = _read_pick_positions(row.get("astra_y_grid_px"), "astra_y_grid_px"),
        crop_start_y = _crop_start_y,
        crop_end_y   = _crop_end_y,
        cal_pick_source = row.get("cal_pick_source", "automatic"),
        mb_detection_source = row.get("mb_detection_source", "automatic"),
        cbd         = row["cbd"],
        file_id     = row["file_id"],
        tiff_id     = row.get("tiff_id", TIFF_ID),
        mb_power_dB = _mb_pwr,
        mb_correction_dB = round(FLIGHT_MB_REF_DB - _mb_pwr, 2),
    )

print(f"\nLYRA Step 3 — {TIFF.name}")
print(f"  Flight : F{FLT}  |  tiff_id : {TIFF_ID}")
print(f"  Calibrated frames in step2 CSV : {len(frame_cal)}")
# Show mb_power correction info
_this_mb = [c["mb_power_dB"] for c in frame_cal.values()]
_this_corr = [c["mb_correction_dB"] for c in frame_cal.values()]
_max_abs_corr = max(abs(c) for c in _this_corr) if _this_corr else 0
print(f"  MB power ref  : {FLIGHT_MB_REF_DB:.1f} dB (flight median)")
if _max_abs_corr > 0.5:
    print(f"  MB correction : this TIFF range [{min(_this_corr):+.1f}, {max(_this_corr):+.1f}] dB")
else:
    print(f"  MB correction : < 0.5 dB (negligible)")
print()


# -- Load TIFF image ------------------------------------------------------------
print("Loading TIFF ...")
Image.MAX_IMAGE_PIXELS = None
img      = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
H, W_img = img_norm.shape
print(f"  Image  : {W_img} × {H} px\n")

# Contact-strip thumbnails are prepared once per TIFF and reused in the
# publication figure for every complete frame. They are visual only.
# Keep the native image proportions by resizing with thumbnail(), not resize().
pub_frame_thumbnails = []
for _, _strip_row in tiff_rows.sort_values(
        "frame_idx", key=lambda col: pd.to_numeric(col, errors="coerce")
).iterrows():
    _a = int(_strip_row["left_px"])
    _b = int(_strip_row["right_px"]) + 1
    _frame_img = np.asarray(np.clip(img_norm[:, _a:_b] * 255, 0, 255), dtype=np.uint8)
    _thumb = Image.fromarray(_frame_img)
    _thumb.thumbnail((240, 235), Image.Resampling.LANCZOS)
    pub_frame_thumbnails.append((
        int(_strip_row["frame_idx"]),
        str(_strip_row["cbd"]) if pd.notna(_strip_row["cbd"]) else "",
        str(_strip_row["frame_type"]).strip().lower() == "partial",
        np.asarray(_thumb),
    ))

# -- Build per-frame cal dict from step2 values + DEFAULT_CAL detection params -
def _make_cal(fc: dict) -> dict:
    """Merge per-frame step2 calibration values into DEFAULT_CAL."""
    cal = DEFAULT_CAL.copy()
    cal["mb_x"]     = fc["mb_x"]     # already in frame, but passed separately
    cal["y_ref_px"] = fc["y_ref_px"]
    cal["db_per_px"]= fc["db_per_px"]
    cal["us_per_px"]= fc["us_per_px"]
    # y_ref_db stays at -60 dB (reference is always picked at -60 dB line)
    return cal


def _draw_astra_pixel_grid(ax, fc: dict):
    for y in fc["y_grid_px"]:
        ax.axhline(y, color="limegreen", lw=0.6, ls=":", alpha=0.65)
    for x in fc["x_grid_px"]:
        ax.axvline(x, color="orange", lw=0.6, ls=":", alpha=0.65)
    if fc["x_ref_px"] is not None:
        ax.axvline(fc["x_ref_px"], color="purple", lw=0.8, ls="-.", alpha=0.75)


# -- Echo overrides ------------------------------------------------------------
OVERRIDES_JSON = PHASE4_DIR / f"F{FLT}_echo_overrides.json"


def load_echo_overrides() -> dict:
    """Load manual echo overrides from JSON. Keys are file_id (e.g. 'CBD0465')."""
    if OVERRIDES_JSON.exists():
        with open(OVERRIDES_JSON) as f:
            return json.load(f)
    return {}


def save_echo_overrides(overrides: dict) -> None:
    """Write echo overrides to JSON."""
    with open(OVERRIDES_JSON, "w") as f:
        json.dump(overrides, f, indent=2)
    print(f"  Overrides -> {OVERRIDES_JSON}  ({len(overrides)} frames)")


echo_overrides = load_echo_overrides()
if echo_overrides:
    print(f"  Echo overrides loaded: {len(echo_overrides)} frames  ({OVERRIDES_JSON.name})")


# ==============================================================================
# Review mode — interactive echo override picker
# ==============================================================================

def review_mode():
    """Interactive review: click on image panel to override surface/bed echo positions.

    Shows ALL calibrated frames in the TIFF. Users press [s] for surface mode,
    [b] for bed mode, then click on the left (image) panel to place the pick
    directly on the CRT trace overlay.
    """

    # Need existing echo CSV to know current statuses
    if not ECHO_CSV.exists():
        sys.exit("ERROR: No echo CSV found. Run normal mode first, then --review.")

    echo_df = pd.read_csv(ECHO_CSV)
    tiff_echos = echo_df[echo_df["tiff"] == TIFF.name].copy()

    if len(tiff_echos) == 0:
        sys.exit(f"ERROR: No echo rows for {TIFF.name}. Run normal mode first.")

    # Build frame queue — default: only bad frames + frames with overrides
    # Use --all to include good frames too
    BAD_STATUSES = {"no_bed", "no_surface", "weak_bed"}
    show_all = getattr(_args, "all", False)

    review_frames = []
    all_status_counts = {}

    for _, idx_row in tiff_rows.iterrows():
        frame_idx = int(idx_row["frame_idx"])
        if idx_row["frame_type"] == "partial":
            continue
        if frame_idx not in frame_cal:
            continue

        fc = frame_cal[frame_idx]
        fkey = fc["file_id"]

        erow = tiff_echos[tiff_echos["file_id"] == fkey]
        current_status = erow.iloc[0]["echo_status"] if len(erow) > 0 else "unknown"
        all_status_counts[current_status] = all_status_counts.get(current_status, 0) + 1

        if show_all or current_status in BAD_STATUSES:
            review_frames.append((frame_idx, fc, current_status, idx_row))

    # Print TIFF summary (all frames, not just filtered)
    total = sum(all_status_counts.values())
    print(f"\nTIFF {TIFF_ID}: {total} frames total")
    for s_name in ["good", "weak_bed", "no_bed", "no_surface"]:
        if s_name in all_status_counts:
            print(f"    {s_name:12s}: {all_status_counts[s_name]}")

    if not review_frames:
        print("No frames to review (all good). Use --all to review every frame.")
        return

    filter_label = "all frames" if show_all else "bad + overridden frames only"
    print(f"\nReview: {len(review_frames)} frames ({filter_label})")
    print(f"  Click on LEFT panel (image) to place pick on CRT trace")
    print(f"  Keys: [s]=surface  [b]=bed  click=set position")
    print(f"        [S]=suppress surface  [B]=suppress bed  (mark as not present)")
    print(f"        [d]=clear overrides  [n/Right]=next  [p/Left]=prev  [q]=save & quit\n")

    frame_i = 0
    while 0 <= frame_i < len(review_frames):
        frame_idx, fc, current_status, idx_row = review_frames[frame_i]
        fkey = fc["file_id"]
        cbd = fc["cbd"]
        mb_x = fc["mb_x"]

        left_px = int(idx_row["left_px"])
        right_px = int(idx_row["right_px"])
        frame = img_norm[:, left_px:right_px + 1]
        frame_w = frame.shape[1]
        cal = _make_cal(fc)

        # Extract trace (fast, pure numpy)
        trace_y, trace_y_s, _ = extract_astra_science_trace(
            frame, cal, mb_x, f"{TIFF.name} {fkey} Phase 4 review",
            fc["crop_start_y"], fc["crop_end_y"])
        noise_floor_dB = estimate_noise_floor(trace_y_s, mb_x, cal)

        # Algorithmic detection (for reference)
        algo_surface_x, algo_bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

        # Current override (mutable copy)
        override = dict(echo_overrides.get(fkey, {}))

        # Mutable state for event handlers
        st = {"mode": "b", "override": override, "action": None, "drawn": []}

        us_per_px = cal["us_per_px"]

        # -- Build interactive figure ------------------------------------------
        fig = plt.figure(figsize=(18, 6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, wspace=0.08)
        ax_img = fig.add_subplot(gs[0, :2])
        ax_wave = fig.add_subplot(gs[0, 2])

        # -- Left panel: frame image (click target) ---------------------------
        ax_img.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      extent=[0, frame_w, frame.shape[0], 0])
        ax_img.axhline(cal["y_ref_px"], color="cyan", lw=1.2, ls="--", alpha=0.8)
        ax_img.axvline(mb_x, color="red", lw=1.5, ls="--", alpha=0.9)
        _draw_astra_pixel_grid(ax_img, fc)
        x_all = np.arange(frame_w)
        ax_img.plot(x_all, trace_y, color="red", lw=0.8, alpha=0.7, label="ASTRA trace")

        # Algo markers (faded reference)
        if algo_surface_x is not None:
            algo_srf_y = trace_y_s[algo_surface_x] if algo_surface_x < len(trace_y_s) else 0
            ax_img.plot(algo_surface_x, algo_srf_y, marker="^", color="lime",
                        markersize=7, alpha=0.35, zorder=4, label="algo srf")
            ax_img.axvline(algo_surface_x, color="lime", lw=0.5, ls=":", alpha=0.35)
        if algo_bed_x is not None:
            algo_bed_y = trace_y_s[algo_bed_x] if algo_bed_x < len(trace_y_s) else 0
            ax_img.plot(algo_bed_x, algo_bed_y, marker="^", color="orange",
                        markersize=7, alpha=0.35, zorder=4, label="algo bed")
            ax_img.axvline(algo_bed_x, color="orange", lw=0.5, ls=":", alpha=0.35)

        ax_img.set_ylim(DEFAULT_CAL["y_disp_hi"] + 50, DEFAULT_CAL["y_disp_lo"] - 50)
        ax_img.set_xlim(0, frame_w)
        ax_img.set_xlabel("Column (px) — click here to place pick", fontsize=8)
        ax_img.set_ylabel("Row (px)", fontsize=8)
        ax_img.spines["top"].set_visible(False)
        ax_img.spines["right"].set_visible(False)
        ax_img.legend(fontsize=7, loc="upper right", framealpha=0.6,
                      facecolor="black", labelcolor="white")

        # -- Right panel: waveform (reference only) ----------------------------
        twt_arr = px_to_us(np.arange(frame_w, dtype=float), mb_x, cal)
        power_arr = px_to_db(trace_y_s, cal)
        t_lo = float(cal["mb_skip_us"])
        t_hi = 25.0
        wf_mask = (twt_arr >= t_lo) & (twt_arr <= t_hi)

        ax_wave.plot(twt_arr[wf_mask], power_arr[wf_mask], color="black", lw=1.2)
        ax_wave.axhline(noise_floor_dB, color="gray", lw=1.0, ls="--", alpha=0.8,
                        label=f"NF {noise_floor_dB:.1f} dB")
        ax_wave.axhline(noise_floor_dB + 5.0, color="silver", lw=0.7, ls=":", alpha=0.8)
        ax_wave.axhline(noise_floor_dB + 10.0, color="darkgray", lw=0.7, ls=":", alpha=0.8)
        for x in fc["x_grid_px"]:
            ax_wave.axvline((x - mb_x) * us_per_px, color="orange",
                            lw=0.5, ls=":", alpha=0.45)
        for y in fc["y_grid_px"]:
            ax_wave.axhline(-60.0 + (cal["y_ref_px"] - y) * cal["db_per_px"],
                            color="limegreen", lw=0.5, ls=":", alpha=0.45)

        # Algo surface/bed on waveform (faded)
        if algo_surface_x is not None:
            algo_srf_twt = px_to_us(float(algo_surface_x), mb_x, cal)
            algo_srf_pwr = px_to_db(trace_y_s[algo_surface_x], cal)
            ax_wave.plot(algo_srf_twt, algo_srf_pwr, marker="^", color="limegreen",
                         markersize=7, alpha=0.35, zorder=4)
            ax_wave.axvline(algo_srf_twt, color="limegreen", lw=0.5, ls=":", alpha=0.35)
        if algo_bed_x is not None:
            algo_bed_twt = px_to_us(float(algo_bed_x), mb_x, cal)
            algo_bed_pwr = px_to_db(trace_y_s[algo_bed_x], cal)
            ax_wave.plot(algo_bed_twt, algo_bed_pwr, marker="^", color="darkorange",
                         markersize=7, alpha=0.35, zorder=4)
            ax_wave.axvline(algo_bed_twt, color="darkorange", lw=0.5, ls=":", alpha=0.35)

        p_floor = noise_floor_dB - 5.0
        p_ceil = noise_floor_dB + 50.0
        ax_wave.set_ylim(p_floor, p_ceil)
        ax_wave.set_xlim(t_lo, t_hi)
        ax_wave.set_xlabel("TWT from MB (µs)", fontsize=8)
        ax_wave.set_ylabel("Power (dB)", fontsize=8)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.tick_params(labelsize=7)
        ax_wave.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

        # -- Title + status bar ------------------------------------------------
        mode_label = "BED" if st["mode"] == "b" else "SURFACE"
        status_color = {"good": "green", "weak_bed": "goldenrod",
                        "no_bed": "tomato", "no_surface": "red"}.get(current_status, "gray")
        title_obj = fig.suptitle(
            f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
            f"({frame_i + 1}/{len(review_frames)})  "
            f"mode=[{mode_label}]",
            fontsize=10, color=status_color,
        )

        status_bar = fig.text(
            0.5, 0.005, "", ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.85),
        )

        def _update_status():
            m = st["mode"]
            ov = st["override"]
            parts = [f"Mode: {'SURFACE' if m == 's' else 'BED'}"]
            if ov.get("surface_suppress"):
                parts.append("Srf: SUPPRESSED")
            elif "surface_twt_us" in ov:
                twt = ((ov["surface_x_px"] - mb_x) * us_per_px
                       if "surface_x_px" in ov else ov["surface_twt_us"])
                parts.append(f"Srf: {twt:.2f} us")
            if ov.get("bed_suppress"):
                parts.append("Bed: SUPPRESSED")
            elif "bed_twt_us" in ov:
                twt = ((ov["bed_x_px"] - mb_x) * us_per_px
                       if "bed_x_px" in ov else ov["bed_twt_us"])
                parts.append(f"Bed: {twt:.2f} us")
            if not any(k in ov for k in ("surface_twt_us", "bed_twt_us",
                                          "surface_suppress", "bed_suppress")):
                parts.append("No overrides")
            status_bar.set_text("  |  ".join(parts))
            col = "magenta" if m == "s" else "deepskyblue"
            status_bar.set_color(col)

        def _redraw_overrides():
            for art in st["drawn"]:
                try:
                    art.remove()
                except Exception:
                    pass
            st["drawn"].clear()

            ov = st["override"]

            # Surface suppress: show X at algo surface position
            if ov.get("surface_suppress"):
                if algo_surface_x is not None and 0 <= algo_surface_x < frame_w:
                    y_px = trace_y_s[algo_surface_x]
                    st["drawn"].append(
                        ax_img.plot(algo_surface_x, y_px, marker="X", color="magenta",
                                    markersize=12, zorder=6, alpha=0.8,
                                    markeredgecolor="white", markeredgewidth=1.0)[0])
                # Label in top-left corner of image panel
                st["drawn"].append(
                    ax_img.text(0.02, 0.96, "SRF SUPPRESSED", transform=ax_img.transAxes,
                                fontsize=9, color="magenta", fontweight="bold",
                                va="top", ha="left",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="magenta", alpha=0.85)))
            elif "surface_twt_us" in ov:
                x_px = ov.get("surface_x_px", mb_x + int(round(ov["surface_twt_us"] / us_per_px)))
                twt = (x_px - mb_x) * us_per_px
                if 0 <= x_px < frame_w:
                    y_px = trace_y[x_px]
                    pwr = float(px_to_db(y_px, cal))
                    # Diamond on image panel (at clicked position)
                    st["drawn"].append(
                        ax_img.plot(x_px, y_px, marker="D", color="magenta",
                                    markersize=7, zorder=6, alpha=0.6,
                                    markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_img.axvline(x_px, color="magenta", lw=0.8, alpha=0.4))
                    # Mirror on waveform panel
                    st["drawn"].append(
                        ax_wave.plot(twt, pwr, marker="D", color="magenta",
                                     markersize=7, zorder=6, alpha=0.6,
                                     markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_wave.axvline(twt, color="magenta", lw=0.8, alpha=0.4))

            # Bed suppress: show X at algo bed position
            if ov.get("bed_suppress"):
                if algo_bed_x is not None and 0 <= algo_bed_x < frame_w:
                    y_px = trace_y_s[algo_bed_x]
                    st["drawn"].append(
                        ax_img.plot(algo_bed_x, y_px, marker="X", color="cyan",
                                    markersize=12, zorder=6, alpha=0.8,
                                    markeredgecolor="white", markeredgewidth=1.0)[0])
                st["drawn"].append(
                    ax_img.text(0.02, 0.90, "BED SUPPRESSED", transform=ax_img.transAxes,
                                fontsize=9, color="cyan", fontweight="bold",
                                va="top", ha="left",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="cyan", alpha=0.85)))
            elif "bed_twt_us" in ov:
                x_px = ov.get("bed_x_px", mb_x + int(round(ov["bed_twt_us"] / us_per_px)))
                twt = (x_px - mb_x) * us_per_px
                if 0 <= x_px < frame_w:
                    y_px = trace_y[x_px]
                    pwr = float(px_to_db(y_px, cal))
                    # Diamond on image panel (at clicked position)
                    st["drawn"].append(
                        ax_img.plot(x_px, y_px, marker="D", color="cyan",
                                    markersize=7, zorder=6, alpha=0.6,
                                    markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_img.axvline(x_px, color="cyan", lw=0.8, alpha=0.4))
                    # Mirror on waveform panel
                    st["drawn"].append(
                        ax_wave.plot(twt, pwr, marker="D", color="cyan",
                                     markersize=7, zorder=6, alpha=0.6,
                                     markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_wave.axvline(twt, color="cyan", lw=0.8, alpha=0.4))

            _update_status()
            fig.canvas.draw_idle()

        def on_click(event):
            # Click on image panel -> convert pixel x to TWT, capture y for display
            if event.inaxes != ax_img or event.button != 1:
                return
            x_px = event.xdata
            y_px = event.ydata
            if x_px is None or x_px < mb_x:
                return
            x_px = int(round(x_px))
            if x_px >= frame_w:
                return
            twt = (x_px - mb_x) * us_per_px
            if st["mode"] == "s":
                st["override"].pop("surface_suppress", None)
                st["override"]["surface_twt_us"] = round(twt, 3)
                st["override"]["surface_x_px"] = x_px
                st["override"]["surface_y_px"] = round(float(y_px), 1)
            elif st["mode"] == "b":
                st["override"].pop("bed_suppress", None)
                st["override"]["bed_twt_us"] = round(twt, 3)
                st["override"]["bed_x_px"] = x_px
                st["override"]["bed_y_px"] = round(float(y_px), 1)
            _redraw_overrides()

        def on_key(event):
            k = event.key
            if k == "s":
                st["mode"] = "s"
                title_obj.set_text(
                    f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
                    f"({frame_i + 1}/{len(review_frames)})  mode=[SURFACE]")
                _update_status()
                fig.canvas.draw_idle()
            elif k == "b":
                st["mode"] = "b"
                title_obj.set_text(
                    f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
                    f"({frame_i + 1}/{len(review_frames)})  mode=[BED]")
                _update_status()
                fig.canvas.draw_idle()
            elif k == "S":   # Shift+S: suppress surface (mark as not present)
                # Remove any surface position override, set suppress flag
                st["override"].pop("surface_twt_us", None)
                st["override"].pop("surface_x_px", None)
                st["override"].pop("surface_y_px", None)
                st["override"]["surface_suppress"] = True
                _redraw_overrides()
            elif k == "B":   # Shift+B: suppress bed (mark as not present)
                st["override"].pop("bed_twt_us", None)
                st["override"].pop("bed_x_px", None)
                st["override"].pop("bed_y_px", None)
                st["override"]["bed_suppress"] = True
                _redraw_overrides()
            elif k == "d":
                st["override"].clear()
                _redraw_overrides()
            elif k in ("n", "right"):
                st["action"] = "next"
                plt.close(fig)
            elif k in ("p", "left"):
                st["action"] = "prev"
                plt.close(fig)
            elif k in ("q", "escape"):
                st["action"] = "quit"
                plt.close(fig)

        # Unbind matplotlib defaults that clash with review keys
        for action in ("keymap.save", "keymap.quit", "keymap.back",
                        "keymap.forward", "keymap.fullscreen"):
            try:
                plt.rcParams[action] = []
            except KeyError:
                pass
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        _redraw_overrides()
        plt.show()

        # Save override for this frame
        if st["override"]:
            echo_overrides[fkey] = st["override"]
        elif fkey in echo_overrides:
            del echo_overrides[fkey]

        # Navigate
        if st["action"] == "next" or st["action"] is None:
            frame_i += 1
        elif st["action"] == "prev":
            frame_i = max(0, frame_i - 1)
        elif st["action"] == "quit":
            break

    # Persist overrides
    save_echo_overrides(echo_overrides)
    n_ov = len(echo_overrides)
    print(f"\nReview complete. {n_ov} override{'s' if n_ov != 1 else ''} saved.")


# ==============================================================================
# Branch: review mode vs normal extraction
# ==============================================================================

if _args.review:
    review_mode()
    print("\n" + "=" * 70)
    print("Re-running normal mode to apply overrides...")
    print("=" * 70 + "\n")
    # Reload overrides that review_mode just saved to disk
    echo_overrides = load_echo_overrides()
    # Fall through to normal extraction below


# -- Per-frame echo extraction --------------------------------------------------
print(f"  {'Fr':>3}  {'CBD':>6}  {'Status':^12}  {'NF':>6}  "
      f"{'Srf TWT':>8}  {'Srf dB':>7}  "
      f"{'Bed TWT':>8}  {'Bed dB':>7}  {'Bed SNR':>7}  "
      f"{'h_air':>7}  {'h_ice':>7}")
print("  " + "-" * 90)

echo_rows = []

for _, idx_row in tiff_rows.iterrows():
    frame_idx  = int(idx_row["frame_idx"])
    frame_type = idx_row["frame_type"]

    if frame_type == "partial":
        continue
    if frame_idx not in frame_cal:
        # Frame was excluded in step2 (or step2 hasn't been run for this frame)
        continue

    fc      = frame_cal[frame_idx]
    cbd     = fc["cbd"]
    file_id = fc["file_id"]
    mb_x    = fc["mb_x"]

    left_px  = int(idx_row["left_px"])
    right_px = int(idx_row["right_px"])
    frame    = img_norm[:, left_px : right_px + 1]
    frame_w  = frame.shape[1]

    cal = _make_cal(fc)

    # -- Extract trace and noise floor ------------------------------------------
    # Use robust mode: constrained-argmin rejects film-grain artefacts and
    # T/R ringing noise that would otherwise create fake peaks in the flat
    # region between the main bang and the first real echo.
    trace_y, trace_y_s, trace_info = extract_astra_science_trace(
        frame, cal, mb_x, f"{TIFF.name} {file_id} Phase 4",
        fc["crop_start_y"], fc["crop_end_y"])
    noise_floor_dB     = estimate_noise_floor(trace_y_s, mb_x, cal)

    # -- Detect surface and bed echoes ------------------------------------------
    surface_x, bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

    # -- Save algorithmic positions before overrides ----------------------------
    algo_surface_x = surface_x
    algo_bed_x     = bed_x

    # -- Apply manual overrides (from echo_overrides.json) --------------------
    is_surface_override = is_bed_override = False
    is_surface_suppress = is_bed_suppress = False
    override = echo_overrides.get(file_id, {})
    us_per_px = cal["us_per_px"]
    if override.get("surface_suppress"):
        surface_x = None
        is_surface_suppress = True
        is_surface_override = True
    elif "surface_twt_us" in override:
        if "surface_x_px" not in override:
            warnings.warn(f"{TIFF.name} {file_id}: legacy surface override is stored "
                          "as travel time, not an absolute frame column; review it after "
                          "changing the main bang.", RuntimeWarning)
        surface_x = override.get("surface_x_px",
                                 mb_x + int(round(override["surface_twt_us"] / us_per_px)))
        surface_x = max(0, min(surface_x, frame_w - 1))
        is_surface_override = True
    if override.get("bed_suppress"):
        bed_x = None
        is_bed_suppress = True
        is_bed_override = True
    elif "bed_twt_us" in override:
        if "bed_x_px" not in override:
            warnings.warn(f"{TIFF.name} {file_id}: legacy bed override is stored "
                          "as travel time, not an absolute frame column; review it after "
                          "changing the main bang.", RuntimeWarning)
        bed_x = override.get("bed_x_px",
                             mb_x + int(round(override["bed_twt_us"] / us_per_px)))
        bed_x = max(0, min(bed_x, frame_w - 1))
        is_bed_override = True

    # -- Compute echo metrics ---------------------------------------------------
    # Use local column profile extraction: immune to global trace failures
    # (graticule lines, reference lines competing with CRT signal).
    # Falls back to compute_echo_metrics only if local profile fails.
    surface = extract_local_profile(
        frame, surface_x, mb_x, noise_floor_dB, cal,
        guide_y=float(trace_y_s[surface_x]) if surface_x < len(trace_y_s) else None,
        max_half_width_us=2.0,   # surface echo physically < 2 µs half-width
        peak_y_override=astra_peak_row(trace_y, surface_x, fc["y_grid_px"],
                                      f"{TIFF.name} {file_id} surface"),
    ) if surface_x is not None else None

    bed = extract_local_profile(
        frame, bed_x, mb_x, noise_floor_dB, cal,
        guide_y=float(trace_y_s[bed_x]) if bed_x < len(trace_y_s) else None,
        max_half_width_us=4.0,   # bed echo physically < 4 µs half-width
        peak_y_override=astra_peak_row(trace_y, bed_x, fc["y_grid_px"],
                                      f"{TIFF.name} {file_id} bed"),
    ) if bed_x is not None else None

    # -- Classify echo_status ---------------------------------------------------
    # Use gain-corrected bed SNR for the weak_bed threshold.  The attenuator
    # sits before the receiver, so the noise floor is constant but the signal
    # level shifts with attenuator/CRT gain changes.  Raw SNR is therefore
    # inflated (or deflated) by the gain offset; corrected SNR reflects the
    # true signal-to-noise ratio.
    _mb_corr = fc["mb_correction_dB"]
    if surface is None:
        echo_status = "no_surface"
    elif bed is None:
        echo_status = "no_bed"
    elif (bed.peak_snr_dB + _mb_corr) < WEAK_BED_SNR_DB:
        echo_status = "weak_bed"
    else:
        echo_status = "good"

    # -- Original algorithmic echo_status (before overrides) -----------------
    # Preserved for post-hoc threshold tuning: what did the algorithm find
    # before the user corrected it?  Only differs from echo_status when an
    # override was applied.
    if not (is_surface_override or is_bed_override):
        algo_echo_status = echo_status
    elif algo_surface_x is None:
        algo_echo_status = "no_surface"
    elif algo_bed_x is None:
        algo_echo_status = "no_bed"
    else:
        _algo_bed = extract_local_profile(
            frame, algo_bed_x, mb_x, noise_floor_dB, cal,
            guide_y=float(trace_y_s[algo_bed_x]) if algo_bed_x < len(trace_y_s) else None,
            peak_y_override=trace_y[algo_bed_x])
        algo_echo_status = ("weak_bed"
                            if (_algo_bed.peak_snr_dB + _mb_corr) < WEAK_BED_SNR_DB
                            else "good")

    # -- Bed envelope artifact flag -------------------------------------------
    # Check whether film artifacts exist in the post-bed region (between
    # the bed peak and the NF+5 trailing edge + 1 µs margin).  Artifacts
    # here bias the trailing edge, inflating width_5, asymmetry, and
    # trailing_tail.  Detection: compare constrained trace vs median-
    # filtered trace (art_diff_px from extract_trace).  Columns where the
    # median filter removed a >10 dB dip in this region are flagged.
    #
    # Artifacts elsewhere (pre-MB, surface-bed gap, pre-surface) are
    # benign for bed envelope estimation and are not flagged.
    bed_envelope_suspect = False
    art_max_dB = 0.0
    art_diff_px = trace_info.get("art_diff_px")
    if bed is not None and art_diff_px is not None:
        bed_trail_5 = getattr(bed, "trail_5_us", np.nan)
        if not np.isnan(bed_trail_5):
            # Check region: bed_peak to trail_5 + 1 µs
            us_per_px = cal["us_per_px"]
            db_per_px = cal["db_per_px"]
            post_bed_start = bed.peak_x
            post_bed_end   = min(len(art_diff_px),
                                 mb_x + int((bed_trail_5 + 1.0) / us_per_px))
            if post_bed_start < post_bed_end:
                region = art_diff_px[post_bed_start:post_bed_end]
                # >10 dB discrepancy = significant artifact
                art_thresh_px = 10.0 / db_per_px
                art_mask = region > art_thresh_px
                if np.any(art_mask):
                    art_max_dB = float(np.max(region[art_mask]) * db_per_px)
                    bed_envelope_suspect = True

    # -- Derived geometry -------------------------------------------------------
    h_air_m = h_ice_m = h_eff_m = np.nan
    h_ice_onset_m = np.nan
    if surface is not None:
        h_air_m = surface.peak_twt_us / 2.0 * C_AIR_M_PER_US
    if surface is not None and bed is not None:
        h_ice_m = (bed.peak_twt_us - surface.peak_twt_us) / 2.0 * C_ICE_M_PER_US
        h_eff_m = h_air_m + h_ice_m / N_ICE
        # Onset-based ice thickness: uses first-break TWT for both surface and
        # bed when available, falling back to peak TWT when onset is NaN.
        srf_onset = surface.onset_twt_us if not np.isnan(surface.onset_twt_us) else surface.peak_twt_us
        bed_onset = bed.onset_twt_us if not np.isnan(bed.onset_twt_us) else bed.peak_twt_us
        h_ice_onset_m = (bed_onset - srf_onset) / 2.0 * C_ICE_M_PER_US

    # -- Print summary line -----------------------------------------------------
    srf_twt = f"{surface.peak_twt_us:8.3f}" if surface else f"{'—':>8}"
    srf_db  = f"{surface.peak_power_dB:+7.1f}" if surface else f"{'—':>7}"
    bed_twt = f"{bed.peak_twt_us:8.3f}" if bed else f"{'—':>8}"
    bed_db  = f"{bed.peak_power_dB:+7.1f}" if bed else f"{'—':>7}"
    bed_snr = f"{bed.peak_snr_dB:+7.1f}" if bed else f"{'—':>7}"
    h_air_s = f"{h_air_m:7.0f}" if not np.isnan(h_air_m) else f"{'—':>7}"
    h_ice_s = f"{h_ice_m:7.0f}" if not np.isnan(h_ice_m) else f"{'—':>7}"

    ov_tag = ""
    if is_surface_override or is_bed_override:
        ov_tag = "  [M]"
    print(f"  {frame_idx:>3}  {str(cbd):>6}  {echo_status:^12}  "
          f"{noise_floor_dB:+6.1f}  "
          f"{srf_twt}  {srf_db}  "
          f"{bed_twt}  {bed_db}  {bed_snr}  "
          f"{h_air_s}  {h_ice_s}{ov_tag}")

    # -- Build output row -------------------------------------------------------
    def _e(echo, attr):
        return getattr(echo, attr, np.nan) if echo is not None else np.nan

    # Per-frame MB power correction: normalise echo powers to the flight-wide
    # reference MB level.  Removes system gain variation (attenuator changes,
    # CRT intensity drift) while preserving real geophysical power differences.
    # The MB is a fixed internal signal (T/R switch leakage) — any change in
    # its measured power is purely instrumental.
    _mb_corr = fc["mb_correction_dB"]  # positive = frame was weaker -> boost

    def _corrected(raw_dB):
        """Apply MB power correction to a raw dB measurement."""
        if np.isnan(raw_dB):
            return np.nan
        return round(raw_dB + _mb_corr, 2)

    echo_rows.append(dict(
        flight      = FLT,
        tiff        = TIFF.name,
        tiff_id     = TIFF_ID,
        frame_idx   = frame_idx,
        cbd         = cbd,
        file_id     = file_id,
        mb_x        = mb_x,
        astra_x_ref_px = fc["x_ref_px"] if fc["x_ref_px"] is not None else "",
        astra_x_grid_px = json.dumps(fc["x_grid_px"]),
        astra_y_grid_px = json.dumps(fc["y_grid_px"]),
        cal_pick_source = fc["cal_pick_source"],
        mb_detection_source = fc["mb_detection_source"],
        y_ref_px    = fc["y_ref_px"],
        db_per_px   = fc["db_per_px"],
        us_per_px   = fc["us_per_px"],
        echo_status = echo_status,
        noise_floor_dB = round(noise_floor_dB, 2),
        # MB power correction
        mb_power_dB       = round(fc["mb_power_dB"], 2),
        mb_correction_dB  = round(_mb_corr, 2),
        # Surface echo (power and SNR corrected for system gain)
        surface_twt_us        = round(_e(surface, "peak_twt_us"),     4),
        surface_power_dB      = _corrected(_e(surface, "peak_power_dB")),
        surface_snr_dB        = _corrected(_e(surface, "peak_snr_dB")),
        surface_width_10_us   = round(_e(surface, "width_10_us"),     4),
        surface_width_5_us    = round(_e(surface, "width_5_us"),      4),
        surface_peakiness     = round(_e(surface, "peakiness"),       3),
        surface_asymmetry     = round(_e(surface, "asymmetry"),       3),
        surface_leading_rise_us  = round(_e(surface, "leading_rise_us"),  4),
        surface_trailing_tail_us = round(_e(surface, "trailing_tail_us"), 4),
        surface_onset_twt_us     = round(_e(surface, "onset_twt_us"),    4),
        # Bed echo (power and SNR corrected for system gain)
        bed_twt_us            = round(_e(bed, "peak_twt_us"),    4),
        bed_power_dB          = _corrected(_e(bed, "peak_power_dB")),
        bed_snr_dB            = _corrected(_e(bed, "peak_snr_dB")),
        bed_width_10_us       = round(_e(bed, "width_10_us"),    4),
        bed_width_5_us        = round(_e(bed, "width_5_us"),     4),
        bed_peakiness         = round(_e(bed, "peakiness"),      3),
        bed_asymmetry         = round(_e(bed, "asymmetry"),      3),
        bed_leading_rise_us   = round(_e(bed, "leading_rise_us"),  4),
        bed_trailing_tail_us  = round(_e(bed, "trailing_tail_us"), 4),
        bed_onset_twt_us      = round(_e(bed, "onset_twt_us"),    4),
        # Peak-relative shape metrics (SNR-independent)
        bed_decay_10_us       = round(_e(bed, "decay_10_us"),     4),
        bed_decay_20_us       = round(_e(bed, "decay_20_us"),     4),
        bed_rise_10_us        = round(_e(bed, "rise_10_us"),      4),
        bed_pk_asymmetry      = round(_e(bed, "pk_asymmetry"),    3),
        # Derived: peak-onset delay (Fresnel-zone roughness diagnostic)
        bed_peak_onset_us     = round(
            _e(bed, "peak_twt_us") - _e(bed, "onset_twt_us"), 4),
        bed_envelope_suspect  = bed_envelope_suspect,
        artifact_max_dB       = round(art_max_dB, 1),
        # Geometry (unaffected by gain correction)
        h_air_m = round(h_air_m, 1) if not np.isnan(h_air_m) else np.nan,
        h_ice_m = round(h_ice_m, 1) if not np.isnan(h_ice_m) else np.nan,
        h_ice_onset_m = round(h_ice_onset_m, 1) if not np.isnan(h_ice_onset_m) else np.nan,
        h_eff_m = round(h_eff_m, 1) if not np.isnan(h_eff_m) else np.nan,
        # Override flags
        surface_override = is_surface_override,
        bed_override     = is_bed_override,
        algo_echo_status = algo_echo_status,
        reflector_type   = "bed",
    ))

    # -- Generate single-panel diagnostic figure ---------------------------------
    # Full-width A-scope image with dual axes (px + physical units),
    # echo markers, and semi-transparent envelope polygons.
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    fw = frame.shape[1]
    y_ref_px = cal["y_ref_px"]
    y_lo_disp = DEFAULT_CAL["y_disp_lo"] - 50
    y_hi_disp = DEFAULT_CAL["y_disp_hi"] + 50

    ax.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
              extent=[0, fw, frame.shape[0], 0])

    # Noise floor reference row (cyan dashed)
    ax.axhline(y_ref_px, color="cyan", lw=1.0, ls="--", alpha=0.6)

    # Main bang (red dashed vertical)
    ax.axvline(mb_x, color="red", lw=1.2, ls="--", alpha=0.8)
    _draw_astra_pixel_grid(ax, fc)

    # -- Envelope polygons (semi-transparent, on CRT image) --------------------
    def _draw_envelope(ax_obj, echo, echo_x, color):
        """Draw semi-transparent envelope polygon for an echo."""
        if echo is None:
            return
        # +10 dB envelope
        if not np.isnan(echo.lead_10_us) and not np.isnan(echo.trail_10_us):
            l10_px = us_to_px(echo.lead_10_us, mb_x, cal)
            t10_px = us_to_px(echo.trail_10_us, mb_x, cal)
            thresh_10_y = db_to_px(noise_floor_dB + 10.0, cal)
            # Polygon: from threshold row at leading edge, up to peak, back to
            # threshold at trailing edge
            xs = [l10_px, echo_x, t10_px, t10_px, l10_px]
            ys = [thresh_10_y, echo.peak_y, thresh_10_y, thresh_10_y, thresh_10_y]
            ax_obj.fill(xs[:3], ys[:3], color=color, alpha=0.15, zorder=3)
        # +5 dB envelope (wider, fainter)
        if not np.isnan(echo.lead_5_us) and not np.isnan(echo.trail_5_us):
            l5_px = us_to_px(echo.lead_5_us, mb_x, cal)
            t5_px = us_to_px(echo.trail_5_us, mb_x, cal)
            thresh_5_y = db_to_px(noise_floor_dB + 5.0, cal)
            xs5 = [l5_px, echo_x, t5_px]
            ys5 = [thresh_5_y, echo.peak_y, thresh_5_y]
            ax_obj.fill(xs5, ys5, color=color, alpha=0.08, zorder=2)

    _draw_envelope(ax, surface, surface_x if surface_x is not None else 0,
                   "limegreen")
    _draw_envelope(ax, bed, bed_x if bed_x is not None else 0,
                   "darkorange")

    # -- Echo markers ----------------------------------------------------------
    # Surface
    if is_surface_suppress:
        if algo_surface_x is not None:
            ax.plot(algo_surface_x, y_ref_px, marker="X",
                    color="lime", markersize=12, zorder=5, alpha=0.6,
                    markeredgecolor="white", markeredgewidth=1.0)
    elif surface is not None:
        srf_marker = "D" if is_surface_override else "^"
        srf_label  = (f"Srf {surface.peak_twt_us:.2f}us  "
                      f"{surface.peak_power_dB:+.1f}dB"
                      + (" [M]" if is_surface_override else ""))
        ax.plot(surface_x, surface.peak_y, marker=srf_marker, color="lime",
                markersize=10, zorder=6, label=srf_label,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(surface_x, color="lime", lw=0.6, ls=":", alpha=0.5)

    # Bed
    if is_bed_suppress:
        if algo_bed_x is not None:
            ax.plot(algo_bed_x, y_ref_px, marker="X",
                    color="orange", markersize=12, zorder=5, alpha=0.6,
                    markeredgecolor="white", markeredgewidth=1.0)
    elif bed is not None:
        bed_marker = "D" if is_bed_override else "^"
        bed_label  = (f"Bed {bed.peak_twt_us:.2f}us  "
                      f"{bed.peak_power_dB:+.1f}dB  "
                      f"SNR={bed.peak_snr_dB:+.1f}dB"
                      + (" [M]" if is_bed_override else ""))
        ax.plot(bed_x, bed.peak_y, marker=bed_marker, color="orange",
                markersize=10, zorder=6, label=bed_label,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(bed_x, color="orange", lw=0.6, ls=":", alpha=0.5)

    # -- Primary axes: pixel coordinates ---------------------------------------
    ax.set_ylim(y_hi_disp, y_lo_disp)
    ax.set_xlim(0, fw)
    ax.set_xlabel("Column (px, frame-relative)", fontsize=8)
    ax.set_ylabel("Row (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -- Secondary X-axis: TWT from MB (us) ------------------------------------
    ax2_x = ax.twiny()
    ax2_x.set_xlim((0 - mb_x) * cal["us_per_px"],
                    (fw - mb_x) * cal["us_per_px"])
    ax2_x.set_xlabel("Two-way travel time from MB (us)", fontsize=8)
    ax2_x.tick_params(labelsize=7)
    ax2_x.spines["top"].set_linewidth(0.4)

    # -- Secondary Y-axis: Power (dB) -----------------------------------------
    ax2_y = ax.twinx()
    db_top    = px_to_db(y_lo_disp, cal)
    db_bottom = px_to_db(y_hi_disp, cal)
    ax2_y.set_ylim(db_bottom, db_top)
    ax2_y.set_ylabel("Power (dB)", fontsize=8)
    ax2_y.tick_params(labelsize=7)
    ax2_y.spines["right"].set_linewidth(0.4)

    # -- Legend and title -------------------------------------------------------
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7,
                  facecolor="black", labelcolor="white")

    # Geometry annotation (bottom-left)
    geo_parts = []
    if not np.isnan(h_air_m):
        geo_parts.append(f"h_air={h_air_m:.0f} m")
    if not np.isnan(h_ice_m):
        geo_parts.append(f"h_ice={h_ice_m:.0f} m")
    if geo_parts:
        ax.text(0.01, 0.04, "  |  ".join(geo_parts),
                transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          edgecolor="none", alpha=0.6))

    ax.set_title(
        f"LYRA Phase 4 -- F{FLT} {file_id}  |  NF={noise_floor_dB:.1f} dB  "
        f"[{echo_status}]",
        fontsize=9, loc="left")

    fig_path = PHASE4_DIR / f"F{FLT}_{file_id}_echoes.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Clean calibrated waveform figure (Panel-C style) -----------------------
    # Uses the SAME calibrated trace and the SAME final surface/bed picks as the
    # diagnostic above. This is visualization only; no repicking/recalculation.
    twt_arr = px_to_us(np.arange(frame_w, dtype=float), mb_x, cal)

    # Show the same full red trace used for the selected peak powers.
    power_arr = px_to_db(trace_y, cal)

    wf_fig, wf_ax = plt.subplots(figsize=(10, 6))
    wf_fig.patch.set_facecolor("white")

    # Show the COMPLETE calibrated A-scope trace.
    # Keep all available samples before and after the detected main bang so the
    # transmitter/main-bang waveform itself is visible rather than being clipped
    # at 0 us.  Time is still referenced to mb_x, so mb_x == 0 us.
    wf_mask = np.isfinite(twt_arr) & np.isfinite(power_arr)
    if np.any(wf_mask):
        wf_ax.plot(twt_arr[wf_mask], power_arr[wf_mask], color="red", lw=1.4)

    # -60 dB calibration reference.
    wf_ax.axhline(-60.0, color="green", lw=1.0, ls=":", alpha=0.8,
                  label="Reference (-60 dB)")

    # Transmitter / main-bang time zero.
    tx_power = (float(px_to_db(trace_y[mb_x], cal))
                if 0 <= mb_x < len(trace_y) else np.nan)
    if np.isfinite(tx_power):
        wf_ax.plot(0.0, tx_power, "o", color="royalblue", ms=7, zorder=5, label="Tx")
        wf_ax.annotate(
            f"transmitter pulse\n(~{tx_power:.1f} dB at 0.0 us)",
            xy=(0.0, tx_power), xytext=(-0.5, tx_power - 7.0),
            textcoords="data", ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="royalblue", lw=1.0),
        )

    # Final surface pick used by Phase 4.
    if surface is not None:
        s_t = float(surface.peak_twt_us)
        s_p = float(surface.peak_power_dB)
        wf_ax.plot(s_t, s_p, "o", color="seagreen", ms=7, zorder=6, label="Surface")
        wf_ax.annotate(
            f"surface\n(~{s_p:.1f} dB at {s_t:.1f} us)",
            xy=(s_t, s_p), xytext=(s_t - 3.0, s_p - 5.0),
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="seagreen", lw=1.0),
        )

    # Final bed pick used by Phase 4.
    if bed is not None:
        b_t = float(bed.peak_twt_us)
        b_p = float(bed.peak_power_dB)
        wf_ax.plot(b_t, b_p, "o", color="mediumorchid", ms=7, zorder=6, label="Bed")
        wf_ax.annotate(
            f"bed\n(~{b_p:.1f} dB at {b_t:.1f} us)",
            xy=(b_t, b_p), xytext=(b_t + 3.0, b_p - 5.0),
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="mediumorchid", lw=1.0),
        )

    wf_ax.set_xlabel("Two-way travel time from MB (us)")
    wf_ax.set_ylabel("Power (dB)")
    wf_ax.set_title(f"LYRA Calibrated A-scope: F{FLT} {file_id}")
    wf_ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    wf_ax.minorticks_on()
    wf_ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25)
    wf_ax.legend(loc="upper right", framealpha=0.9)

    # DISPLAY CROP ONLY.  The full frame is still extracted above, but the
    # publication-style waveform omits the ugly frame-edge regions.  Keep a
    # short pre-MB interval so the transmitter pulse is visible, then continue
    # several microseconds beyond the bed to retain the useful post-bed tail.
    if np.any(wf_mask):
        _available_lo = float(np.nanmin(twt_arr[wf_mask]))
        _available_hi = float(np.nanmax(twt_arr[wf_mask]))
        _plot_lo = max(_available_lo, -2.5)
        if bed is not None and np.isfinite(float(bed.peak_twt_us)):
            _plot_hi = min(_available_hi, float(bed.peak_twt_us) + 6.0)
        elif surface is not None and np.isfinite(float(surface.peak_twt_us)):
            _plot_hi = min(_available_hi, float(surface.peak_twt_us) + 12.0)
        else:
            _plot_hi = min(_available_hi, 20.0)
        wf_ax.set_xlim(_plot_lo, _plot_hi)
    finite_power = power_arr[wf_mask]
    if finite_power.size:
        wf_ax.set_ylim(min(-65.0, float(np.nanmin(finite_power)) - 2.0),
                       max(0.0, float(np.nanmax(finite_power)) + 3.0))

    waveform_path = PHASE4_DIR / f"F{FLT}_{file_id}_waveform.png"
    wf_fig.savefig(waveform_path, dpi=150, bbox_inches="tight", facecolor="white")
    
    # One publication PNG per A-scope, combining the complete TIFF frame strip
    # and two scientific panels. All measurements / picked positions are reused.
    # The figure is sized to 165 x 100 mm for consistent output dimensions.
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    MM_TO_IN = 1.0 / 25.4
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "text.usetex": False,
    }):
        pub = plt.figure(figsize=(165 * MM_TO_IN, 100 * MM_TO_IN),
                         facecolor="white")

        # Top: the same sequence and CBD identifiers as the horizontal contact
        # sheet created by detect_frames.py. Rebuild it from TIFF/frame-index
        # data so it stays legible at 165 mm and can highlight the current CBD.
        pub.text(0.5, 0.984,
                 f"LYRA A-scope analysis  |  F{FLT}  CBD {cbd}  |  "
                 f"{echo_status.replace('_', ' ')}",
                 ha="center", va="top", fontsize=10.0, fontweight="bold")
        strip_n = len(pub_frame_thumbnails)
        if strip_n:
            left = 0.027
            usable = 0.946
            tile_w = usable / strip_n
            for i, (fi, cb, is_partial, thumb) in enumerate(pub_frame_thumbnails):
                cx = left + (i + 0.5) * tile_w
                selected = (fi == frame_idx)
                if is_partial:
                    label = f"Frame {fi}\\npartial"
                else:
                    label = f"Frame {fi}\\nCBD {cb.zfill(4)}"
                # Separate two-line labels, never squeeze tiny text into images.
                pub.text(cx, 0.934, label.replace("\\n", "\n"),
                         ha="center", va="top", fontsize=5.8,
                         fontweight="bold" if selected else "normal",
                         linespacing=0.92,
                         color="#166534" if selected else "black")
                # Each film thumbnail uses equal X/Y image scale. No stretching.
                ax_strip = pub.add_axes([left + i * tile_w + 0.001,
                                         0.791, tile_w - 0.002, 0.112])
                ax_strip.imshow(thumb, cmap="gray", vmin=0, vmax=255,
                                interpolation="nearest", aspect="equal",
                                rasterized=True)
                ax_strip.set_axis_off()
                if selected:
                    pub.add_artist(Rectangle((left + i * tile_w + 0.001,
                                              0.791),
                                             tile_w - 0.002, 0.112,
                                             transform=pub.transFigure,
                                             fc="none", ec="#15803d", lw=1.4))

        # Lower panels have identical *physical* widths and heights. Use the
        # image pixel aspect ratio to set BOTH panel heights, rather than
        # squeezing film pixels with imshow(aspect='auto').
        film_y_span = float(y_hi_disp - y_lo_disp)
        # Narrow frames reduce both panel widths together; never change the
        # physical aspect of the film axes or make the waveform panel larger.
        panel_w = min(0.365, 0.390 * 100.0 / 165.0 * float(fw) / film_y_span)
        panel_h = panel_w * 165.0 / 100.0 * film_y_span / float(fw)
        panel_bottom = 0.295
        left_x = 0.088
        right_x = 0.590
        ax_film = pub.add_axes([left_x, panel_bottom, panel_w, panel_h])
        ax_curve = pub.add_axes([right_x, panel_bottom, panel_w, panel_h])

        pub.text(left_x + panel_w / 2, 0.729, "(a) Detected echoes",
                 ha="center", va="center", fontsize=9.5, fontweight="bold")
        pub.text(right_x + panel_w / 2, 0.729, "(b) Calibrated waveform",
                 ha="center", va="center", fontsize=9.5, fontweight="bold")

        # Calibrated film image: preserve true image aspect (no auto stretching)
        # and full horizontal extent of the frame, as used for echo detection.
        ax_film.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="equal",
                       extent=[0, fw, frame.shape[0], 0], rasterized=True)
        ax_film.set_xlim(0, fw)
        ax_film.set_ylim(y_hi_disp, y_lo_disp)
        ax_film.axvline(mb_x, color="#d62728", ls="--", lw=1.0)
        ax_film.axhline(y_ref_px, color="#00bcd4", ls="--", lw=0.85)
        _draw_astra_pixel_grid(ax_film, fc)

        for _echo, _xx, _edge, _pale in (
            (surface, surface_x, "#198754", "#b9e9cc"),
            (bed, bed_x, "#d97706", "#ffe0a3"),
        ):
            if _echo is None or _xx is None:
                continue
            if np.isfinite(_echo.lead_10_us) and np.isfinite(_echo.trail_10_us):
                _x1 = us_to_px(float(_echo.lead_10_us), mb_x, cal)
                _x2 = us_to_px(float(_echo.trail_10_us), mb_x, cal)
                _yt = db_to_px(noise_floor_dB + 10.0, cal)
                ax_film.fill([_x1, _xx, _x2], [_yt, _echo.peak_y, _yt],
                             fc=_pale, ec=_edge, lw=0.7, zorder=4)
            ax_film.plot([_xx], [_echo.peak_y], marker="^", color=_edge,
                         ms=5.0, mec="white", mew=0.7, ls="None", zorder=6)
        ax_film.set_xlabel("Image column (px)", fontsize=8.3, labelpad=2)
        ax_film.set_ylabel("Image row (px)", fontsize=8.3, labelpad=1)
        ax_film.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax_film.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax_film.tick_params(axis="both", labelsize=7.7, length=2.5, pad=2)

        ax_film_top = ax_film.twiny()
        ax_film_top.set_xlim((-mb_x) * cal["us_per_px"],
                              (fw - mb_x) * cal["us_per_px"])
        ax_film_top.set_xlabel("TWTT from MB (µs)", fontsize=8.3, labelpad=1.5)
        ax_film_top.xaxis.set_major_locator(MaxNLocator(5))
        ax_film_top.tick_params(axis="x", labelsize=7.7, length=2.5, pad=2)

        ax_film_power = ax_film.twinx()
        ax_film_power.set_ylim(px_to_db(y_hi_disp, cal),
                               px_to_db(y_lo_disp, cal))
        ax_film_power.set_ylabel("dB", fontsize=8.0, labelpad=1)
        ax_film_power.yaxis.set_major_locator(MaxNLocator(5))
        ax_film_power.tick_params(axis="y", labelsize=7.7, length=2.5, pad=2)

        # Calibrated waveform: unchanged data and picks, exactly matched panel.
        if np.any(wf_mask):
            ax_curve.plot(twt_arr[wf_mask], power_arr[wf_mask],
                          color="#d62728", lw=1.2, zorder=3)
        ax_curve.axhline(-60.0, color="green", ls=":", lw=0.9, zorder=2)
        if np.isfinite(tx_power):
            ax_curve.plot([0.0], [tx_power], "o", color="royalblue", ms=5, zorder=5)
        if surface is not None:
            ax_curve.plot([surface.peak_twt_us], [surface.peak_power_dB],
                          "o", color="seagreen", ms=5, zorder=6)
        if bed is not None:
            ax_curve.plot([bed.peak_twt_us], [bed.peak_power_dB],
                          "o", color="mediumorchid", ms=5, zorder=6)
        ax_curve.set_xlim(wf_ax.get_xlim())
        ax_curve.set_ylim(wf_ax.get_ylim())
        ax_curve.set_xlabel("TWTT from MB (µs)", fontsize=8.3, labelpad=2)
        ax_curve.set_ylabel("Power (dB)", fontsize=8.3, labelpad=1.5)
        ax_curve.xaxis.set_major_locator(MaxNLocator(5))
        ax_curve.yaxis.set_major_locator(MaxNLocator(5))
        ax_curve.tick_params(axis="both", labelsize=7.7, length=2.5, pad=2)
        ax_curve.grid(True, which="major", ls="-", lw=0.4,
                      color="#d0d0d0", zorder=0)

        handles = [
            Line2D([0], [0], color="royalblue", marker="o", ls="None", ms=4.0, label="Tx"),
            Line2D([0], [0], color="seagreen", marker="o", ls="None", ms=4.0, label="Surface"),
            Line2D([0], [0], color="mediumorchid", marker="o", ls="None", ms=4.0, label="Bed"),
            Line2D([0], [0], color="green", ls=":", lw=1, label="-60 dB"),
        ]
        ax_curve.legend(handles=handles, loc="upper right", ncol=2,
                        framealpha=1, facecolor="white", edgecolor="#bbbbbb",
                        fontsize=7.0, borderpad=0.22, labelspacing=0.2,
                        handletextpad=0.2, columnspacing=0.45)

        # Measurements below panels, large enough to read at final 165 mm width.
        values = []
        if surface is not None:
            values.append(f"Surface: {surface.peak_twt_us:.2f} µs / {surface.peak_power_dB:+.1f} dB")
        if bed is not None:
            values.append(f"Bed: {bed.peak_twt_us:.2f} µs / {bed.peak_power_dB:+.1f} dB")
        pub.text(0.5, 0.167, "     |     ".join(values) if values else "No resolved echoes",
                 ha="center", va="center", fontsize=8.5)
        metrics = [f"Noise: {noise_floor_dB:.1f} dB"]
        if np.isfinite(tx_power):
            metrics.append(f"Tx: {tx_power:+.1f} dB")
        if np.isfinite(h_air_m):
            metrics.append(f"Air range: {h_air_m:.0f} m")
        if np.isfinite(h_ice_m):
            metrics.append(f"Ice thickness: {h_ice_m:.0f} m")
        pub.text(0.5, 0.092, "     |     ".join(metrics),
                 ha="center", va="center", fontsize=8.0)

        publication_png = PHASE4_DIR / f"F{FLT}_{file_id}_publication.png"
        pub.savefig(publication_png, dpi=150, facecolor="white")
        plt.close(pub)

    plt.close(wf_fig)


# -- Update echo CSV (incremental, TIFF-specific rows) -------------------------
if echo_rows:
    new_df = pd.DataFrame(echo_rows)
    if ECHO_CSV.exists():
        existing = pd.read_csv(ECHO_CSV, dtype=str)
        # Preserve non-"bed" reflector_type flags set by the user
        rt_map: dict[str, str] = {}
        if "reflector_type" in existing.columns:
            for _, r in existing[existing["tiff"] == TIFF.name].iterrows():
                rt = r.get("reflector_type", "bed")
                if pd.notna(rt) and str(rt) != "bed":
                    rt_map[str(r["file_id"])] = str(rt)
        existing = existing[existing["tiff"] != TIFF.name]
        merged   = pd.concat([existing, new_df.astype(str)], ignore_index=True)
        # Re-apply preserved flags to new rows
        if rt_map:
            for fid, rt in rt_map.items():
                merged.loc[merged["file_id"] == fid, "reflector_type"] = rt
    else:
        merged = new_df.astype(str)
    # Deduplicate overlapping TIFFs: keep highest SNR per CBD
    if "cbd" in merged.columns and "bed_snr_dB" in merged.columns:
        merged["_snr_sort"] = pd.to_numeric(merged["bed_snr_dB"], errors="coerce")
        merged = merged.sort_values("_snr_sort", ascending=False)
        n_before = len(merged)
        merged = merged.drop_duplicates(subset=["cbd"], keep="first")
        merged = merged.drop(columns=["_snr_sort"])
        n_dropped = n_before - len(merged)
        if n_dropped > 0:
            print(f"  Deduped {n_dropped} overlapping-TIFF rows (kept highest SNR per CBD)")
        merged = merged.sort_values("cbd")
    merged.to_csv(ECHO_CSV, index=False)

    # -- Summary statistics -----------------------------------------------------
    print(f"\n{'-' * 60}")
    print(f"  Echo CSV -> {ECHO_CSV}")
    print(f"  Phase 4 figures -> {PHASE4_DIR}")
    print(f"  Frames processed : {len(echo_rows)}")

    statuses = [r["echo_status"] for r in echo_rows]
    for st in ["good", "weak_bed", "no_bed", "no_surface"]:
        n = statuses.count(st)
        if n:
            print(f"    {st:12s}: {n}")

    good_rows = [r for r in echo_rows if r["echo_status"] in ("good", "weak_bed")]
    if good_rows:
        h_airs = [r["h_air_m"] for r in good_rows
                  if not (isinstance(r["h_air_m"], float) and np.isnan(r["h_air_m"]))]
        h_ices = [r["h_ice_m"] for r in good_rows
                  if not (isinstance(r["h_ice_m"], float) and np.isnan(r["h_ice_m"]))]
        if h_airs:
            print(f"  h_air  : {np.mean(h_airs):.0f} ± {np.std(h_airs):.0f} m "
                  f"(n={len(h_airs)})")
        if h_ices:
            print(f"  h_ice  : {np.mean(h_ices):.0f} ± {np.std(h_ices):.0f} m "
                  f"(n={len(h_ices)})")

    # -- Generate review summary JSON (full flight, all TIFFs) -------------
    review_json_path = PHASE4_DIR / f"F{FLT}_echo_review.json"
    full_df = pd.read_csv(ECHO_CSV)
    review_data = {
        "flight": FLT,
        "total_frames": len(full_df),
        "summary": {},
    }
    for st_name in ["good", "weak_bed", "no_bed", "no_surface"]:
        st_df = full_df[full_df["echo_status"] == st_name]
        review_data["summary"][st_name] = len(st_df)

    # Per-status frame lists (only non-good statuses)
    for st_name in ["no_bed", "no_surface", "weak_bed"]:
        st_df = full_df[full_df["echo_status"] == st_name]
        if len(st_df) == 0:
            continue
        frames = []
        for _, row in st_df.iterrows():
            frames.append({
                "tiff_id": str(row.get("tiff_id", "")),
                "cbd": str(int(float(row["cbd"]))) if str(row["cbd"]).replace('.','',1).isdigit() else str(row["cbd"]),
                "file_id": str(row["file_id"]),
            })
        review_data[st_name] = frames

    # Per-TIFF breakdown: how many bad frames each TIFF has
    bad_df = full_df[full_df["echo_status"].isin(["no_bed", "no_surface", "weak_bed"])]
    if len(bad_df) > 0:
        tiff_breakdown = {}
        for tid, grp in bad_df.groupby("tiff_id"):
            counts = grp["echo_status"].value_counts().to_dict()
            tiff_breakdown[str(tid)] = {
                "total_bad": len(grp),
                **{k: int(v) for k, v in counts.items()},
            }
        # Sort by total_bad descending
        review_data["tiff_breakdown"] = dict(
            sorted(tiff_breakdown.items(), key=lambda x: x[1]["total_bad"], reverse=True)
        )

    with open(review_json_path, "w") as f:
        json.dump(review_data, f, indent=2)
    print(f"  Review JSON -> {review_json_path}")

print("\nDone.")
''',
    'otherechoes': r'''# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""
echoes.py — LYRA Phase 4: Surface/bed echo extraction
============================================================
For each calibrated frame (from phase3 cal CSV) extract surface and bed echoes:

  • Noise floor estimation (pre-bang 75th-percentile baseline)
  • Echo peak detection  (prominence >= 5 dB, distance >= 80 px)
  • Envelope walking at +5 dB and +10 dB above noise floor
  • Waveform shape metrics: peakiness, asymmetry, trailing_tail, leading_rise
  • Derived geometry:
      h_air_m  = surface_twt / 2 × c_air   (air column height)
      h_ice_m  = (bed_twt - surface_twt) / 2 × c_ice   (ice thickness)
      h_eff_m  = h_air_m + h_ice_m / n_ice   (effective propagation distance)

Echo status:
  good       — surface + bed detected, bed SNR >= WEAK_BED_SNR_DB (5 dB)
  weak_bed   — bed detected but SNR < WEAK_BED_SNR_DB (marginal; inspect figure)
  no_bed     — surface detected, no bed echo found
  no_surface — surface not detected; geometry undefined

Per-frame calibration is taken directly from the phase3 cal CSV (mb_x, y_ref_px,
db_per_px, us_per_px).  Detection parameters (prominence, distance, Gaussian sigma)
are taken from DEFAULT_CAL.

Usage
-----
Normal mode (extract echoes):
    python codes/echoes.py [TIFF_PATH]

Review mode (interactive override -- click on image panel):
    python codes/echoes.py TIFF --review          # bad frames only
    python codes/echoes.py TIFF --review --all     # all frames

Flight-wide mode (all TIFFs for a flight):
    python codes/echoes.py --flight 125                    # normal mode, all TIFFs
    python codes/echoes.py --flight 125 --review           # review all TIFFs
    python codes/echoes.py --flight 125 --review --all     # review all frames in all TIFFs

    Keys: [s]=surface mode  [b]=bed mode  click=set position
          [S]=suppress surface  [B]=suppress bed  (mark as not present)
          [d]=clear all overrides  [n/Right]=next  [p/Left]=prev  [q]=save & quit

Pass a TIFF path, or use --flight for the flight-wide mode.

Outputs
-------
Per-flight echo CSV (updated incrementally):
    output/F{FLT}/phase4/F{FLT}_echoes.csv
      Columns: flight, tiff, cbd, file_id, echo_status,
               noise_floor_dB, mb_power_dB, mb_correction_dB,
               surface_twt_us, surface_power_dB, surface_snr_dB,
               surface_width_10_us, surface_width_5_us, surface_peakiness,
               surface_asymmetry, surface_leading_rise_us, surface_trailing_tail_us,
               bed_twt_us, bed_power_dB, bed_snr_dB,
               bed_width_10_us, bed_width_5_us, bed_peakiness,
               bed_asymmetry, bed_leading_rise_us, bed_trailing_tail_us,
               h_air_m, h_ice_m, h_eff_m
      Power/SNR values are gain-corrected using the main bang as a per-frame
      reference (mb_correction_dB = flight_median_mb - frame_mb_power).

Per-frame diagnostic figure (two-panel):
    output/F{FLT}/phase4/F{FLT}_{file_id}_echoes.png
      Left  : frame image with echo markers (pixel space, same scale as phase3)
      Right : waveform in physical units (TWT µs vs power dB)
"""

from pathlib import Path
import sys
import os
import json
import argparse

import numpy as np
import pandas as pd

# -- Parse arguments before matplotlib (backend depends on --review) -----------
_parser = argparse.ArgumentParser(
    description="LYRA Phase 4: Surface/bed echo extraction",
    epilog="Review mode: --review shows bad frames for manual echo override",
)
_parser.add_argument("tiff", nargs="?", help="TIFF path or FLT/TIFF_ID format")
_parser.add_argument("--review", action="store_true",
                     help="Interactive review: click on image to override surface/bed picks")
_parser.add_argument("--all", action="store_true",
                     help="With --review: show all frames (default: only no_bed/no_surface/weak_bed)")
_parser.add_argument("--flight", type=int, metavar="FLT",
                     help="Process all TIFFs for a flight (e.g. --flight 125). "
                          "Works with --review to iterate through every TIFF.")
_args = _parser.parse_args()

import matplotlib
if not _args.review:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from lyra import display_path, output_root

# -- Flight-wide mode: loop through all TIFFs for a flight ---------------------
# When --flight N is given, find all TIFFs and re-invoke echoes.py on each one.
# Works with --review, --all, and normal mode.
if _args.flight is not None:
    import subprocess
    flt = _args.flight
    cal_csv = output_root(ROOT / "output") / f"F{flt}/phase3/F{flt}_cal.csv"
    if not cal_csv.exists():
        sys.exit(f"ERROR: No cal CSV at {cal_csv}")
    cal_df = pd.read_csv(cal_csv, dtype=str)
    tiff_ids = sorted(cal_df["tiff_id"].unique(), key=int)
    print(f"Flight F{flt}: {len(tiff_ids)} TIFFs to process\n")
    script = str(Path(__file__).resolve())
    for i, tid in enumerate(tiff_ids, 1):
        cmd = [sys.executable, script, f"{flt}/{tid}"]
        if _args.review:
            cmd.append("--review")
        if _args.all:
            cmd.append("--all")
        print(f"{'='*60}")
        print(f"  TIFF {tid}  ({i}/{len(tiff_ids)})")
        print(f"{'='*60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: TIFF {tid} exited with code {result.returncode}")
        print()
    print(f"Flight F{flt}: all {len(tiff_ids)} TIFFs done.")
    sys.exit(0)

from lyra import (
    DEFAULT_CAL,
    extract_trace, estimate_noise_floor,
    detect_echoes, compute_echo_metrics,
    extract_local_profile,
    px_to_db, px_to_us, db_to_px, us_to_px,
    C_AIR_M_PER_US, C_ICE_M_PER_US,
    ensure_canonical_name, output_root, parse_tiff_filename,
    resolve_tiff_arg, tiff_id,
)

# -- Scientific constants -------------------------------------------------------
N_ICE           = 1.78   # refractive index of ice (Neal 1977)
WEAK_BED_SNR_DB = 5.0    # bed SNR threshold below which status = "weak_bed"

# -- Resolve TIFF path ----------------------------------------------------------
if _args.tiff:
    TIFF = resolve_tiff_arg(_args.tiff, ROOT)
else:
    sys.exit("ERROR: TIFF path or --flight is required.")

TIFF = ensure_canonical_name(TIFF)

_metadata = parse_tiff_filename(TIFF)
try:
    FLT = _metadata[0] if _metadata else int(TIFF.parent.name.lstrip("Ff"))
except ValueError:
    FLT = int(os.environ.get("LYRA_FLIGHT", "0"))
if FLT <= 0:
    sys.exit("ERROR: Cannot infer flight; use FLT/TIFF_ID (e.g. 127/4800) or tiff/127/.")


TIFF_ID   = tiff_id(TIFF)
OUT_DIR   = output_root(ROOT / "output") / f"F{FLT}"
PHASE1_DIR = OUT_DIR / "phase1"
PHASE3_DIR = OUT_DIR / "phase3"
PHASE4_DIR = OUT_DIR / "phase4"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV  = PHASE1_DIR / f"F{FLT}_frame_index.csv"
CAL_CSV    = PHASE3_DIR / f"F{FLT}_cal.csv"
ECHO_CSV   = PHASE4_DIR / f"F{FLT}_echoes.csv"

# -- Validate prerequisites -----------------------------------------------------
if not INDEX_CSV.exists():
    sys.exit(f"ERROR: frame index not found at {INDEX_CSV}\n"
             "Run phase 1 (detect_frames.py) first.")

if not CAL_CSV.exists():
    sys.exit(f"ERROR: phase 3 calibration CSV not found at {CAL_CSV}\n"
             "Run phase 3 (calibrate.py) first.")

# -- Load phase1 frame index ---------------------------------------------------
index     = pd.read_csv(INDEX_CSV, dtype=str)
tiff_rows = index[index["tiff"] == TIFF.name].copy()

if len(tiff_rows) == 0:
    sys.exit(f"ERROR: {TIFF.name} not found in frame index.\n"
             "Run phase 1 (detect_frames.py) for this TIFF first.")

# -- Load phase3 calibration CSV ------------------------------------------------
cal_df    = pd.read_csv(CAL_CSV, dtype=str)
tiff_cals = cal_df[cal_df["tiff"] == TIFF.name].copy()

if len(tiff_cals) == 0:
    sys.exit(f"ERROR: No phase 3 calibration rows for {TIFF.name} in {CAL_CSV}\n"
             "Run phase 3 (calibrate.py) for this TIFF first.")

# -- Flight-wide mb_power reference for per-frame gain correction --------------
# The main bang is a fixed internal signal (T/R switch leakage) that traverses
# the full receiver chain but never leaves the aircraft.  Variation in mb_power
# across frames reflects system gain changes (attenuator setting, CRT intensity)
# rather than geophysical signal.  We use the flight-wide median as the expected
# MB power and correct each frame's echo powers accordingly.
_all_cals = cal_df.copy()
_all_cals["mb_power_dB"] = pd.to_numeric(_all_cals["mb_power_dB"], errors="coerce")
_excl_mask = _all_cals["exclude_reason"].fillna("").astype(str).str.strip().isin(["", "nan"])
_valid_mb  = _all_cals.loc[_excl_mask, "mb_power_dB"].dropna()
FLIGHT_MB_REF_DB = float(_valid_mb.median()) if len(_valid_mb) > 0 else -37.0

# Build frame_idx -> calibration dict mapping (skip excluded frames)
frame_cal: dict[int, dict] = {}
for _, row in tiff_cals.iterrows():
    # Skip excluded frames (empty mb_x means no calibration was performed)
    if row.get("exclude_reason", "") not in ("", "nan") and str(row.get("exclude_reason", "")) not in ("", "nan"):
        print(f"  Skipping frame {row['frame_idx']} ({row.get('cbd','?')}): {row['exclude_reason']}")
        continue
    if not row["mb_x"] or str(row["mb_x"]).strip() == "":
        continue
    fidx = int(row["frame_idx"])
    _mb_pwr = float(row["mb_power_dB"]) if row.get("mb_power_dB") else FLIGHT_MB_REF_DB
    frame_cal[fidx] = dict(
        mb_x        = int(float(row["mb_x"])),
        y_ref_px    = float(row["y_ref_px"]),
        db_per_px   = float(row["db_per_px"]),
        us_per_px   = float(row["us_per_px"]),
        cbd         = row["cbd"],
        file_id     = row["file_id"],
        tiff_id     = row.get("tiff_id", TIFF_ID),
        mb_power_dB = _mb_pwr,
        mb_correction_dB = round(FLIGHT_MB_REF_DB - _mb_pwr, 2),
    )

print(f"\nLYRA Step 3 — {TIFF.name}")
print(f"  Flight : F{FLT}  |  tiff_id : {TIFF_ID}")
print(f"  Calibrated frames in step2 CSV : {len(frame_cal)}")
# Show mb_power correction info
_this_mb = [c["mb_power_dB"] for c in frame_cal.values()]
_this_corr = [c["mb_correction_dB"] for c in frame_cal.values()]
_max_abs_corr = max(abs(c) for c in _this_corr) if _this_corr else 0
print(f"  MB power ref  : {FLIGHT_MB_REF_DB:.1f} dB (flight median)")
if _max_abs_corr > 0.5:
    print(f"  MB correction : this TIFF range [{min(_this_corr):+.1f}, {max(_this_corr):+.1f}] dB")
else:
    print(f"  MB correction : < 0.5 dB (negligible)")
print()

# -- Load TIFF image ------------------------------------------------------------
print("Loading TIFF ...")
Image.MAX_IMAGE_PIXELS = None
img      = np.array(Image.open(TIFF), dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
H, W_img = img_norm.shape
print(f"  Image  : {W_img} × {H} px\n")

# -- Build per-frame cal dict from step2 values + DEFAULT_CAL detection params -
def _make_cal(fc: dict) -> dict:
    """Merge per-frame step2 calibration values into DEFAULT_CAL."""
    cal = DEFAULT_CAL.copy()
    cal["mb_x"]     = fc["mb_x"]     # already in frame, but passed separately
    cal["y_ref_px"] = fc["y_ref_px"]
    cal["db_per_px"]= fc["db_per_px"]
    cal["us_per_px"]= fc["us_per_px"]
    # y_ref_db stays at -60 dB (reference is always picked at -60 dB line)
    return cal


# -- Echo overrides ------------------------------------------------------------
OVERRIDES_JSON = PHASE4_DIR / f"F{FLT}_echo_overrides.json"


def load_echo_overrides() -> dict:
    """Load manual echo overrides from JSON. Keys are file_id (e.g. 'CBD0465')."""
    if OVERRIDES_JSON.exists():
        with open(OVERRIDES_JSON) as f:
            return json.load(f)
    return {}


def save_echo_overrides(overrides: dict) -> None:
    """Write echo overrides to JSON."""
    with open(OVERRIDES_JSON, "w") as f:
        json.dump(overrides, f, indent=2)
    print(f"  Overrides -> {display_path(OVERRIDES_JSON, ROOT)}  ({len(overrides)} frames)")


echo_overrides = load_echo_overrides()
if echo_overrides:
    print(f"  Echo overrides loaded: {len(echo_overrides)} frames  ({OVERRIDES_JSON.name})")


# ==============================================================================
# Review mode — interactive echo override picker
# ==============================================================================

def review_mode():
    """Interactive review: click on image panel to override surface/bed echo positions.

    Shows ALL calibrated frames in the TIFF. Users press [s] for surface mode,
    [b] for bed mode, then click on the left (image) panel to place the pick
    directly on the CRT trace overlay.
    """

    # Need existing echo CSV to know current statuses
    if not ECHO_CSV.exists():
        sys.exit("ERROR: No echo CSV found. Run normal mode first, then --review.")

    echo_df = pd.read_csv(ECHO_CSV)
    tiff_echos = echo_df[echo_df["tiff"] == TIFF.name].copy()

    if len(tiff_echos) == 0:
        sys.exit(f"ERROR: No echo rows for {TIFF.name}. Run normal mode first.")

    # Build frame queue — default: only bad frames + frames with overrides
    # Use --all to include good frames too
    BAD_STATUSES = {"no_bed", "no_surface", "weak_bed"}
    show_all = getattr(_args, "all", False)

    review_frames = []
    all_status_counts = {}

    for _, idx_row in tiff_rows.iterrows():
        frame_idx = int(idx_row["frame_idx"])
        if idx_row["frame_type"] == "partial":
            continue
        if frame_idx not in frame_cal:
            continue

        fc = frame_cal[frame_idx]
        fkey = fc["file_id"]

        erow = tiff_echos[tiff_echos["file_id"] == fkey]
        current_status = erow.iloc[0]["echo_status"] if len(erow) > 0 else "unknown"
        all_status_counts[current_status] = all_status_counts.get(current_status, 0) + 1

        if show_all or current_status in BAD_STATUSES:
            review_frames.append((frame_idx, fc, current_status, idx_row))

    # Print TIFF summary (all frames, not just filtered)
    total = sum(all_status_counts.values())
    print(f"\nTIFF {TIFF_ID}: {total} frames total")
    for s_name in ["good", "weak_bed", "no_bed", "no_surface"]:
        if s_name in all_status_counts:
            print(f"    {s_name:12s}: {all_status_counts[s_name]}")

    if not review_frames:
        print("No frames to review (all good). Use --all to review every frame.")
        return

    filter_label = "all frames" if show_all else "bad + overridden frames only"
    print(f"\nReview: {len(review_frames)} frames ({filter_label})")
    print(f"  Click on LEFT panel (image) to place pick on CRT trace")
    print(f"  Keys: [s]=surface  [b]=bed  click=set position")
    print(f"        [S]=suppress surface  [B]=suppress bed  (mark as not present)")
    print(f"        [d]=clear overrides  [n/Right]=next  [p/Left]=prev  [q]=save & quit\n")

    frame_i = 0
    while 0 <= frame_i < len(review_frames):
        frame_idx, fc, current_status, idx_row = review_frames[frame_i]
        fkey = fc["file_id"]
        cbd = fc["cbd"]
        mb_x = fc["mb_x"]

        left_px = int(idx_row["left_px"])
        right_px = int(idx_row["right_px"])
        frame = img_norm[:, left_px:right_px + 1]
        frame_w = frame.shape[1]
        cal = _make_cal(fc)

        # Extract trace (fast, pure numpy)
        trace_y, trace_y_s, _ = extract_trace(frame, cal, robust=True, mb_x=mb_x)
        noise_floor_dB = estimate_noise_floor(trace_y_s, mb_x, cal)

        # Algorithmic detection (for reference)
        algo_surface_x, algo_bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

        # Current override (mutable copy)
        override = dict(echo_overrides.get(fkey, {}))

        # Mutable state for event handlers
        st = {"mode": "b", "override": override, "action": None, "drawn": []}

        us_per_px = cal["us_per_px"]

        # -- Build interactive figure ------------------------------------------
        fig = plt.figure(figsize=(18, 6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, wspace=0.08)
        ax_img = fig.add_subplot(gs[0, :2])
        ax_wave = fig.add_subplot(gs[0, 2])

        # -- Left panel: frame image (click target) ---------------------------
        ax_img.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      extent=[0, frame_w, frame.shape[0], 0])
        ax_img.axhline(cal["y_ref_px"], color="cyan", lw=1.2, ls="--", alpha=0.8)
        ax_img.axvline(mb_x, color="red", lw=1.5, ls="--", alpha=0.9)
        x_all = np.arange(frame_w)
        ax_img.plot(x_all, trace_y_s, color="magenta", lw=0.8, alpha=0.7, label="trace")

        # Algo markers (faded reference)
        if algo_surface_x is not None:
            algo_srf_y = trace_y_s[algo_surface_x] if algo_surface_x < len(trace_y_s) else 0
            ax_img.plot(algo_surface_x, algo_srf_y, marker="^", color="lime",
                        markersize=7, alpha=0.35, zorder=4, label="algo srf")
            ax_img.axvline(algo_surface_x, color="lime", lw=0.5, ls=":", alpha=0.35)
        if algo_bed_x is not None:
            algo_bed_y = trace_y_s[algo_bed_x] if algo_bed_x < len(trace_y_s) else 0
            ax_img.plot(algo_bed_x, algo_bed_y, marker="^", color="orange",
                        markersize=7, alpha=0.35, zorder=4, label="algo bed")
            ax_img.axvline(algo_bed_x, color="orange", lw=0.5, ls=":", alpha=0.35)

        ax_img.set_ylim(DEFAULT_CAL["y_disp_hi"] + 50, DEFAULT_CAL["y_disp_lo"] - 50)
        ax_img.set_xlim(0, frame_w)
        ax_img.set_xlabel("Column (px) — click here to place pick", fontsize=8)
        ax_img.set_ylabel("Row (px)", fontsize=8)
        ax_img.spines["top"].set_visible(False)
        ax_img.spines["right"].set_visible(False)
        ax_img.legend(fontsize=7, loc="upper right", framealpha=0.6,
                      facecolor="black", labelcolor="white")

        # -- Right panel: waveform (reference only) ----------------------------
        twt_arr = px_to_us(np.arange(frame_w, dtype=float), mb_x, cal)
        power_arr = px_to_db(trace_y_s, cal)
        t_lo = float(cal["mb_skip_us"])
        t_hi = 25.0
        wf_mask = (twt_arr >= t_lo) & (twt_arr <= t_hi)

        ax_wave.plot(twt_arr[wf_mask], power_arr[wf_mask], color="black", lw=1.2)
        ax_wave.axhline(noise_floor_dB, color="gray", lw=1.0, ls="--", alpha=0.8,
                        label=f"NF {noise_floor_dB:.1f} dB")
        ax_wave.axhline(noise_floor_dB + 5.0, color="silver", lw=0.7, ls=":", alpha=0.8)
        ax_wave.axhline(noise_floor_dB + 10.0, color="darkgray", lw=0.7, ls=":", alpha=0.8)

        # Algo surface/bed on waveform (faded)
        if algo_surface_x is not None:
            algo_srf_twt = px_to_us(float(algo_surface_x), mb_x, cal)
            algo_srf_pwr = px_to_db(trace_y_s[algo_surface_x], cal)
            ax_wave.plot(algo_srf_twt, algo_srf_pwr, marker="^", color="limegreen",
                         markersize=7, alpha=0.35, zorder=4)
            ax_wave.axvline(algo_srf_twt, color="limegreen", lw=0.5, ls=":", alpha=0.35)
        if algo_bed_x is not None:
            algo_bed_twt = px_to_us(float(algo_bed_x), mb_x, cal)
            algo_bed_pwr = px_to_db(trace_y_s[algo_bed_x], cal)
            ax_wave.plot(algo_bed_twt, algo_bed_pwr, marker="^", color="darkorange",
                         markersize=7, alpha=0.35, zorder=4)
            ax_wave.axvline(algo_bed_twt, color="darkorange", lw=0.5, ls=":", alpha=0.35)

        p_floor = noise_floor_dB - 5.0
        p_ceil = noise_floor_dB + 50.0
        ax_wave.set_ylim(p_floor, p_ceil)
        ax_wave.set_xlim(t_lo, t_hi)
        ax_wave.set_xlabel("TWT from MB (µs)", fontsize=8)
        ax_wave.set_ylabel("Power (dB)", fontsize=8)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.tick_params(labelsize=7)
        ax_wave.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

        # -- Title + status bar ------------------------------------------------
        mode_label = "BED" if st["mode"] == "b" else "SURFACE"
        status_color = {"good": "green", "weak_bed": "goldenrod",
                        "no_bed": "tomato", "no_surface": "red"}.get(current_status, "gray")
        title_obj = fig.suptitle(
            f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
            f"({frame_i + 1}/{len(review_frames)})  "
            f"mode=[{mode_label}]",
            fontsize=10, color=status_color,
        )

        status_bar = fig.text(
            0.5, 0.005, "", ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.85),
        )

        def _update_status():
            m = st["mode"]
            ov = st["override"]
            parts = [f"Mode: {'SURFACE' if m == 's' else 'BED'}"]
            if ov.get("surface_suppress"):
                parts.append("Srf: SUPPRESSED")
            elif "surface_twt_us" in ov:
                parts.append(f"Srf: {ov['surface_twt_us']:.2f} us")
            if ov.get("bed_suppress"):
                parts.append("Bed: SUPPRESSED")
            elif "bed_twt_us" in ov:
                parts.append(f"Bed: {ov['bed_twt_us']:.2f} us")
            if not any(k in ov for k in ("surface_twt_us", "bed_twt_us",
                                          "surface_suppress", "bed_suppress")):
                parts.append("No overrides")
            status_bar.set_text("  |  ".join(parts))
            col = "magenta" if m == "s" else "deepskyblue"
            status_bar.set_color(col)

        def _redraw_overrides():
            for art in st["drawn"]:
                try:
                    art.remove()
                except Exception:
                    pass
            st["drawn"].clear()

            ov = st["override"]

            # Surface suppress: show X at algo surface position
            if ov.get("surface_suppress"):
                if algo_surface_x is not None and 0 <= algo_surface_x < frame_w:
                    y_px = trace_y_s[algo_surface_x]
                    st["drawn"].append(
                        ax_img.plot(algo_surface_x, y_px, marker="X", color="magenta",
                                    markersize=12, zorder=6, alpha=0.8,
                                    markeredgecolor="white", markeredgewidth=1.0)[0])
                # Label in top-left corner of image panel
                st["drawn"].append(
                    ax_img.text(0.02, 0.96, "SRF SUPPRESSED", transform=ax_img.transAxes,
                                fontsize=9, color="magenta", fontweight="bold",
                                va="top", ha="left",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="magenta", alpha=0.85)))
            elif "surface_twt_us" in ov:
                twt = ov["surface_twt_us"]
                x_px = mb_x + int(round(twt / us_per_px))
                if 0 <= x_px < frame_w:
                    # Use clicked y if available, else fall back to smoothed trace
                    y_px = ov.get("surface_y_px", trace_y_s[x_px])
                    pwr = -(y_px - cal["y_ref_px"]) * cal["db_per_px"]
                    # Diamond on image panel (at clicked position)
                    st["drawn"].append(
                        ax_img.plot(x_px, y_px, marker="D", color="magenta",
                                    markersize=7, zorder=6, alpha=0.6,
                                    markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_img.axvline(x_px, color="magenta", lw=0.8, alpha=0.4))
                    # Mirror on waveform panel
                    st["drawn"].append(
                        ax_wave.plot(twt, pwr, marker="D", color="magenta",
                                     markersize=7, zorder=6, alpha=0.6,
                                     markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_wave.axvline(twt, color="magenta", lw=0.8, alpha=0.4))

            # Bed suppress: show X at algo bed position
            if ov.get("bed_suppress"):
                if algo_bed_x is not None and 0 <= algo_bed_x < frame_w:
                    y_px = trace_y_s[algo_bed_x]
                    st["drawn"].append(
                        ax_img.plot(algo_bed_x, y_px, marker="X", color="cyan",
                                    markersize=12, zorder=6, alpha=0.8,
                                    markeredgecolor="white", markeredgewidth=1.0)[0])
                st["drawn"].append(
                    ax_img.text(0.02, 0.90, "BED SUPPRESSED", transform=ax_img.transAxes,
                                fontsize=9, color="cyan", fontweight="bold",
                                va="top", ha="left",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="cyan", alpha=0.85)))
            elif "bed_twt_us" in ov:
                twt = ov["bed_twt_us"]
                x_px = mb_x + int(round(twt / us_per_px))
                if 0 <= x_px < frame_w:
                    # Use clicked y if available, else fall back to smoothed trace
                    y_px = ov.get("bed_y_px", trace_y_s[x_px])
                    pwr = -(y_px - cal["y_ref_px"]) * cal["db_per_px"]
                    # Diamond on image panel (at clicked position)
                    st["drawn"].append(
                        ax_img.plot(x_px, y_px, marker="D", color="cyan",
                                    markersize=7, zorder=6, alpha=0.6,
                                    markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_img.axvline(x_px, color="cyan", lw=0.8, alpha=0.4))
                    # Mirror on waveform panel
                    st["drawn"].append(
                        ax_wave.plot(twt, pwr, marker="D", color="cyan",
                                     markersize=7, zorder=6, alpha=0.6,
                                     markeredgecolor="white", markeredgewidth=0.5)[0])
                    st["drawn"].append(
                        ax_wave.axvline(twt, color="cyan", lw=0.8, alpha=0.4))

            _update_status()
            fig.canvas.draw_idle()

        def on_click(event):
            # Click on image panel -> convert pixel x to TWT, capture y for display
            if event.inaxes != ax_img or event.button != 1:
                return
            x_px = event.xdata
            y_px = event.ydata
            if x_px is None or x_px < mb_x:
                return
            x_px = int(round(x_px))
            if x_px >= frame_w:
                return
            twt = (x_px - mb_x) * us_per_px
            if st["mode"] == "s":
                st["override"].pop("surface_suppress", None)
                st["override"]["surface_twt_us"] = round(twt, 3)
                st["override"]["surface_y_px"] = round(float(y_px), 1)
            elif st["mode"] == "b":
                st["override"].pop("bed_suppress", None)
                st["override"]["bed_twt_us"] = round(twt, 3)
                st["override"]["bed_y_px"] = round(float(y_px), 1)
            _redraw_overrides()

        def on_key(event):
            k = event.key
            if k == "s":
                st["mode"] = "s"
                title_obj.set_text(
                    f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
                    f"({frame_i + 1}/{len(review_frames)})  mode=[SURFACE]")
                _update_status()
                fig.canvas.draw_idle()
            elif k == "b":
                st["mode"] = "b"
                title_obj.set_text(
                    f"REVIEW  F{FLT} {fkey}  [{current_status}]  "
                    f"({frame_i + 1}/{len(review_frames)})  mode=[BED]")
                _update_status()
                fig.canvas.draw_idle()
            elif k == "S":   # Shift+S: suppress surface (mark as not present)
                # Remove any surface position override, set suppress flag
                st["override"].pop("surface_twt_us", None)
                st["override"].pop("surface_y_px", None)
                st["override"]["surface_suppress"] = True
                _redraw_overrides()
            elif k == "B":   # Shift+B: suppress bed (mark as not present)
                st["override"].pop("bed_twt_us", None)
                st["override"].pop("bed_y_px", None)
                st["override"]["bed_suppress"] = True
                _redraw_overrides()
            elif k == "d":
                st["override"].clear()
                _redraw_overrides()
            elif k in ("n", "right"):
                st["action"] = "next"
                plt.close(fig)
            elif k in ("p", "left"):
                st["action"] = "prev"
                plt.close(fig)
            elif k in ("q", "escape"):
                st["action"] = "quit"
                plt.close(fig)

        # Unbind matplotlib defaults that clash with review keys
        for action in ("keymap.save", "keymap.quit", "keymap.back",
                        "keymap.forward", "keymap.fullscreen"):
            try:
                plt.rcParams[action] = []
            except KeyError:
                pass
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        _redraw_overrides()
        plt.show()

        # Save override for this frame
        if st["override"]:
            echo_overrides[fkey] = st["override"]
        elif fkey in echo_overrides:
            del echo_overrides[fkey]

        # Navigate
        if st["action"] == "next" or st["action"] is None:
            frame_i += 1
        elif st["action"] == "prev":
            frame_i = max(0, frame_i - 1)
        elif st["action"] == "quit":
            break

    # Persist overrides
    save_echo_overrides(echo_overrides)
    n_ov = len(echo_overrides)
    print(f"\nReview complete. {n_ov} override{'s' if n_ov != 1 else ''} saved.")


# ==============================================================================
# Branch: review mode vs normal extraction
# ==============================================================================

if _args.review:
    review_mode()
    print("\n" + "=" * 70)
    print("Re-running normal mode to apply overrides...")
    print("=" * 70 + "\n")
    # Reload overrides that review_mode just saved to disk
    echo_overrides = load_echo_overrides()
    # Fall through to normal extraction below


# -- Per-frame echo extraction --------------------------------------------------
print(f"  {'Fr':>3}  {'CBD':>6}  {'Status':^12}  {'NF':>6}  "
      f"{'Srf TWT':>8}  {'Srf dB':>7}  "
      f"{'Bed TWT':>8}  {'Bed dB':>7}  {'Bed SNR':>7}  "
      f"{'h_air':>7}  {'h_ice':>7}")
print("  " + "-" * 90)

echo_rows = []

for _, idx_row in tiff_rows.iterrows():
    frame_idx  = int(idx_row["frame_idx"])
    frame_type = idx_row["frame_type"]

    if frame_type == "partial":
        continue
    if frame_idx not in frame_cal:
        # Frame was excluded in step2 (or step2 hasn't been run for this frame)
        continue

    fc      = frame_cal[frame_idx]
    cbd     = fc["cbd"]
    file_id = fc["file_id"]
    mb_x    = fc["mb_x"]

    left_px  = int(idx_row["left_px"])
    right_px = int(idx_row["right_px"])
    frame    = img_norm[:, left_px : right_px + 1]
    frame_w  = frame.shape[1]

    cal = _make_cal(fc)

    # -- Extract trace and noise floor ------------------------------------------
    # Use robust mode: constrained-argmin rejects film-grain artefacts and
    # T/R ringing noise that would otherwise create fake peaks in the flat
    # region between the main bang and the first real echo.
    trace_y, trace_y_s, trace_info = extract_trace(frame, cal, robust=True, mb_x=mb_x)
    noise_floor_dB     = estimate_noise_floor(trace_y_s, mb_x, cal)

    # -- Detect surface and bed echoes ------------------------------------------
    surface_x, bed_x = detect_echoes(trace_y_s, mb_x, noise_floor_dB, cal)

    # -- Save algorithmic positions before overrides ----------------------------
    algo_surface_x = surface_x
    algo_bed_x     = bed_x

    # -- Apply manual overrides (from echo_overrides.json) --------------------
    is_surface_override = is_bed_override = False
    is_surface_suppress = is_bed_suppress = False
    override = echo_overrides.get(file_id, {})
    us_per_px = cal["us_per_px"]
    if override.get("surface_suppress"):
        surface_x = None
        is_surface_suppress = True
        is_surface_override = True
    elif "surface_twt_us" in override:
        surface_x = mb_x + int(round(override["surface_twt_us"] / us_per_px))
        surface_x = max(0, min(surface_x, frame_w - 1))
        is_surface_override = True
    if override.get("bed_suppress"):
        bed_x = None
        is_bed_suppress = True
        is_bed_override = True
    elif "bed_twt_us" in override:
        bed_x = mb_x + int(round(override["bed_twt_us"] / us_per_px))
        bed_x = max(0, min(bed_x, frame_w - 1))
        is_bed_override = True

    # -- Compute echo metrics ---------------------------------------------------
    # Use local column profile extraction: immune to global trace failures
    # (graticule lines, reference lines competing with CRT signal).
    # Falls back to compute_echo_metrics only if local profile fails.
    surface = extract_local_profile(
        frame, surface_x, mb_x, noise_floor_dB, cal,
        guide_y=float(trace_y_s[surface_x]) if surface_x < len(trace_y_s) else None,
        max_half_width_us=2.0,   # surface echo physically < 2 µs half-width
    ) if surface_x is not None else None

    bed = extract_local_profile(
        frame, bed_x, mb_x, noise_floor_dB, cal,
        guide_y=float(trace_y_s[bed_x]) if bed_x < len(trace_y_s) else None,
        max_half_width_us=4.0,   # bed echo physically < 4 µs half-width
    ) if bed_x is not None else None

    # -- Classify echo_status ---------------------------------------------------
    # Use gain-corrected bed SNR for the weak_bed threshold.  The attenuator
    # sits before the receiver, so the noise floor is constant but the signal
    # level shifts with attenuator/CRT gain changes.  Raw SNR is therefore
    # inflated (or deflated) by the gain offset; corrected SNR reflects the
    # true signal-to-noise ratio.
    _mb_corr = fc["mb_correction_dB"]
    if surface is None:
        echo_status = "no_surface"
    elif bed is None:
        echo_status = "no_bed"
    elif (bed.peak_snr_dB + _mb_corr) < WEAK_BED_SNR_DB:
        echo_status = "weak_bed"
    else:
        echo_status = "good"

    # -- Original algorithmic echo_status (before overrides) -----------------
    # Preserved for post-hoc threshold tuning: what did the algorithm find
    # before the user corrected it?  Only differs from echo_status when an
    # override was applied.
    if not (is_surface_override or is_bed_override):
        algo_echo_status = echo_status
    elif algo_surface_x is None:
        algo_echo_status = "no_surface"
    elif algo_bed_x is None:
        algo_echo_status = "no_bed"
    else:
        _algo_bed = extract_local_profile(
            frame, algo_bed_x, mb_x, noise_floor_dB, cal,
            guide_y=float(trace_y_s[algo_bed_x]) if algo_bed_x < len(trace_y_s) else None)
        algo_echo_status = ("weak_bed"
                            if (_algo_bed.peak_snr_dB + _mb_corr) < WEAK_BED_SNR_DB
                            else "good")

    # -- Bed envelope artifact flag -------------------------------------------
    # Check whether film artifacts exist in the post-bed region (between
    # the bed peak and the NF+5 trailing edge + 1 µs margin).  Artifacts
    # here bias the trailing edge, inflating width_5, asymmetry, and
    # trailing_tail.  Detection: compare constrained trace vs median-
    # filtered trace (art_diff_px from extract_trace).  Columns where the
    # median filter removed a >10 dB dip in this region are flagged.
    #
    # Artifacts elsewhere (pre-MB, surface-bed gap, pre-surface) are
    # benign for bed envelope estimation and are not flagged.
    bed_envelope_suspect = False
    art_max_dB = 0.0
    art_diff_px = trace_info.get("art_diff_px")
    if bed is not None and art_diff_px is not None:
        bed_trail_5 = getattr(bed, "trail_5_us", np.nan)
        if not np.isnan(bed_trail_5):
            # Check region: bed_peak to trail_5 + 1 µs
            us_per_px = cal["us_per_px"]
            db_per_px = cal["db_per_px"]
            post_bed_start = bed.peak_x
            post_bed_end   = min(len(art_diff_px),
                                 mb_x + int((bed_trail_5 + 1.0) / us_per_px))
            if post_bed_start < post_bed_end:
                region = art_diff_px[post_bed_start:post_bed_end]
                # >10 dB discrepancy = significant artifact
                art_thresh_px = 10.0 / db_per_px
                art_mask = region > art_thresh_px
                if np.any(art_mask):
                    art_max_dB = float(np.max(region[art_mask]) * db_per_px)
                    bed_envelope_suspect = True

    # -- Derived geometry -------------------------------------------------------
    h_air_m = h_ice_m = h_eff_m = np.nan
    h_ice_onset_m = np.nan
    if surface is not None:
        h_air_m = surface.peak_twt_us / 2.0 * C_AIR_M_PER_US
    if surface is not None and bed is not None:
        h_ice_m = (bed.peak_twt_us - surface.peak_twt_us) / 2.0 * C_ICE_M_PER_US
        h_eff_m = h_air_m + h_ice_m / N_ICE
        # Onset-based ice thickness: uses first-break TWT for both surface and
        # bed when available, falling back to peak TWT when onset is NaN.
        srf_onset = surface.onset_twt_us if not np.isnan(surface.onset_twt_us) else surface.peak_twt_us
        bed_onset = bed.onset_twt_us if not np.isnan(bed.onset_twt_us) else bed.peak_twt_us
        h_ice_onset_m = (bed_onset - srf_onset) / 2.0 * C_ICE_M_PER_US

    # -- Print summary line -----------------------------------------------------
    srf_twt = f"{surface.peak_twt_us:8.3f}" if surface else f"{'—':>8}"
    srf_db  = f"{surface.peak_power_dB:+7.1f}" if surface else f"{'—':>7}"
    bed_twt = f"{bed.peak_twt_us:8.3f}" if bed else f"{'—':>8}"
    bed_db  = f"{bed.peak_power_dB:+7.1f}" if bed else f"{'—':>7}"
    bed_snr = f"{bed.peak_snr_dB:+7.1f}" if bed else f"{'—':>7}"
    h_air_s = f"{h_air_m:7.0f}" if not np.isnan(h_air_m) else f"{'—':>7}"
    h_ice_s = f"{h_ice_m:7.0f}" if not np.isnan(h_ice_m) else f"{'—':>7}"

    ov_tag = ""
    if is_surface_override or is_bed_override:
        ov_tag = "  [M]"
    print(f"  {frame_idx:>3}  {str(cbd):>6}  {echo_status:^12}  "
          f"{noise_floor_dB:+6.1f}  "
          f"{srf_twt}  {srf_db}  "
          f"{bed_twt}  {bed_db}  {bed_snr}  "
          f"{h_air_s}  {h_ice_s}{ov_tag}")

    # -- Build output row -------------------------------------------------------
    def _e(echo, attr):
        return getattr(echo, attr, np.nan) if echo is not None else np.nan

    # Per-frame MB power correction: normalise echo powers to the flight-wide
    # reference MB level.  Removes system gain variation (attenuator changes,
    # CRT intensity drift) while preserving real geophysical power differences.
    # The MB is a fixed internal signal (T/R switch leakage) — any change in
    # its measured power is purely instrumental.
    _mb_corr = fc["mb_correction_dB"]  # positive = frame was weaker -> boost

    def _corrected(raw_dB):
        """Apply MB power correction to a raw dB measurement."""
        if np.isnan(raw_dB):
            return np.nan
        return round(raw_dB + _mb_corr, 2)

    echo_rows.append(dict(
        flight      = FLT,
        tiff        = TIFF.name,
        tiff_id     = TIFF_ID,
        frame_idx   = frame_idx,
        cbd         = cbd,
        file_id     = file_id,
        mb_x        = mb_x,
        echo_status = echo_status,
        noise_floor_dB = round(noise_floor_dB, 2),
        # MB power correction
        mb_power_dB       = round(fc["mb_power_dB"], 2),
        mb_correction_dB  = round(_mb_corr, 2),
        # Surface echo (power and SNR corrected for system gain)
        surface_twt_us        = round(_e(surface, "peak_twt_us"),     4),
        surface_power_dB      = _corrected(_e(surface, "peak_power_dB")),
        surface_snr_dB        = _corrected(_e(surface, "peak_snr_dB")),
        surface_width_10_us   = round(_e(surface, "width_10_us"),     4),
        surface_width_5_us    = round(_e(surface, "width_5_us"),      4),
        surface_peakiness     = round(_e(surface, "peakiness"),       3),
        surface_asymmetry     = round(_e(surface, "asymmetry"),       3),
        surface_leading_rise_us  = round(_e(surface, "leading_rise_us"),  4),
        surface_trailing_tail_us = round(_e(surface, "trailing_tail_us"), 4),
        surface_onset_twt_us     = round(_e(surface, "onset_twt_us"),    4),
        # Bed echo (power and SNR corrected for system gain)
        bed_twt_us            = round(_e(bed, "peak_twt_us"),    4),
        bed_power_dB          = _corrected(_e(bed, "peak_power_dB")),
        bed_snr_dB            = _corrected(_e(bed, "peak_snr_dB")),
        bed_width_10_us       = round(_e(bed, "width_10_us"),    4),
        bed_width_5_us        = round(_e(bed, "width_5_us"),     4),
        bed_peakiness         = round(_e(bed, "peakiness"),      3),
        bed_asymmetry         = round(_e(bed, "asymmetry"),      3),
        bed_leading_rise_us   = round(_e(bed, "leading_rise_us"),  4),
        bed_trailing_tail_us  = round(_e(bed, "trailing_tail_us"), 4),
        bed_onset_twt_us      = round(_e(bed, "onset_twt_us"),    4),
        # Peak-relative shape metrics (SNR-independent)
        bed_decay_10_us       = round(_e(bed, "decay_10_us"),     4),
        bed_decay_20_us       = round(_e(bed, "decay_20_us"),     4),
        bed_rise_10_us        = round(_e(bed, "rise_10_us"),      4),
        bed_pk_asymmetry      = round(_e(bed, "pk_asymmetry"),    3),
        # Derived: peak-onset delay (Fresnel-zone roughness diagnostic)
        bed_peak_onset_us     = round(
            _e(bed, "peak_twt_us") - _e(bed, "onset_twt_us"), 4),
        bed_envelope_suspect  = bed_envelope_suspect,
        artifact_max_dB       = round(art_max_dB, 1),
        # Geometry (unaffected by gain correction)
        h_air_m = round(h_air_m, 1) if not np.isnan(h_air_m) else np.nan,
        h_ice_m = round(h_ice_m, 1) if not np.isnan(h_ice_m) else np.nan,
        h_ice_onset_m = round(h_ice_onset_m, 1) if not np.isnan(h_ice_onset_m) else np.nan,
        h_eff_m = round(h_eff_m, 1) if not np.isnan(h_eff_m) else np.nan,
        # Override flags
        surface_override = is_surface_override,
        bed_override     = is_bed_override,
        algo_echo_status = algo_echo_status,
        reflector_type   = "bed",
    ))

    # -- Generate single-panel diagnostic figure ---------------------------------
    # Full-width A-scope image with dual axes (px + physical units),
    # echo markers, and semi-transparent envelope polygons.
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    fw = frame.shape[1]
    y_ref_px = cal["y_ref_px"]
    y_lo_disp = DEFAULT_CAL["y_disp_lo"] - 50
    y_hi_disp = DEFAULT_CAL["y_disp_hi"] + 50

    ax.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
              extent=[0, fw, frame.shape[0], 0])

    # Noise floor reference row (cyan dashed)
    ax.axhline(y_ref_px, color="cyan", lw=1.0, ls="--", alpha=0.6)

    # Main bang (red dashed vertical)
    ax.axvline(mb_x, color="red", lw=1.2, ls="--", alpha=0.8)

    # -- Envelope polygons (semi-transparent, on CRT image) --------------------
    def _draw_envelope(ax_obj, echo, echo_x, color):
        """Draw semi-transparent envelope polygon for an echo."""
        if echo is None:
            return
        # +10 dB envelope
        if not np.isnan(echo.lead_10_us) and not np.isnan(echo.trail_10_us):
            l10_px = us_to_px(echo.lead_10_us, mb_x, cal)
            t10_px = us_to_px(echo.trail_10_us, mb_x, cal)
            thresh_10_y = db_to_px(noise_floor_dB + 10.0, cal)
            # Polygon: from threshold row at leading edge, up to peak, back to
            # threshold at trailing edge
            xs = [l10_px, echo_x, t10_px, t10_px, l10_px]
            ys = [thresh_10_y, echo.peak_y, thresh_10_y, thresh_10_y, thresh_10_y]
            ax_obj.fill(xs[:3], ys[:3], color=color, alpha=0.15, zorder=3)
        # +5 dB envelope (wider, fainter)
        if not np.isnan(echo.lead_5_us) and not np.isnan(echo.trail_5_us):
            l5_px = us_to_px(echo.lead_5_us, mb_x, cal)
            t5_px = us_to_px(echo.trail_5_us, mb_x, cal)
            thresh_5_y = db_to_px(noise_floor_dB + 5.0, cal)
            xs5 = [l5_px, echo_x, t5_px]
            ys5 = [thresh_5_y, echo.peak_y, thresh_5_y]
            ax_obj.fill(xs5, ys5, color=color, alpha=0.08, zorder=2)

    _draw_envelope(ax, surface, surface_x if surface_x is not None else 0,
                   "limegreen")
    _draw_envelope(ax, bed, bed_x if bed_x is not None else 0,
                   "darkorange")

    # -- Echo markers ----------------------------------------------------------
    # Surface
    if is_surface_suppress:
        if algo_surface_x is not None:
            ax.plot(algo_surface_x, y_ref_px, marker="X",
                    color="lime", markersize=12, zorder=5, alpha=0.6,
                    markeredgecolor="white", markeredgewidth=1.0)
    elif surface is not None:
        srf_marker = "D" if is_surface_override else "^"
        srf_label  = (f"Srf {surface.peak_twt_us:.2f}us  "
                      f"{surface.peak_power_dB:+.1f}dB"
                      + (" [M]" if is_surface_override else ""))
        ax.plot(surface_x, surface.peak_y, marker=srf_marker, color="lime",
                markersize=10, zorder=6, label=srf_label,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(surface_x, color="lime", lw=0.6, ls=":", alpha=0.5)

    # Bed
    if is_bed_suppress:
        if algo_bed_x is not None:
            ax.plot(algo_bed_x, y_ref_px, marker="X",
                    color="orange", markersize=12, zorder=5, alpha=0.6,
                    markeredgecolor="white", markeredgewidth=1.0)
    elif bed is not None:
        bed_marker = "D" if is_bed_override else "^"
        bed_label  = (f"Bed {bed.peak_twt_us:.2f}us  "
                      f"{bed.peak_power_dB:+.1f}dB  "
                      f"SNR={bed.peak_snr_dB:+.1f}dB"
                      + (" [M]" if is_bed_override else ""))
        ax.plot(bed_x, bed.peak_y, marker=bed_marker, color="orange",
                markersize=10, zorder=6, label=bed_label,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(bed_x, color="orange", lw=0.6, ls=":", alpha=0.5)

    # -- Primary axes: pixel coordinates ---------------------------------------
    ax.set_ylim(y_hi_disp, y_lo_disp)
    ax.set_xlim(0, fw)
    ax.set_xlabel("Column (px, frame-relative)", fontsize=8)
    ax.set_ylabel("Row (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -- Secondary X-axis: TWT from MB (us) ------------------------------------
    ax2_x = ax.twiny()
    ax2_x.set_xlim((0 - mb_x) * cal["us_per_px"],
                    (fw - mb_x) * cal["us_per_px"])
    ax2_x.set_xlabel("Two-way travel time from MB (us)", fontsize=8)
    ax2_x.tick_params(labelsize=7)
    ax2_x.spines["top"].set_linewidth(0.4)

    # -- Secondary Y-axis: Power (dB) -----------------------------------------
    ax2_y = ax.twinx()
    db_top    = px_to_db(y_lo_disp, cal)
    db_bottom = px_to_db(y_hi_disp, cal)
    ax2_y.set_ylim(db_bottom, db_top)
    ax2_y.set_ylabel("Power (dB)", fontsize=8)
    ax2_y.tick_params(labelsize=7)
    ax2_y.spines["right"].set_linewidth(0.4)

    # -- Legend and title -------------------------------------------------------
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7,
                  facecolor="black", labelcolor="white")

    # Geometry annotation (bottom-left)
    geo_parts = []
    if not np.isnan(h_air_m):
        geo_parts.append(f"h_air={h_air_m:.0f} m")
    if not np.isnan(h_ice_m):
        geo_parts.append(f"h_ice={h_ice_m:.0f} m")
    if geo_parts:
        ax.text(0.01, 0.04, "  |  ".join(geo_parts),
                transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          edgecolor="none", alpha=0.6))

    ax.set_title(
        f"LYRA Phase 4 -- F{FLT} {file_id}  |  NF={noise_floor_dB:.1f} dB  "
        f"[{echo_status}]",
        fontsize=9, loc="left")

    fig_path = PHASE4_DIR / f"F{FLT}_{file_id}_echoes.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -- Clean calibrated waveform figure (Panel-C style) -----------------------
    # Uses the SAME calibrated trace and the SAME final surface/bed picks as the
    # diagnostic above. This is visualization only; no repicking/recalculation.
    twt_arr = px_to_us(np.arange(frame_w, dtype=float), mb_x, cal)

    # DISPLAY-ONLY full A-scope trace.
    # The robust science trace intentionally clamps the T/R ringing zone around
    # the main bang to the noise-floor reference.  That is desirable for echo
    # detection, but it removes the actual transmitter pulse from a waveform
    # visualization.  Re-extract a non-robust trace solely for this figure so
    # the red curve follows the visible CRT trace across the complete frame.
    # The science trace_y_s above remains unchanged for all picks, metrics, CSVs,
    # and thickness calculations.
    _display_trace_y, _display_trace_y_s, _display_trace_info = extract_trace(
        frame, cal, robust=False, mb_x=mb_x
    )
    power_arr = px_to_db(_display_trace_y_s, cal)

    wf_fig, wf_ax = plt.subplots(figsize=(10, 6))
    wf_fig.patch.set_facecolor("white")

    # Show the COMPLETE calibrated A-scope trace.
    # Keep all available samples before and after the detected main bang so the
    # transmitter/main-bang waveform itself is visible rather than being clipped
    # at 0 us.  Time is still referenced to mb_x, so mb_x == 0 us.
    wf_mask = np.isfinite(twt_arr) & np.isfinite(power_arr)
    if np.any(wf_mask):
        wf_ax.plot(twt_arr[wf_mask], power_arr[wf_mask], color="red", lw=1.4)

    # -60 dB calibration reference.
    wf_ax.axhline(-60.0, color="green", lw=1.0, ls=":", alpha=0.8,
                  label="Reference (-60 dB)")

    # Transmitter / main-bang time zero.
    tx_power = (float(px_to_db(_display_trace_y_s[mb_x], cal))
                if 0 <= mb_x < len(_display_trace_y_s) else np.nan)
    if np.isfinite(tx_power):
        wf_ax.plot(0.0, tx_power, "o", color="royalblue", ms=7, zorder=5, label="Tx")
        wf_ax.annotate(
            f"transmitter pulse\n(~{tx_power:.1f} dB at 0.0 us)",
            xy=(0.0, tx_power), xytext=(-0.5, tx_power - 7.0),
            textcoords="data", ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="royalblue", lw=1.0),
        )

    # Final surface pick used by Phase 4.
    if surface is not None:
        s_t = float(surface.peak_twt_us)
        s_p = float(surface.peak_power_dB)
        wf_ax.plot(s_t, s_p, "o", color="seagreen", ms=7, zorder=6, label="Surface")
        wf_ax.annotate(
            f"surface\n(~{s_p:.1f} dB at {s_t:.1f} us)",
            xy=(s_t, s_p), xytext=(s_t - 3.0, s_p - 5.0),
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="seagreen", lw=1.0),
        )

    # Final bed pick used by Phase 4.
    if bed is not None:
        b_t = float(bed.peak_twt_us)
        b_p = float(bed.peak_power_dB)
        wf_ax.plot(b_t, b_p, "o", color="mediumorchid", ms=7, zorder=6, label="Bed")
        wf_ax.annotate(
            f"bed\n(~{b_p:.1f} dB at {b_t:.1f} us)",
            xy=(b_t, b_p), xytext=(b_t + 3.0, b_p - 5.0),
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="mediumorchid", lw=1.0),
        )

    wf_ax.set_xlabel("Two-way travel time from MB (us)")
    wf_ax.set_ylabel("Power (dB)")
    wf_ax.set_title(f"LYRA Calibrated A-scope: F{FLT} {file_id}")
    wf_ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    wf_ax.minorticks_on()
    wf_ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25)
    wf_ax.legend(loc="upper right", framealpha=0.9)

    # DISPLAY CROP ONLY.  The full frame is still extracted above, but the
    # publication-style waveform omits the ugly frame-edge regions.  Keep a
    # short pre-MB interval so the transmitter pulse is visible, then continue
    # several microseconds beyond the bed to retain the useful post-bed tail.
    if np.any(wf_mask):
        _available_lo = float(np.nanmin(twt_arr[wf_mask]))
        _available_hi = float(np.nanmax(twt_arr[wf_mask]))
        _plot_lo = max(_available_lo, -2.5)
        if bed is not None and np.isfinite(float(bed.peak_twt_us)):
            _plot_hi = min(_available_hi, float(bed.peak_twt_us) + 6.0)
        elif surface is not None and np.isfinite(float(surface.peak_twt_us)):
            _plot_hi = min(_available_hi, float(surface.peak_twt_us) + 12.0)
        else:
            _plot_hi = min(_available_hi, 20.0)
        wf_ax.set_xlim(_plot_lo, _plot_hi)
    finite_power = power_arr[wf_mask]
    if finite_power.size:
        wf_ax.set_ylim(min(-65.0, float(np.nanmin(finite_power)) - 2.0),
                       max(0.0, float(np.nanmax(finite_power)) + 3.0))

    waveform_path = PHASE4_DIR / f"F{FLT}_{file_id}_waveform.png"
    wf_fig.savefig(waveform_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(wf_fig)


# -- Update echo CSV (incremental, TIFF-specific rows) -------------------------
if echo_rows:
    new_df = pd.DataFrame(echo_rows)
    if ECHO_CSV.exists():
        existing = pd.read_csv(ECHO_CSV, dtype=str)
        # Preserve non-"bed" reflector_type flags set by the user
        rt_map: dict[str, str] = {}
        if "reflector_type" in existing.columns:
            for _, r in existing[existing["tiff"] == TIFF.name].iterrows():
                rt = r.get("reflector_type", "bed")
                if pd.notna(rt) and str(rt) != "bed":
                    rt_map[str(r["file_id"])] = str(rt)
        existing = existing[existing["tiff"] != TIFF.name]
        merged   = pd.concat([existing, new_df.astype(str)], ignore_index=True)
        # Re-apply preserved flags to new rows
        if rt_map:
            for fid, rt in rt_map.items():
                merged.loc[merged["file_id"] == fid, "reflector_type"] = rt
    else:
        merged = new_df.astype(str)
    # Deduplicate overlapping TIFFs: keep highest SNR per CBD
    if "cbd" in merged.columns and "bed_snr_dB" in merged.columns:
        merged["_snr_sort"] = pd.to_numeric(merged["bed_snr_dB"], errors="coerce")
        merged = merged.sort_values("_snr_sort", ascending=False)
        n_before = len(merged)
        merged = merged.drop_duplicates(subset=["cbd"], keep="first")
        merged = merged.drop(columns=["_snr_sort"])
        n_dropped = n_before - len(merged)
        if n_dropped > 0:
            print(f"  Deduped {n_dropped} overlapping-TIFF rows (kept highest SNR per CBD)")
        merged = merged.sort_values("cbd")
    merged.to_csv(ECHO_CSV, index=False)

    # -- Summary statistics -----------------------------------------------------
    print(f"\n{'-' * 60}")
    print(f"  Echo CSV -> {display_path(ECHO_CSV, ROOT)}")
    print(f"  Phase 4 figures -> {display_path(PHASE4_DIR, ROOT)}/")
    print(f"  Frames processed : {len(echo_rows)}")

    statuses = [r["echo_status"] for r in echo_rows]
    for st in ["good", "weak_bed", "no_bed", "no_surface"]:
        n = statuses.count(st)
        if n:
            print(f"    {st:12s}: {n}")

    good_rows = [r for r in echo_rows if r["echo_status"] in ("good", "weak_bed")]
    if good_rows:
        h_airs = [r["h_air_m"] for r in good_rows
                  if not (isinstance(r["h_air_m"], float) and np.isnan(r["h_air_m"]))]
        h_ices = [r["h_ice_m"] for r in good_rows
                  if not (isinstance(r["h_ice_m"], float) and np.isnan(r["h_ice_m"]))]
        if h_airs:
            print(f"  h_air  : {np.mean(h_airs):.0f} ± {np.std(h_airs):.0f} m "
                  f"(n={len(h_airs)})")
        if h_ices:
            print(f"  h_ice  : {np.mean(h_ices):.0f} ± {np.std(h_ices):.0f} m "
                  f"(n={len(h_ices)})")

    # -- Generate review summary JSON (full flight, all TIFFs) -------------
    review_json_path = PHASE4_DIR / f"F{FLT}_echo_review.json"
    full_df = pd.read_csv(ECHO_CSV)
    review_data = {
        "flight": FLT,
        "total_frames": len(full_df),
        "summary": {},
    }
    for st_name in ["good", "weak_bed", "no_bed", "no_surface"]:
        st_df = full_df[full_df["echo_status"] == st_name]
        review_data["summary"][st_name] = len(st_df)

    # Per-status frame lists (only non-good statuses)
    for st_name in ["no_bed", "no_surface", "weak_bed"]:
        st_df = full_df[full_df["echo_status"] == st_name]
        if len(st_df) == 0:
            continue
        frames = []
        for _, row in st_df.iterrows():
            frames.append({
                "tiff_id": str(row.get("tiff_id", "")),
                "cbd": str(int(float(row["cbd"]))) if str(row["cbd"]).replace('.','',1).isdigit() else str(row["cbd"]),
                "file_id": str(row["file_id"]),
            })
        review_data[st_name] = frames

    # Per-TIFF breakdown: how many bad frames each TIFF has
    bad_df = full_df[full_df["echo_status"].isin(["no_bed", "no_surface", "weak_bed"])]
    if len(bad_df) > 0:
        tiff_breakdown = {}
        for tid, grp in bad_df.groupby("tiff_id"):
            counts = grp["echo_status"].value_counts().to_dict()
            tiff_breakdown[str(tid)] = {
                "total_bad": len(grp),
                **{k: int(v) for k, v in counts.items()},
            }
        # Sort by total_bad descending
        review_data["tiff_breakdown"] = dict(
            sorted(tiff_breakdown.items(), key=lambda x: x[1]["total_bad"], reverse=True)
        )

    with open(review_json_path, "w") as f:
        json.dump(review_data, f, indent=2)
    print(f"  Review JSON -> {display_path(review_json_path, ROOT)}")

print("\nDone.")
''',
    'pick_calibration': r'''"""
pick_calibration.py — LYRA interactive calibration picker (multi-frame)
========================================================================
For each complete frame in a TIFF, collect guidance clicks for:

  M — Main bang              (1 click  -> red   vertical  line)
  R — Reference -60 dB      (1 click  -> cyan  horizontal line)
  Y — Y-axis major grid line (2 clicks -> lime  horizontal lines)
  X — X-axis major grid line (4 clicks -> orange vertical   lines)

These clicks are GUIDANCE ONLY — approximate hints ±tens of pixels are fine.
The algorithm will snap to the true feature near each guide pick.

Keyboard shortcuts (case-insensitive):
  M  — switch to Main Bang mode
  R  — switch to Reference line -60 dB mode
  Y  — switch to Y-grid mode     (click two visible major lines)
  X  — switch to X-grid mode     (click four visible major lines)
  U  — undo last pick in the current mode
  N  — save picks for this frame and move to next frame
  B  — go back to previous frame (save current picks first)
  S  — skip this frame (no picks saved)
  Q  — save picks and advance to next TIFF (flight mode) or quit (single mode)
  Esc — quit entirely (save current picks first)

Usage
-----
Single TIFF:
    python codes/pick_calibration.py <TIFF_PATH>

All TIFFs in a flight that need picks (M+R+X):
    python codes/pick_calibration.py --flight 126

Re-pick a specific TIFF by ID:
    python codes/pick_calibration.py --flight 126 --tiff 2750

Fix only specific flagged CBDs:
    python codes/pick_calibration.py --flight 126 --tiff 2600 --cbds 301,305,312

Outputs
-------
  output/F{FLT}/phase1/F{FLT}_cal_picks.json
  output/F{FLT}/phase1/F{FLT}_cal_derived.json
"""

from pathlib import Path
import argparse
import sys
import os
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from lyra import (
    display_path, ensure_canonical_name, output_root, parse_tiff_filename,
    resolve_tiff_arg, tiff_id as lyra_tiff_id,
)

# Display crop: show full display band plus context
Y_CROP_LO = 100
Y_CROP_HI = 1950

# Default calibration constants (for drawing faint guide lines)
_DB_REF_PX    = 1507
_PX_PER_DIV_Y = 205.0
_PX_PER_DIV_X = 205.54

MODE_COLORS = {"mb": "red", "ref": "cyan", "y_grid": "limegreen", "x_grid": "orange"}
MODE_LABELS = {
    "mb":     "Main Bang  [M] — 1 click (red)",
    "ref":    "Reference -60 dB  [R] — 1 click (cyan)",
    "y_grid": "Y-grid major line  [Y] — 2 clicks (lime)",
    "x_grid": "X-grid major line  [X] — 4 clicks (orange)",
}


# -- Argument parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LYRA interactive calibration picker")
    parser.add_argument("tiff_path", nargs="?", default=None,
                        help="Path to a single TIFF file")
    parser.add_argument("--flight", type=int, default=None,
                        help="Flight number — auto-discover TIFFs needing picks")
    parser.add_argument("--tiff", type=int, default=None,
                        help="Specific TIFF ID to re-pick (requires --flight)")
    parser.add_argument("--cbds", type=str, default=None,
                        help="Comma-separated CBD numbers to show (skip others)")
    return parser.parse_args()


# -- TIFF queue logic ----------------------------------------------------------

def _row_fkey(row, dup_cbds: set | None = None) -> str:
    """Build a unique frame key.  When a CBD appears on multiple frames
    (duplicate CBD), disambiguate with frame_idx to avoid key collision."""
    cbd = str(getattr(row, "cbd", "") or "")
    fidx = int(getattr(row, "frame_idx", -1))
    if cbd and dup_cbds and cbd in dup_cbds:
        return f"CBD{cbd}_f{fidx}"
    return f"CBD{cbd}" if cbd else f"fr{fidx}"


def _detect_dup_cbds(rows) -> set:
    """Return set of CBD strings that appear more than once in the frame list."""
    from collections import Counter
    cbds = [str(getattr(r, "cbd", "") or "") for r in rows
            if str(getattr(r, "cbd", "") or "")]
    return {c for c, n in Counter(cbds).items() if n > 1}


def tiff_needs_picks(tiff_name: str, index_df: pd.DataFrame, picks: dict) -> bool:
    """Check if a TIFF has at least one frame with M + R + X picks."""
    tiff_rows = index_df[(index_df["tiff"] == tiff_name) &
                         (index_df["frame_type"] == "complete")]
    for _, row in tiff_rows.iterrows():
        cbd = str(row.get("cbd", "") or "")
        fkey = f"CBD{cbd}" if cbd else f"fr{row['frame_idx']}"
        p = picks.get(fkey, {})
        if (p.get("mb") is not None and p.get("ref") is not None
                and len(p.get("x_grid", [])) >= 2):
            return False
    return True


def build_tiff_queue(args) -> tuple[list[Path], int, Path]:
    """Build list of TIFF paths to process. Returns (tiff_list, flt, out_dir)."""

    if args.flight is not None:
        flt = args.flight
        raw_dir = Path(os.environ.get("LYRA_TIFF_DIR", ROOT / f"tiff/{flt}"))
        if not raw_dir.exists():
            raw_dir = ROOT / "tiff"
        out_dir = output_root(ROOT / "output") / f"F{flt}"
        phase1_dir = out_dir / "phase1"
        index_csv = phase1_dir / f"F{flt}_frame_index.csv"
        picks_json = phase1_dir / f"F{flt}_cal_picks.json"

        if not raw_dir.exists():
            sys.exit(f"ERROR: Raw data directory not found: {raw_dir}")
        if not index_csv.exists():
            sys.exit(f"ERROR: Frame index not found: {index_csv}\n"
                     "Run phase 1 first (detect_frames.py or run_flight.py).")

        index_df = pd.read_csv(index_csv, dtype=str)
        picks = {}
        if picks_json.exists():
            with open(picks_json) as f:
                picks = json.load(f)

        # Discover all canonical TIFFs
        all_tiffs = sorted(
            (t for t in raw_dir.iterdir()
             if t.is_file() and t.suffix.lower() in {".tif", ".tiff"}),
            key=lambda t: lyra_tiff_id(t),
        )
        canonical = [t for t in all_tiffs if parse_tiff_filename(t)]

        if args.tiff is not None:
            # Single TIFF by ID — exact match (compare as integers to handle
            # zero-padding: --tiff 25 matches tiff_id "0025")
            target_id = int(args.tiff)
            matches = [t for t in canonical
                       if lyra_tiff_id(t).lstrip("0") == str(target_id)
                       or (target_id == 0 and lyra_tiff_id(t).lstrip("0") == "")]
            if not matches:
                sys.exit(f"ERROR: No TIFF matching ID {args.tiff} in {raw_dir}")
            return [ensure_canonical_name(matches[0])], flt, out_dir

        # Filter to TIFFs needing picks
        queue = []
        for tiff in canonical:
            tiff = ensure_canonical_name(tiff)
            if tiff_needs_picks(tiff.name, index_df, picks):
                queue.append(tiff)

        if not queue:
            print(f"\n  All TIFFs in F{flt} already have M+R+X picks. Nothing to do.")
            sys.exit(0)

        print(f"\n  Flight F{flt}: {len(queue)} of {len(canonical)} TIFFs need picks")
        return queue, flt, out_dir

    else:
        # Single TIFF mode (backward compatible)
        if args.tiff is not None:
            sys.exit("ERROR: --tiff requires --flight")

        if args.tiff_path:
            tiff = resolve_tiff_arg(args.tiff_path, ROOT)
        else:
            sys.exit("ERROR: TIFF path or --flight is required.")

        tiff = ensure_canonical_name(tiff)
        try:
            metadata = parse_tiff_filename(tiff)
            flt = metadata[0] if metadata else int(tiff.parent.name.lstrip("Ff"))
        except ValueError:
            flt = int(os.environ.get("LYRA_FLIGHT", "0"))
        if flt <= 0:
            sys.exit("ERROR: Cannot infer flight; use FLT/TIFF_ID or tiff/<flight>/.")
        out_dir = output_root(ROOT / "output") / f"F{flt}"
        return [tiff], flt, out_dir


# -- Save + derive -------------------------------------------------------------

def save_picks_and_derive(all_picks: dict, flt: int, out_dir: Path):
    """Save raw picks JSON and derive spacing constants."""
    phase1_dir = out_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    picks_json = phase1_dir / f"F{flt}_cal_picks.json"

    with open(picks_json, "w") as f:
        json.dump(all_picks, f, indent=2)

    # Derive spacing constants
    x_spacings: list[float] = []
    y_spacings: list[float] = []

    for fkey, p in all_picks.items():
        if p.get("exclude"):
            continue
        xg = sorted(float(v) for v in p.get("x_grid", []))
        if len(xg) >= 4:
            # Use all four X-grid clicks.  Clicks may skip major divisions, so
            # infer each click's integer division index from the default spacing,
            # then fit x = intercept + spacing * division_index.
            k = np.array([round((x - xg[0]) / _PX_PER_DIV_X) for x in xg], dtype=float)
            if len(np.unique(k)) >= 2:
                spacing = abs(float(np.polyfit(k, np.array(xg, dtype=float), 1)[0]))
                x_spacings.append(spacing)

        yg = sorted(p.get("y_grid", []))
        if len(yg) == 2:
            raw = abs(yg[1] - yg[0])
            n_div = max(1, round(raw / _PX_PER_DIV_Y))
            y_spacings.append(raw / n_div)

    x_spacing_px = float(np.median(x_spacings)) if x_spacings else _PX_PER_DIV_X
    y_spacing_px = float(np.median(y_spacings)) if y_spacings else _PX_PER_DIV_Y

    mb_guides  = {k: v["mb"]  for k, v in all_picks.items()
                  if v.get("mb")  is not None and not v.get("exclude")}
    ref_guides = {k: v["ref"] for k, v in all_picks.items()
                  if v.get("ref") is not None}

    derived = {
        "flight":            flt,
        "n_frames_picked":   len(all_picks),
        "x_spacing_px":      round(x_spacing_px, 3),
        "y_spacing_px":      round(y_spacing_px, 3),
        "us_per_px":         round(2.0  / x_spacing_px, 8),
        "db_per_px":         round(10.0 / y_spacing_px, 8),
        "x_spacings_per_frame": {
            k: round(abs(float(np.polyfit(
                np.array([round((x - min(v.get("x_grid", []))) / _PX_PER_DIV_X)
                          for x in sorted(float(z) for z in v.get("x_grid", []))], dtype=float),
                np.array(sorted(float(z) for z in v.get("x_grid", [])), dtype=float), 1)[0])), 2)
            for k, v in all_picks.items()
            if len(v.get("x_grid", [])) >= 4
        },
        "y_spacings_per_frame": {
            k: round(abs(sorted(v.get("y_grid",[])[:])[1] - sorted(v.get("y_grid",[])[:])[0]) /
                     max(1, round(abs(sorted(v.get("y_grid",[])[:])[1] - sorted(v.get("y_grid",[])[:])[0]) / _PX_PER_DIV_Y)), 2)
            for k, v in all_picks.items()
            if len(v.get("y_grid", [])) == 2
        },
        "mb_guides":  mb_guides,
        "ref_guides": ref_guides,
    }

    derived_json = phase1_dir / f"F{flt}_cal_derived.json"
    with open(derived_json, "w") as f:
        json.dump(derived, f, indent=2)

    return picks_json, derived_json, derived


# -- Interactive picker for one TIFF -------------------------------------------

def pick_one_tiff(tiff_path: Path, flt: int, all_picks: dict,
                  tiff_num: int, total_tiffs: int, is_flight_mode: bool,
                  cbd_filter: set[int] | None = None) -> str:
    """Run interactive picker on one TIFF.

    Args:
        cbd_filter: if set, only show frames with CBD in this set.

    Returns action: 'next_tiff' (Q in flight mode), 'quit' (Q in single mode
    or Esc), or 'done' (all frames processed).
    """
    tiff_path = ensure_canonical_name(tiff_path)
    tid = lyra_tiff_id(tiff_path)
    out_dir = output_root(ROOT / "output") / f"F{flt}"
    index_csv = out_dir / "phase1" / f"F{flt}_frame_index.csv"

    index = pd.read_csv(index_csv, dtype=str)
    tiff_rows = index[(index["tiff"] == tiff_path.name) &
                      (index["frame_type"] == "complete")].copy()

    if len(tiff_rows) == 0:
        print(f"  WARNING: no complete frames for {tiff_path.name} — skipping")
        return "next_tiff"

    # Load image
    tiff_label = f"TIFF {tiff_num}/{total_tiffs}" if is_flight_mode else ""
    print(f"\n{'='*60}")
    print(f"  {tiff_label}  {tiff_path.name}")
    print(f"  Flight F{flt}  |  TIFF_ID {tid}  |  Complete frames: {len(tiff_rows)}")
    print(f"  Loading TIFF ...")
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W = img_norm.shape
    print(f"  Image: {W} x {H} px")

    if is_flight_mode:
        print(f"  Q = save & next TIFF  |  N = next  |  B = back  |  Esc = quit entirely")
    else:
        print(f"  Q = save & quit  |  N = next  |  B = back")

    # Detect duplicate CBDs within this TIFF (need disambiguated keys)
    all_rows = list(tiff_rows.itertuples())
    dup_cbds = _detect_dup_cbds(all_rows)
    if dup_cbds:
        print(f"  Duplicate CBDs in TIFF: {sorted(dup_cbds)} -- using frame_idx-disambiguated keys")
        # Migrate existing picks from old CBD-only key to disambiguated key
        for r in all_rows:
            cbd = str(getattr(r, "cbd", "") or "")
            if cbd in dup_cbds:
                old_key = f"CBD{cbd}"
                new_key = _row_fkey(r, dup_cbds)
                if old_key in all_picks and new_key not in all_picks:
                    # Only migrate if the new key doesn't exist yet
                    all_picks[new_key] = all_picks[old_key]

    # Build frame list (skip excluded)
    frame_list = [r for r in all_rows
                  if not all_picks.get(_row_fkey(r, dup_cbds), {}).get("exclude")]
    n_excl = len(all_rows) - len(frame_list)
    if n_excl:
        excl_keys = [_row_fkey(r, dup_cbds) for r in all_rows
                     if all_picks.get(_row_fkey(r, dup_cbds), {}).get("exclude")]
        print(f"  Skipping {n_excl} excluded frame(s): {excl_keys}")

    # Apply CBD filter (show only specified CBDs)
    if cbd_filter is not None:
        pre_len = len(frame_list)
        frame_list = [r for r in frame_list
                      if int(getattr(r, "cbd", 0) or 0) in cbd_filter]
        print(f"  CBD filter: showing {len(frame_list)} of {pre_len} frames")

    frame_i = 0
    final_action = "done"

    while frame_i < len(frame_list):
        row = frame_list[frame_i]
        cbd = str(getattr(row, "cbd", "") or "")
        fkey = _row_fkey(row, dup_cbds)
        left_px = int(row.left_px)
        right_px = int(row.right_px)
        frame_w = right_px - left_px + 1

        print(f"\n{'-'*60}")
        frame_label = f"  [{frame_i+1}/{len(frame_list)}]  {fkey}"
        if is_flight_mode:
            frame_label = f"  TIFF {tiff_num}/{total_tiffs}  " + frame_label
        print(f"{frame_label}  cols {left_px}–{right_px}  (width {frame_w} px)")
        print(f"  M=main bang  R=ref -60dB  Y=y-grid  X=x-grid  "
              f"U=undo  N=next  B=back  S=skip  Q={'next TIFF' if is_flight_mode else 'quit'}")

        # Extract + display-stretch this frame
        frame = img_norm[:, left_px:right_px + 1]
        crop = frame[Y_CROP_LO:min(Y_CROP_HI, H), :]
        p2, p98 = np.percentile(crop, 2), np.percentile(crop, 98)
        disp = np.clip((crop - p2) / (p98 - p2 + 1e-9), 0, 1)

        # Per-frame mutable state
        init = all_picks.get(fkey, {})
        st = {
            "mode":  "mb",
            "picks": {
                "mb":     init.get("mb"),
                "ref":    init.get("ref"),
                "y_grid": list(init.get("y_grid", [])),
                "x_grid": list(init.get("x_grid", [])),
            },
            "action": None,
            "drawn":  [],
        }

        # -- Build figure --------------------------------------------------
        # Disable matplotlib default keybindings that conflict with our keys
        for param in list(plt.rcParams):
            if param.startswith("keymap."):
                plt.rcParams[param] = []
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor("white")

        ax.imshow(disp, cmap="gray", vmin=0, vmax=1, aspect="auto",
                  extent=[0, frame_w, Y_CROP_HI, Y_CROP_LO])
        ax.set_ylim(Y_CROP_HI, Y_CROP_LO)
        ax.set_xlabel("X pixel (frame-relative)", fontsize=9)
        ax.set_ylabel("Y pixel (image)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for k in range(-3, 7):
            y_guide = _DB_REF_PX - k * _PX_PER_DIV_Y
            if Y_CROP_LO < y_guide < Y_CROP_HI:
                ax.axhline(y_guide, color="0.65", lw=0.5, ls="--", alpha=0.5)
                db_val = -60 + k * 10
                ax.text(frame_w - 8, y_guide - 10, f"{db_val:+d} dB",
                        color="0.55", fontsize=6, ha="right", va="bottom")

        status_bar = fig.text(
            0.5, 0.005, "", ha="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

        def _update_status():
            m = st["mode"]
            p = st["picks"]
            col = MODE_COLORS[m]
            lbl = MODE_LABELS[m]
            prefix = f"TIFF {tiff_num}/{total_tiffs} | " if is_flight_mode else ""
            status_bar.set_text(
                f"{prefix}[{frame_i+1}/{len(frame_list)}] {fkey}  |  {lbl}    "
                f"mb={1 if p['mb'] is not None else 0}  "
                f"ref={1 if p['ref'] is not None else 0}  "
                f"y_grid={len(p['y_grid'])}/2  x_grid={len(p['x_grid'])}/4"
            )
            status_bar.set_color(col)

        def _redraw():
            for art in st["drawn"]:
                try:
                    art.remove()
                except Exception:
                    pass
            st["drawn"].clear()
            p = st["picks"]

            if p["mb"] is not None:
                x = p["mb"]
                st["drawn"] += [
                    ax.axvline(x, color="red", lw=2.0, ls="--", alpha=0.9),
                    ax.text(x + 8, Y_CROP_LO + 60, f"MB\nx={x}",
                            color="red", fontsize=8, fontweight="bold", va="top"),
                ]

            if p["ref"] is not None:
                y = p["ref"]
                st["drawn"] += [
                    ax.axhline(y, color="cyan", lw=2.0, ls="--", alpha=0.9),
                    ax.text(10, y - 18, f"-60 dB  y={y}",
                            color="cyan", fontsize=8, fontweight="bold"),
                ]

            for y in p["y_grid"]:
                st["drawn"] += [
                    ax.axhline(y, color="limegreen", lw=1.5, ls=":", alpha=0.9),
                    ax.text(10, y - 14, f"Ygrid y={y}",
                            color="limegreen", fontsize=7),
                ]

            for x in p["x_grid"]:
                st["drawn"] += [
                    ax.axvline(x, color="orange", lw=1.5, ls=":", alpha=0.9),
                    ax.text(x + 5, Y_CROP_LO + 130, f"Xgrid\nx={x}",
                            color="orange", fontsize=7, va="top", rotation=90),
                ]

            title_prefix = f"TIFF {tiff_num}/{total_tiffs} — " if is_flight_mode else ""
            q_label = "Q=next TIFF" if is_flight_mode else "Q=quit"
            ax.set_title(
                f"LYRA Calibration Picker — {title_prefix}TIFF {tid} — F{flt}  {fkey}  "
                f"[{frame_i+1}/{len(frame_list)}]\n"
                f"Gray dashed = default-cal prediction  |  "
                f"M=main bang  R=ref-60dB  Y=y-grid  X=x-grid  "
                f"U=undo  N=next  B=back  S=skip  {q_label}",
                fontsize=8, loc="left",
            )

        def _refresh():
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # -- Event handlers ------------------------------------------------
        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            p = st["picks"]
            m = st["mode"]

            if m == "mb":
                p["mb"] = x
                print(f"    MB:     x={x} px (frame-relative)")
            elif m == "ref":
                p["ref"] = y
                print(f"    Ref:    y={y} px  ({(-60 + (_DB_REF_PX - y) * 10.0/_PX_PER_DIV_Y):+.1f} dB by default cal)")
            elif m == "y_grid":
                if len(p["y_grid"]) < 2:
                    p["y_grid"].append(y)
                    p["y_grid"].sort()
                    db_approx = -60 + (_DB_REF_PX - y) * 10.0 / _PX_PER_DIV_Y
                    print(f"    Y-grid: y={y} px  (~{db_approx:+.1f} dB by default cal)  "
                          f"total {len(p['y_grid'])}/2")
                else:
                    print(f"    Y-grid: already has 2 picks — press U to undo, or switch mode")
            elif m == "x_grid":
                if len(p["x_grid"]) < 4:
                    p["x_grid"].append(x)
                    p["x_grid"].sort()
                    us_approx = (x - (p["mb"] or 800)) * 2.0 / _PX_PER_DIV_X
                    print(f"    X-grid: x={x} px  (~{us_approx:+.2f} µs from MB guess)  "
                          f"total {len(p['x_grid'])}/4")
                else:
                    print(f"    X-grid: already has 4 picks — press U to undo, or switch mode")

            _redraw()
            _update_status()
            _refresh()

        def on_key(event):
            key = (event.key or "").lower()
            p = st["picks"]

            if key == "m":
                st["mode"] = "mb"
                print(">> Mode: Main Bang  (click near the leftmost spike)")
            elif key == "r":
                st["mode"] = "ref"
                print(">> Mode: Reference line -60 dB  (click the bottom baseline)")
            elif key == "y":
                st["mode"] = "y_grid"
                p["y_grid"].clear()
                print(">> Mode: Y-grid  (click 2 horizontal major lines)")
            elif key == "x":
                st["mode"] = "x_grid"
                p["x_grid"].clear()
                print(">> Mode: X-grid  (click 2 vertical major lines)")
            elif key == "u":
                m = st["mode"]
                if m == "mb" and p["mb"] is not None:
                    p["mb"] = None
                    print("  Undo: MB cleared")
                elif m == "ref" and p["ref"] is not None:
                    p["ref"] = None
                    print("  Undo: ref cleared")
                elif m == "y_grid" and p["y_grid"]:
                    v = p["y_grid"].pop()
                    print(f"  Undo: y_grid y={v} removed")
                elif m == "x_grid" and p["x_grid"]:
                    v = p["x_grid"].pop()
                    print(f"  Undo: x_grid x={v} removed")
            elif key == "n":
                st["action"] = "next"
                plt.close(fig)
                return
            elif key in ("b", "backspace"):
                st["action"] = "back"
                plt.close(fig)
                return
            elif key == "s":
                st["action"] = "skip"
                plt.close(fig)
                return
            elif key == "q":
                st["action"] = "quit"
                plt.close(fig)
                return
            elif key == "escape":
                st["action"] = "escape"
                plt.close(fig)
                return

            _redraw()
            _update_status()
            _refresh()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.subplots_adjust(bottom=0.06, top=0.92)
        _redraw()
        _update_status()
        _refresh()
        plt.show()

        # -- Post-window: process action -----------------------------------
        p = st["picks"]

        def _save_frame_picks(fkey, p):
            existing = all_picks.get(fkey, {})
            p_save = dict(p)
            for k in ("exclude", "exclude_reason"):
                if k in existing:
                    p_save[k] = existing[k]
            all_picks[fkey] = p_save

        if st["action"] == "escape":
            _save_frame_picks(fkey, p)
            print(f"\n  Escape — picks for {fkey} saved. Quitting entirely.")
            final_action = "quit"
            break
        elif st["action"] == "quit":
            _save_frame_picks(fkey, p)
            if is_flight_mode:
                print(f"  Saved {fkey} — advancing to next TIFF.")
                final_action = "next_tiff"
            else:
                print(f"\n  Quitting — picks for {fkey} saved.")
                final_action = "quit"
            break
        elif st["action"] == "back":
            _save_frame_picks(fkey, p)
            if frame_i > 0:
                print(f"  Saved {fkey} — going back to previous frame.")
                frame_i -= 1
            else:
                print(f"  Already at first frame.")
            continue
        elif st["action"] == "skip":
            print(f"  Skipped {fkey} (no picks saved).")
            frame_i += 1
            continue
        else:  # "next" or window closed
            _save_frame_picks(fkey, p)
            print(f"  Saved {fkey}:  "
                  f"mb={p['mb']}  ref={p['ref']}  "
                  f"y_grid={p['y_grid']}  x_grid={p['x_grid']}")
            frame_i += 1

    return final_action


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()
    tiff_queue, flt, out_dir = build_tiff_queue(args)
    is_flight_mode = args.flight is not None
    phase1_dir = out_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    picks_json = phase1_dir / f"F{flt}_cal_picks.json"

    # Load existing picks
    all_picks: dict = {}
    if picks_json.exists():
        with open(picks_json) as f:
            all_picks = json.load(f)
        print(f"  Loaded {len(all_picks)} existing picks from {picks_json.name}")

    # Parse CBD filter
    cbd_filter: set[int] | None = None
    if args.cbds:
        cbd_filter = {int(c.strip()) for c in args.cbds.split(",")}

    total = len(tiff_queue)
    for i, tiff in enumerate(tiff_queue, 1):
        action = pick_one_tiff(tiff, flt, all_picks, i, total, is_flight_mode,
                               cbd_filter=cbd_filter)

        # Save after each TIFF (incremental — don't lose work)
        pj, dj, derived = save_picks_and_derive(all_picks, flt, out_dir)
        print(f"  Picks saved -> {display_path(pj, ROOT)}")

        if action == "quit":
            break

    # Final summary
    _, _, derived = save_picks_and_derive(all_picks, flt, out_dir)
    print(f"\nDone. Total frames with picks: {len(all_picks)}")
    print(f"  x_spacing_px : {derived['x_spacing_px']:.2f} px  -> {derived['us_per_px']:.6f} µs/px")
    print(f"  y_spacing_px : {derived['y_spacing_px']:.2f} px  -> {derived['db_per_px']:.6f} dB/px")
    print(f"  MB guides    : {len(derived['mb_guides'])}")
    print(f"  Ref guides   : {len(derived['ref_guides'])}")

    if is_flight_mode:
        # Check how many TIFFs still need picks
        index_csv = phase1_dir / f"F{flt}_frame_index.csv"
        if index_csv.exists():
            index_df = pd.read_csv(index_csv, dtype=str)
            raw_dir = ROOT / f"tiff/{flt}"
            if not raw_dir.exists():
                raw_dir = ROOT / "tiff"
            all_tiffs = sorted(
                (t for t in raw_dir.iterdir()
                 if t.is_file() and t.suffix.lower() in {".tif", ".tiff"}),
                key=lambda t: lyra_tiff_id(t),
            )
            canonical = [t for t in all_tiffs if parse_tiff_filename(t)]
            still_need = sum(1 for t in canonical
                             if tiff_needs_picks(ensure_canonical_name(t).name,
                                                 index_df, all_picks))
            if still_need:
                print(f"\n  {still_need} TIFFs still need M+R+X picks.")
                print(f"  Re-run: python codes/pick_calibration.py --flight {flt}")
            else:
                print(f"\n  All TIFFs in F{flt} now have M+R+X picks.")



if __name__ == "__main__":
    main()
''',
    'run_flight': r'''#!/usr/bin/env python3
"""LYRA flight-level orchestrator.

Batch-processes all TIFFs for a single flight through the full LYRA pipeline:

  Phase 1: Frame Detection + CBD Assignment
  Phase 2: ASTRA grid/reference calibration picks
  Phase 3: Per-Frame Calibration + Quality Review (GOOD/CHECK flags)
  Phase 4: Echo Extraction

Usage:
    python LYRA.py --tiff "path\\to\\TIFFs" --output "path\\to\\output" --nav "path\\to\\navigation"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lyra import (ASTRA_X_POINTS, ASTRA_Y_POINTS, navigation_root, output_root,
                  parse_tiff_filename, tiff_id as shared_tiff_id)

# -- Paths --------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DETECT_SCRIPT   = ROOT / "codes/detect_frames.py"
ASTRA_SCRIPT    = ROOT / "codes/ASTRA_GridPixel_V2.py"
CALIBRATE_SCRIPT = ROOT / "codes/calibrate.py"
ECHOES_SCRIPT   = ROOT / "codes/echoes.py"
PICKS_DIR = ROOT / "picks"


# -- Utility ------------------------------------------------------------------

def _tiff_id(path: Path) -> int:
    """Return the starting CBD, used as the current TIFF's stable identifier."""
    try:
        return int(shared_tiff_id(path))
    except ValueError:
        return 0


def discover_tiffs(raw_dir: Path) -> list[Path]:
    """Find supported TIFF files in the input folder, sorted by first CBD."""
    return sorted(
        (path for path in raw_dir.iterdir()
         if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}),
        key=_tiff_id,
    )


def get_astra_coverage(flt: int) -> dict[str, list[int]]:
    """Read CombinedASTRAPicks.csv and return {tiff_stem: [cbd1, cbd2, ...]}."""
    csv_path = PICKS_DIR / str(flt) / f"{flt}_CombinedASTRAPicks.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    coverage = {}
    for stem, grp in df.groupby("filename"):
        cbds = sorted(grp["CBD"].dropna().astype(int).tolist())
        coverage[stem] = cbds
    return coverage


def check_tiff_in_csv(tiff_name: str, csv_path: Path) -> bool:
    """Check if a TIFF already has rows in a phase output CSV."""
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path, dtype=str, usecols=["tiff"])
    df["tiff"] = df["tiff"].str.strip()
    return tiff_name in df["tiff"].values


def _run_step(script: Path, tiff_path: Path, extra_args: list[str] | None = None,
              ) -> subprocess.CompletedProcess:
    """Run a LYRA phase script via subprocess."""
    cmd = [sys.executable, str(script), str(tiff_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def _pause(msg: str = "Press ENTER to continue ..."):
    """Pause and wait for user input."""
    try:
        input(f"\n  {msg} ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Interrupted.")
        sys.exit(1)


# -- Phase 1: Frame Detection ------------------------------------------------

def _count_complete_frames(tiff_name: str, index_csv: Path) -> int:
    """Count complete frames for a TIFF from the frame index CSV."""
    if not index_csv.exists():
        return 0
    df = pd.read_csv(index_csv, dtype=str)
    df.columns = df.columns.str.strip()
    tiff_rows = df[df["tiff"].str.strip() == tiff_name]
    return int((tiff_rows["frame_type"].str.strip() == "complete").sum())


def run_phase1(flt: int, tiffs: list[Path], method: str, out_dir: Path,
               cbd_start_map: dict[int, int] | None = None) -> bool:
    """Run frame detection on all TIFFs; return whether all succeeded."""

    index_csv = out_dir / "phase1" / f"F{flt}_frame_index.csv"
    astra = get_astra_coverage(flt) if method == "manual" else {}
    if cbd_start_map is None:
        cbd_start_map = {}

    # Track next CBD for propagation
    next_cbd: int | None = None

    success, failed, skipped = [], [], []

    for i, tiff in enumerate(tiffs, 1):
        tid = _tiff_id(tiff)

        # Propagate cbd_start: explicit map entry takes priority, else use
        # the running next_cbd from the previous TIFF
        tiff_cbd_start: int | None = None
        if tid in cbd_start_map:
            tiff_cbd_start = cbd_start_map[tid]
        elif next_cbd is not None:
            tiff_cbd_start = next_cbd

        # Skip if already processed
        if check_tiff_in_csv(tiff.name, index_csv):
            skipped.append(tiff)
            # Still propagate: count complete frames to keep next_cbd correct
            if tiff_cbd_start is not None:
                n_complete = _count_complete_frames(tiff.name, index_csv)
                next_cbd = tiff_cbd_start + n_complete
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  [skip]")
            continue

        # Determine method for this TIFF
        tiff_method = method
        extra = ["--method", method]
        if method == "manual" and tiff.stem not in astra:
            tiff_method = "segment"
            extra = ["--method", "segment"]

        # If cbd_start is known, override OCR entirely
        if tiff_cbd_start is not None:
            extra = ["--cbd-start", str(tiff_cbd_start)]
            tiff_method = "cbd-start"

        result = _run_step(DETECT_SCRIPT, tiff, extra)
        if result.returncode == 0:
            success.append(tiff)
            # Extract per-TIFF frame count from stdout
            n_frames = 0
            for line in result.stdout.splitlines():
                if "Frames" in line and "detected" in line:
                    try:
                        n_frames = int(line.split(":")[1].split("detected")[0].strip())
                    except (ValueError, IndexError):
                        pass
            # Propagate next CBD from complete frame count
            if tiff_cbd_start is not None:
                n_complete = _count_complete_frames(tiff.name, index_csv)
                next_cbd = tiff_cbd_start + n_complete
            note = ""
            if tiff_cbd_start is not None:
                note = f"  (cbd-start={tiff_cbd_start})"
            elif method == "manual" and tiff_method == "segment":
                note = "  (no ASTRA)"
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  [{n_frames:>3} fr]  {tiff_method}{note}")
        else:
            details = result.stderr.strip() or result.stdout.strip()
            err_line = details.splitlines()[-1] if details else f"Process exited with code {result.returncode}"
            failed.append((tiff, err_line))
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  \033[31m[FAIL]\033[0m {err_line}")
            if result.stderr.strip():
                print(result.stderr.strip())
            # On failure, stop propagation — don't guess
            if tiff_cbd_start is not None:
                next_cbd = None

    # Print summary
    print(f"\n  Phase 1 summary: {len(success)} new, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        print("\n  Failed TIFFs:")
        for t, msg in failed:
            print(f"    {t.name}: {msg[:100]}")

    # Print CBD lookup table
    print_cbd_table(flt, out_dir)
    return not failed


def print_cbd_table(flt: int, out_dir: Path):
    """Pretty-print TIFF -> CBD mapping from the frame index."""
    index_csv = out_dir / "phase1" / f"F{flt}_frame_index.csv"
    if not index_csv.exists():
        print("  (no frame index found)")
        return

    df = pd.read_csv(index_csv)
    df.columns = df.columns.str.strip()

    print(f"\n  {'TIFF ID':>7}  {'CBD Range':>12}  {'Frames':>6}  TIFF Filename")
    print(f"  {'-'*7}  {'-'*12}  {'-'*6}  {'-'*50}")

    for tid, grp in df.groupby("tiff_id", sort=True):
        tiff_name = grp["tiff"].iloc[0]
        cbds = grp["cbd"].dropna()
        n_frames = len(grp)
        if len(cbds) > 0:
            cbd_min = int(float(cbds.min()))
            cbd_max = int(float(cbds.max()))
            cbd_range = f"{cbd_min:04d}-{cbd_max:04d}"
        else:
            cbd_range = "  (unknown)"
        short_name = tiff_name.split("-")[0] if "-" in tiff_name else tiff_name
        print(f"  {int(tid):>7}  {cbd_range:>12}  {n_frames:>6}  {short_name}")


def _calibration_key(row: pd.Series, duplicate_cbds: set[str]) -> str:
    """Build the frame key used by calibrate.py for an index row."""
    cbd = str(row.get("cbd", "")).strip()
    frame_idx = int(row["frame_idx"])
    if cbd and cbd.lower() != "nan":
        cbd_number = str(int(float(cbd)))
        return (f"CBD{cbd_number}_f{frame_idx}"
                if cbd_number in duplicate_cbds else f"CBD{cbd_number}")
    return f"fr{frame_idx}"


def import_astra_grid_picks(flt: int, tiffs: list[Path], out_dir: Path) -> int:
    """Convert ASTRA's pixel CSV into the guide JSON consumed by calibrate.py."""
    phase1_dir = out_dir / "phase1"
    index_path = phase1_dir / f"F{flt}_frame_index.csv"
    astra_path = phase1_dir / f"F{flt}_astra_grid.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"LYRA frame index not found: {index_path}")
    if not astra_path.exists():
        raise FileNotFoundError(f"ASTRA grid picks not found: {astra_path}")

    frame_index = pd.read_csv(index_path, dtype=str)
    astra = pd.read_csv(astra_path)
    required = {"filename", "frame", "x_ref_abs", "y_ref_abs", "montage_approved"}
    required.update(f"x{i}_abs" for i in range(1, ASTRA_X_POINTS + 1))
    required.update(f"y{i}_abs" for i in range(1, ASTRA_Y_POINTS + 1))
    missing_columns = required.difference(astra.columns)
    if missing_columns:
        raise ValueError(f"ASTRA CSV missing columns: {', '.join(sorted(missing_columns))}")
    crop_columns = {"crop_start_y", "crop_end_y"}
    has_crop_columns = crop_columns.issubset(astra.columns)
    if crop_columns.intersection(astra.columns) and not has_crop_columns:
        raise ValueError("ASTRA CSV must include both crop_start_y and crop_end_y")

    picks_path = phase1_dir / f"F{flt}_cal_picks.json"
    picks = {}
    if picks_path.exists():
        with picks_path.open(encoding="utf-8") as handle:
            picks = json.load(handle)

    imported = 0
    missing_frames = []
    x_spacings, y_spacings = [], []
    for tiff in tiffs:
        rows = frame_index[
            (frame_index["tiff"] == tiff.name)
            & (frame_index["frame_type"] == "complete")
        ].sort_values("frame_idx")
        cbd_counts = pd.to_numeric(rows["cbd"], errors="coerce").dropna().astype(int).value_counts()
        duplicate_cbds = {
            str(cbd) for cbd, count in cbd_counts.items() if count > 1
        }
        captured = astra[astra["filename"].astype(str).str.strip() == tiff.name]
        captured_by_frame = {
            int(float(row["frame"])): row for _, row in captured.iterrows()
        }

        for frame_number, (_, frame_row) in enumerate(rows.iterrows(), start=1):
            astro_row = captured_by_frame.get(frame_number)
            if astro_row is None:
                missing_frames.append(f"{tiff.name} frame {frame_number}")
                continue
            left = int(frame_row["left_px"])
            x_grid = sorted([
                int(round(float(astro_row[f"x{i}_abs"]) - left))
                for i in range(1, ASTRA_X_POINTS + 1)
                if pd.notna(astro_row[f"x{i}_abs"])
            ])
            y_grid = sorted([
                int(round(float(astro_row[f"y{i}_abs"])))
                for i in range(1, ASTRA_Y_POINTS + 1)
                if pd.notna(astro_row[f"y{i}_abs"])
            ])
            y_ref = astro_row["y_ref_abs"]
            x_ref = astro_row["x_ref_abs"]
            if (str(astro_row["montage_approved"]).strip().lower() != "true"
                    or pd.isna(x_ref) or pd.isna(y_ref)
                    or len(x_grid) != ASTRA_X_POINTS or len(y_grid) != ASTRA_Y_POINTS
                    or any(b <= a for a, b in zip(x_grid, x_grid[1:]))
                    or any(b <= a for a, b in zip(y_grid, y_grid[1:]))):
                missing_frames.append(f"{tiff.name} frame {frame_number} (incomplete grid picks)")
                continue
            x_ref_frame = int(round(float(x_ref) - left))
            key = _calibration_key(frame_row, duplicate_cbds)
            previous = picks.get(key, {})
            picks[key] = {
                "x_ref": x_ref_frame,
                "ref": int(round(float(y_ref))),
                "y_grid": y_grid,
                "x_grid": x_grid,
                "montage_approved": True,
            }
            if has_crop_columns:
                crop_start = pd.to_numeric(astro_row["crop_start_y"], errors="coerce")
                crop_end = pd.to_numeric(astro_row["crop_end_y"], errors="coerce")
                if not (pd.isna(crop_start) and pd.isna(crop_end)):
                    if (pd.isna(crop_start) or pd.isna(crop_end)
                            or not float(crop_start).is_integer()
                            or not float(crop_end).is_integer()
                            or crop_start < 0 or crop_end <= crop_start):
                        raise ValueError(
                            f"Invalid ASTRA crop bounds for {tiff.name} frame "
                            f"{frame_number}: {astro_row['crop_start_y']}, "
                            f"{astro_row['crop_end_y']}"
                        )
                    picks[key]["crop_start_y"] = int(crop_start)
                    picks[key]["crop_end_y"] = int(crop_end)
            for field in ("exclude", "exclude_reason"):
                if field in previous:
                    picks[key][field] = previous[field]

            if len(x_grid) >= 4:
                x_values = np.array(sorted(x_grid), dtype=float)
                x_spacings.append(float(np.median(np.diff(x_values))))
            if len(y_grid) >= 2:
                y_values = np.array(sorted(y_grid), dtype=float)
                y_spacings.append(float(np.median(np.diff(y_values))))
            imported += 1

    with picks_path.open("w", encoding="utf-8") as handle:
        json.dump(picks, handle, indent=2)

    x_spacing = float(np.median(x_spacings)) if x_spacings else 205.54
    y_spacing = float(np.median(y_spacings)) if y_spacings else 205.0
    derived = {
        "flight": flt,
        "n_frames_picked": imported,
        "x_spacing_px": round(x_spacing, 3),
        "y_spacing_px": round(y_spacing, 3),
        "us_per_px": round(2.0 / x_spacing, 8),
        "db_per_px": round(10.0 / y_spacing, 8),
        "ref_guides": {key: value["ref"] for key, value in picks.items()
                       if value.get("ref") is not None},
    }
    with (phase1_dir / f"F{flt}_cal_derived.json").open("w", encoding="utf-8") as handle:
        json.dump(derived, handle, indent=2)

    print(f"  Imported ASTRA guides for {imported} frames.")
    if missing_frames:
        print(f"  WARNING: ASTRA grid/reference picks missing or incomplete "
              f"for {len(missing_frames)} frames:")
        for frame in missing_frames[:20]:
            print(f"    {frame}")
        if len(missing_frames) > 20:
            print(f"    ... and {len(missing_frames) - 20} more")
    return imported


def missing_astra_calibration_picks(flt: int, tiffs: list[Path],
                                    out_dir: Path) -> list[str]:
    """List complete frames without approved ASTRA grid and reference guides."""
    phase1_dir = out_dir / "phase1"
    index_path = phase1_dir / f"F{flt}_frame_index.csv"
    picks_path = phase1_dir / f"F{flt}_cal_picks.json"
    if not index_path.exists() or not picks_path.exists():
        return [f"ASTRA picks or frame index missing for F{flt}"]

    frame_index = pd.read_csv(index_path, dtype=str)
    with picks_path.open(encoding="utf-8") as handle:
        picks = json.load(handle)
    missing = []
    for tiff in tiffs:
        rows = frame_index[
            (frame_index["tiff"] == tiff.name)
            & (frame_index["frame_type"] == "complete")
        ].sort_values("frame_idx")
        cbd_counts = pd.to_numeric(rows["cbd"], errors="coerce").dropna().astype(int).value_counts()
        duplicate_cbds = {
            str(cbd) for cbd, count in cbd_counts.items() if count > 1
        }
        for _, row in rows.iterrows():
            key = _calibration_key(row, duplicate_cbds)
            guide = picks.get(key, {})
            x_grid = guide.get("x_grid")
            y_grid = guide.get("y_grid")
            if (guide.get("x_ref") is None or guide.get("ref") is None
                    or guide.get("montage_approved") is not True
                    or not isinstance(x_grid, list) or len(x_grid) != ASTRA_X_POINTS
                    or not isinstance(y_grid, list) or len(y_grid) != ASTRA_Y_POINTS):
                missing.append(f"{tiff.name} {key}")
    return missing


def run_astra_phase(flt: int, tiffs: list[Path], out_dir: Path,
                    only_tiffs: list[Path] | None = None) -> bool:
    """Launch ASTRA on LYRA's detected frames and import its grid picks."""
    if not tiffs:
        print("  ASTRA skipped: no TIFFs were selected.")
        return False
    phase1_dir = out_dir / "phase1"
    index_path = phase1_dir / f"F{flt}_frame_index.csv"
    output_csv = phase1_dir / f"F{flt}_astra_grid.csv"
    if not index_path.exists():
        print(f"  ASTRA cannot start: frame index is missing at {index_path}")
        return False

    cmd = [
        sys.executable, str(ASTRA_SCRIPT), "--lyra",
        "--tiff-dir", str(tiffs[0].parent),
        "--frame-index", str(index_path),
        "--output-csv", str(output_csv),
    ]
    if only_tiffs:
        cmd.extend(["--tiff-names", *[tiff.name for tiff in only_tiffs]])
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        print(f"  ASTRA grid/reference picking failed (exit code {result.returncode}).")
        return False
    imported = import_astra_grid_picks(flt, tiffs, out_dir)
    frame_index = pd.read_csv(index_path, dtype=str)
    expected = sum(
        int(((frame_index["tiff"] == tiff.name)
             & (frame_index["frame_type"] == "complete")).sum())
        for tiff in tiffs
    )
    if imported != expected:
        print(f"  ASTRA grid/reference picks are incomplete: {imported}/{expected} "
              "complete frames imported.")
        return False
    return True


def report_filename_ranges(flt: int, tiffs: list[Path], out_dir: Path):
    """Compare detected complete-frame counts to the CBD ranges in filenames."""
    index_path = out_dir / "phase1" / f"F{flt}_frame_index.csv"
    if not index_path.exists():
        return
    frame_index = pd.read_csv(index_path, dtype=str)
    for tiff in tiffs:
        metadata = parse_tiff_filename(tiff)
        if metadata is None:
            continue
        _, first_cbd, last_cbd = metadata
        rows = frame_index[
            (frame_index["tiff"] == tiff.name)
            & (frame_index["frame_type"] == "complete")
        ]
        expected = last_cbd - first_cbd + 1
        actual = len(rows)
        assigned = (
            pd.to_numeric(rows["cbd"], errors="coerce")
            .dropna().astype(int).tolist()
        )
        if actual != expected or assigned != list(range(first_cbd, last_cbd + 1)):
            print(
                f"  WARNING: {tiff.name} declares CBD {first_cbd}-{last_cbd} "
                f"({expected} expected), but Phase 1 found {actual} complete frames "
                f"({len(assigned)} assigned). Check the frame index."
            )


def _delete_tiff_rows(csv_path: Path, tiff_ids: list[int]):
    """Remove rows for given tiff_ids from a phase CSV so they can be re-run."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    before = len(df)
    df = df[~df["tiff_id"].isin(tiff_ids)]
    df.to_csv(csv_path, index=False)
    print(f"  Removed {before - len(df)} rows for TIFFs {tiff_ids} from {csv_path.name}")


# -- Phase 3: Calibration ----------------------------------------------------

def run_phase3(flt: int, tiffs: list[Path], out_dir: Path) -> dict:
    """Run calibration on all TIFFs with y_ref failure detection."""

    cal_csv = out_dir / "phase3" / f"F{flt}_cal.csv"
    success, failed, skipped, needs_picks = [], [], [], []

    for i, tiff in enumerate(tiffs, 1):
        tid = _tiff_id(tiff)

        result = _run_step(CALIBRATE_SCRIPT, tiff)

        if result.returncode == 0:
            if result.stderr.strip():
                print(result.stderr.strip())
            success.append(tiff)
            # Parse frame count from "Frames in index: N  (complete: M)"
            n_frames = 0
            for line in result.stdout.splitlines():
                if "Frames in index" in line:
                    try:
                        n_frames = int(line.split(":")[1].split("(")[0].strip())
                    except (ValueError, IndexError):
                        pass
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  [{n_frames:>3} fr]")
        elif "No reference-line pick" in (result.stderr + result.stdout):
            needs_picks.append(tiff)
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  \033[33m[NEED PICK]\033[0m missing y_ref")
        else:
            err_line = (result.stderr.strip() or result.stdout.strip()).split("\n")[-1]
            failed.append((tiff, err_line))
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  \033[31m[FAIL]\033[0m {err_line}")

    print(f"\n  Phase 3 summary: {len(success)} recalibrated, {len(skipped)} skipped, "
          f"{len(needs_picks)} need picks, {len(failed)} failed")

    return {
        "success": success, "failed": failed,
        "skipped": skipped, "needs_picks": needs_picks,
    }


def analyze_phase3_results(flt: int, out_dir: Path):
    """Analyze calibration results across the flight."""
    cal_csv = out_dir / "phase3" / f"F{flt}_cal.csv"
    if not cal_csv.exists():
        return

    df = pd.read_csv(cal_csv)
    df.columns = df.columns.str.strip()

    # Filter to calibrated frames (non-excluded)
    cal = df[df["exclude_reason"].isna() | (df["exclude_reason"].fillna("").astype(str).str.strip() == "")]

    if len(cal) == 0:
        print("  No calibrated frames found.")
        return

    mb_power = cal["mb_power_dB"].astype(float)
    mb_median = mb_power.median()
    x_spacing = cal["x_spacing_px"].astype(float)
    x_median = x_spacing.median()

    print(f"\n  Phase 3 flight diagnostics ({len(cal)} calibrated frames):")
    print(f"    mb_power:  median {mb_median:.1f} dB, range [{mb_power.min():.1f}, {mb_power.max():.1f}] dB")
    print(f"    x_spacing: median {x_median:.1f} px, range [{x_spacing.min():.1f}, {x_spacing.max():.1f}] px")

    # Flag outliers
    outliers = cal[mb_power < mb_median - 8]
    if len(outliers) > 0:
        tids = outliers["tiff_id"].unique()
        print(f"    WARNING: {len(outliers)} frames with mb_power > 8 dB below median "
              f"(TIFFs: {', '.join(str(int(t)) for t in tids)})")

    x_outliers = cal[(x_spacing - x_median).abs() > 10]
    if len(x_outliers) > 0:
        tids = x_outliers["tiff_id"].unique()
        print(f"    WARNING: {len(x_outliers)} frames with x_spacing > 10 px from median "
              f"(TIFFs: {', '.join(str(int(t)) for t in tids)})")

    n_excluded = len(df) - len(cal)
    if n_excluded > 0:
        print(f"    Excluded frames: {n_excluded}")


def _flag_bad_cbds(sub: pd.DataFrame, flt_xs_med: float) -> list[int]:
    """Flag individual CBDs with geometric calibration problems within a TIFF.

    Focuses on X-grid detection failures (affects h_ice accuracy).
    MB power variation is NOT flagged — it is a known instrumental effect
    corrected in phase 4 using per-frame mb_power as a gain reference.

    A frame is flagged if:
      - x_spacing_px deviates > 15 px from flight median (~7% error in TWT)
    """
    bad = sub[
        (sub["x_spacing_px"] - flt_xs_med).abs() > 15
    ]
    return sorted(bad["cbd"].dropna().astype(int).tolist())


def print_phase3_cbd_table(flt: int, out_dir: Path) -> tuple[list[int], dict[int, tuple[str, list[int]]]]:
    """Print TIFF -> CBD mapping with MB quality flags.

    Returns (check_tiff_ids, check_details) where check_details maps
    tiff_id -> (reason, bad_cbd_list).
    """
    cal_csv = out_dir / "phase3" / f"F{flt}_cal.csv"
    if not cal_csv.exists():
        return [], {}

    df = pd.read_csv(cal_csv)
    df.columns = df.columns.str.strip()
    # Ensure numeric types for flagging
    for col in ("mb_x", "mb_power_dB", "x_spacing_px"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flight-wide x_spacing median for outlier detection
    _excl = df["exclude_reason"].fillna("").astype(str).str.strip()
    _valid = df[_excl.isin(["", "nan"])].dropna(subset=["x_spacing_px"])
    flt_xs_med = float(_valid["x_spacing_px"].median()) if len(_valid) > 0 else 205.0

    print(f"\n  TIFF -> CBD mapping (phase 3 calibration):")
    print(f"  {'TIFF':>6}  {'CBD range':>14}  {'frames':>6}  {'x_sp med':>9}  {'x_sp std':>9}  "
          f"{'mb_pwr med':>10}  {'mb_pwr std':>10}  {'flag':>6}")
    print(f"  {'-'*85}")

    check_tiffs: list[int] = []
    check_details: dict[int, tuple[str, list[int]]] = {}  # tid -> (reason, bad_cbds)

    for tid in sorted(df["tiff_id"].unique()):
        sub = df[df["tiff_id"] == tid].sort_values("cbd")
        n = len(sub)
        cbds = sub["cbd"].dropna()
        cbd_lo = f"{cbds.min():04.0f}" if len(cbds) > 0 else "?"
        cbd_hi = f"{cbds.max():04.0f}" if len(cbds) > 0 else "?"
        xs_med = sub["x_spacing_px"].median()
        xs_std = sub["x_spacing_px"].std() if len(sub) > 1 else 0.0
        mp_med = sub["mb_power_dB"].median()
        mp_std = sub["mb_power_dB"].std() if len(sub) > 1 else 0.0

        # CHECK criteria: geometric calibration problems only.
        # MB power variation is NOT flagged — it is a known instrumental
        # effect (attenuator / CRT intensity) corrected in phase 4.
        flag = "GOOD"
        reason = ""
        if xs_std > 10:
            flag = "CHECK"
            reason = "x_spacing varies > 10 px within TIFF"
        elif abs(xs_med - flt_xs_med) > 15:
            flag = "CHECK"
            reason = f"x_spacing median {xs_med:.0f} deviates from flight ({flt_xs_med:.0f})"

        if flag == "CHECK":
            check_tiffs.append(int(tid))
            bad_cbds = _flag_bad_cbds(sub, flt_xs_med)
            check_details[int(tid)] = (reason, bad_cbds)

        if flag == "CHECK":
            flag_str = f"\033[1;31m{'CHECK':>6}\033[0m"
        else:
            flag_str = f"\033[32m{'GOOD':>6}\033[0m"
        print(f"  {tid:6.0f}  {cbd_lo}-{cbd_hi:>4}  {n:6d}  {xs_med:9.1f}  {xs_std:9.1f}  "
              f"{mp_med:10.1f}  {mp_std:10.1f}  {flag_str}")

    # Summary
    n_total_tiffs = df["tiff_id"].nunique()
    n_good = n_total_tiffs - len(check_tiffs)
    print(f"\n  Total: {len(df)} frames across {n_total_tiffs} TIFFs "
          f"({n_good} GOOD, {len(check_tiffs)} CHECK)")

    if check_tiffs:
        # ANSI codes for bold yellow on terminal
        BOLD = "\033[1m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        RESET = "\033[0m"

        print(f"\n  {BOLD}{YELLOW}+{'-'*70}+{RESET}")
        print(f"  {BOLD}{YELLOW}|  [!]  CHECK TIFFs — review diagnostic PNGs in phase3/{' '*19}|{RESET}")
        print(f"  {BOLD}{YELLOW}+{'-'*70}+{RESET}")
        for tid in check_tiffs:
            reason, bad_cbds = check_details[tid]
            # Count total CBDs in this TIFF
            n_total = len(df[df["tiff_id"] == tid]["cbd"].dropna())
            if len(bad_cbds) == 0 or len(bad_cbds) >= n_total:
                cbd_str = "all CBDs"
            else:
                cbd_str = ", ".join(f"{c:04d}" for c in bad_cbds)
            line = f"TIFF {tid}: {reason} -> {cbd_str}"
            # Wrap long lines
            if len(line) <= 68:
                print(f"  {BOLD}{YELLOW}|{RESET}  {RED}{line:<68}{RESET}{BOLD}{YELLOW}|{RESET}")
            else:
                print(f"  {BOLD}{YELLOW}|{RESET}  {RED}{line[:68]}{RESET}{BOLD}{YELLOW}|{RESET}")
                print(f"  {BOLD}{YELLOW}|{RESET}  {RED}{'':>16}{line[68:]:<52}{RESET}{BOLD}{YELLOW}|{RESET}")
        # Count total flagged CBDs across all CHECK TIFFs
        n_flagged_cbds = 0
        n_total_cbds = len(df)
        for tid in check_tiffs:
            _, bad_cbds = check_details[tid]
            n_in_tiff = len(df[df["tiff_id"] == tid]["cbd"].dropna())
            n_flagged_cbds += len(bad_cbds) if bad_cbds and len(bad_cbds) < n_in_tiff else n_in_tiff
        summary = f"{n_flagged_cbds} of {n_total_cbds} frames need checking"
        print(f"  {BOLD}{YELLOW}+{'-'*70}+{RESET}")
        print(f"  {BOLD}{YELLOW}|{RESET}  {BOLD}{summary:<68}{RESET}{BOLD}{YELLOW}|{RESET}")
        print(f"  {BOLD}{YELLOW}+{'-'*70}+{RESET}")

    return check_tiffs, check_details


def run_phase4(flt: int, tiffs: list[Path], out_dir: Path) -> dict:
    """Run echo extraction on all TIFFs."""

    echo_csv = out_dir / "phase4" / f"F{flt}_echoes.csv"
    success, failed, skipped = [], [], []

    for i, tiff in enumerate(tiffs, 1):
        tid = _tiff_id(tiff)

        result = _run_step(ECHOES_SCRIPT, tiff)

        if result.returncode == 0:
            if result.stderr.strip():
                print(result.stderr.strip())
            success.append(tiff)
            # Parse echo status counts from stdout
            counts: dict[str, int] = {}
            n_frames = 0
            for line in result.stdout.splitlines():
                if "Frames processed" in line:
                    try:
                        n_frames = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                for st in ("good", "weak_bed", "no_bed", "no_surface"):
                    if line.strip().startswith(st):
                        try:
                            counts[st] = int(line.split(":")[-1].strip())
                        except ValueError:
                            pass
            n_good = counts.get("good", 0) + counts.get("weak_bed", 0)
            n_nobed = counts.get("no_bed", 0)
            parts = []
            if n_good:
                parts.append(f"\033[32m{n_good} good\033[0m")
            if n_nobed:
                parts.append(f"\033[33m{n_nobed} no_bed\033[0m")
            n_nosurf = counts.get("no_surface", 0)
            if n_nosurf:
                parts.append(f"\033[31m{n_nosurf} no_surface\033[0m")
            summary = ", ".join(parts) if parts else f"{n_frames} frames"
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  [{n_frames:>3} fr]  {summary}")
        else:
            err_line = (result.stderr + result.stdout).strip().split("\n")[-1]
            failed.append((tiff, err_line))
            print(f"  [{i:>2}/{len(tiffs)}]  {tid:>5}  \033[31m[FAIL]\033[0m {err_line[:60]}")

    print(f"\n  Phase 4 summary: {len(success)} new, {len(skipped)} skipped, {len(failed)} failed")

    return {"success": success, "failed": failed, "skipped": skipped}


def print_flight_summary(flt: int, out_dir: Path):
    """Print final flight-level summary from echo extraction."""
    echo_csv = out_dir / "phase4" / f"F{flt}_echoes.csv"
    if not echo_csv.exists():
        print("  (no echoes found)")
        return

    df = pd.read_csv(echo_csv)
    df.columns = df.columns.str.strip()

    n = len(df)
    status = df["echo_status"].str.strip()

    n_good = int((status == "good").sum())
    n_weak = int((status == "weak_bed").sum())
    n_nobed = int((status == "no_bed").sum())
    n_nosurf = int((status == "no_surface").sum())

    cbds = df["cbd"].dropna().astype(float)
    cbd_min = int(cbds.min()) if len(cbds) > 0 else 0
    cbd_max = int(cbds.max()) if len(cbds) > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  Flight F{flt} Summary")
    print(f"  {'='*50}")
    print(f"    Total frames:  {n}")
    print(f"    good:          {n_good:>4}  ({100*n_good/n:.1f}%)" if n else "")
    print(f"    weak_bed:      {n_weak:>4}  ({100*n_weak/n:.1f}%)" if n else "")
    print(f"    no_bed:        {n_nobed:>4}  ({100*n_nobed/n:.1f}%)" if n else "")
    print(f"    no_surface:    {n_nosurf:>4}  ({100*n_nosurf/n:.1f}%)" if n else "")
    print(f"    CBD range:     {cbd_min:04d} - {cbd_max:04d}")

    # Good-frame statistics
    good = df[status == "good"]
    if len(good) > 0:
        h_ice = good["h_ice_m"].dropna().astype(float)
        h_air = good["h_air_m"].dropna().astype(float)
        if len(h_ice) > 0:
            print(f"    h_ice (good):  {h_ice.mean():.0f} +/- {h_ice.std():.0f} m "
                  f"(range {h_ice.min():.0f}-{h_ice.max():.0f} m)")
        if len(h_air) > 0:
            print(f"    h_air (good):  {h_air.mean():.0f} +/- {h_air.std():.0f} m "
                  f"(range {h_air.min():.0f}-{h_air.max():.0f} m)")

    n_tiffs = df["tiff_id"].nunique()
    print(f"    TIFFs:         {n_tiffs}")


# -- Main ---------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="LYRA flight-level orchestrator")
    parser.add_argument("--tiff", required=True, type=Path,
                        help="Folder containing TIFFs named F<flight>-C<first>_C<last>.tiff")
    parser.add_argument("--output", type=Path, default=output_root(),
                        help="Output root; an F<flight> folder is created inside it")
    parser.add_argument("--nav", type=Path, default=navigation_root(),
                        help="Folder containing the flight navigation CSV (e.g. 103.csv)")
    parser.add_argument("--resume-from", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Resume from phase N (skip earlier phases)")
    parser.add_argument("--skip-astra", "--skip-picks", dest="skip_astra",
                        action="store_true",
                        help="Skip ASTRA grid/reference picking (use previously saved picks)")
    parser.add_argument("--cbd-start", metavar="TIFF_ID:CBD",
                        help="Override the filename's starting CBD for one TIFF, "
                             "e.g. --cbd-start 1024:1024")
    args = parser.parse_args(argv)

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.tiff.expanduser().resolve()
    if not raw_dir.is_dir():
        sys.exit(f"ERROR: TIFF input folder not found: {raw_dir}")
    tiffs = discover_tiffs(raw_dir)
    if not tiffs:
        sys.exit(f"ERROR: No .tif or .tiff files found in {raw_dir}")

    try:
        parsed = [(tiff, parse_tiff_filename(tiff)) for tiff in tiffs]
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    invalid = [tiff.name for tiff, metadata in parsed if metadata is None]
    if invalid:
        sys.exit(
            "ERROR: TIFF names must follow F<flight>-C<first>_C<last>.tiff. "
            f"Invalid files: {', '.join(invalid)}"
        )
    flights = {metadata[0] for _, metadata in parsed if metadata is not None}
    if len(flights) != 1:
        sys.exit(f"ERROR: Expected TIFFs from one flight, found: {sorted(flights)}")
    flt = flights.pop()

    nav_dir = args.nav.expanduser().resolve()
    os.environ["LYRA_FLIGHT"] = str(flt)
    os.environ["LYRA_OUTPUT_DIR"] = str(output_dir)
    os.environ["LYRA_NAV_DIR"] = str(nav_dir)
    os.environ["LYRA_TIFF_DIR"] = str(raw_dir)
    out_dir = output_dir / f"F{flt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cbd_start_map: dict[int, int] = {
        int(shared_tiff_id(tiff)): metadata[1]
        for tiff, metadata in parsed if metadata is not None
    }
    if args.cbd_start:
        if ":" not in args.cbd_start:
            sys.exit("ERROR: --cbd-start expects TIFF_ID:CBD format, e.g. 1024:1024")
        try:
            _tid_str, _cbd_str = args.cbd_start.split(":", 1)
            cbd_start_map[int(_tid_str)] = int(_cbd_str)
        except ValueError:
            sys.exit(f"ERROR: --cbd-start expects integers, got '{args.cbd_start}'")

    print(f"\n  LYRA Flight Orchestrator — F{flt}")
    print(f"  {'-'*40}")
    print(f"  Raw directory:  {raw_dir}")
    print(f"  TIFFs found:    {len(tiffs)}")
    print(f"  Output root:    {output_dir}")
    print(f"  Navigation:     {nav_dir / f'{flt}.csv'}")
    if cbd_start_map:
        print("  CBD ranges:     read from TIFF filenames")
    if not nav_dir.is_dir():
        print(f"  WARNING: Navigation folder not found yet: {nav_dir}")
    elif not (nav_dir / f"{flt}.csv").is_file():
        print(f"  WARNING: Navigation CSV not found yet: {nav_dir / f'{flt}.csv'}")

    # -- Phase 1 ----------------------------------------------------------
    if args.resume_from <= 1:
        print(f"\n  {'='*55}")
        print(f"  PHASE 1: Frame Detection + CBD Assignment")
        print(f"  {'='*55}\n")

        if not run_phase1(flt, tiffs, "segment", out_dir, cbd_start_map):
            sys.exit("ERROR: Phase 1 failed. Resolve the errors above before continuing.")
        report_filename_ranges(flt, tiffs, out_dir)

        print(f"\n  Check contact sheets in: {out_dir / 'phase1'}/")
        _pause("Press ENTER when frame detection and CBD ranges look correct ...")

    # -- Phase 2 ----------------------------------------------------------
    if args.resume_from <= 2 and not args.skip_astra:
        print(f"\n  {'='*55}")
        print(f"  PHASE 2: ASTRA Grid/Reference Picks")
        print(f"  {'='*55}\n")
        print("  ASTRA uses LYRA's detected frames to approve the grids and reference lines. "
              "The approved X reference is the main bang.\n")
        if not run_astra_phase(flt, tiffs, out_dir):
            sys.exit("ERROR: ASTRA calibration picks were not completed.")

    if args.resume_from == 3 and not args.skip_astra:
        missing_guides = missing_astra_calibration_picks(flt, tiffs, out_dir)
        if missing_guides and not run_astra_phase(flt, tiffs, out_dir):
            sys.exit("ERROR: ASTRA grid/reference picks are incomplete.")

    if args.skip_astra and args.resume_from <= 3:
        missing_guides = missing_astra_calibration_picks(flt, tiffs, out_dir)
        if missing_guides:
            listed = ", ".join(missing_guides[:10])
            if len(missing_guides) > 10:
                listed += f", and {len(missing_guides) - 10} more"
            sys.exit(
                "ERROR: --skip-astra requires saved ASTRA x/y grids and reference "
                "lines for every complete frame. "
                f"Missing: {listed}. Re-run from phase 2 without --skip-astra."
            )

    # -- Phase 3: Calibration ---------------------------------------------
    if args.resume_from <= 3:
        print(f"\n  {'='*55}")
        print(f"  PHASE 3: Per-Frame Calibration")
        print(f"  {'='*55}\n")
        print("  Using ASTRA's approved grids, references, and main-bang positions.\n")

        s2 = run_phase3(flt, tiffs, out_dir)

        # Handle y_ref failures
        if s2["needs_picks"]:
            print(f"\n  {len(s2['needs_picks'])} TIFFs missing ASTRA reference picks.")
            if not run_astra_phase(flt, tiffs, out_dir, s2["needs_picks"]):
                sys.exit("ERROR: ASTRA did not provide the missing reference picks.")

            # Re-run calibration on previously-failed TIFFs
            print(f"\n  Re-running calibration on {len(s2['needs_picks'])} TIFFs ...\n")
            s2_retry = run_phase3(flt, s2["needs_picks"], out_dir)

            if s2_retry["needs_picks"] or s2_retry["failed"]:
                still_bad = s2_retry["needs_picks"] + [t for t, _ in s2_retry["failed"]]
                sys.exit(f"ERROR: {len(still_bad)} TIFFs still lack valid calibration after retry.")

        if s2["failed"]:
            sys.exit("ERROR: Phase 3 calibration failed; resolve the errors before Phase 4.")

        # Flight-level analysis + CBD table
        analyze_phase3_results(flt, out_dir)
        check_tiffs, check_details = print_phase3_cbd_table(flt, out_dir)

        # Fix loop — repeat until user is satisfied
        while check_tiffs:
            print(f"\n  {len(check_tiffs)} TIFFs flagged CHECK — review diagnostic PNGs in:")
            print(f"    {out_dir / 'phase3'}/")
            print(f"  Verify: red dashed mb_line should be on the main bang, not surface/bed.")
            try:
                choice = input("\n  Fix flagged TIFFs? [f]ix / [c]ontinue to phase 4: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "c"

            if not choice.startswith("f"):
                break

            check_paths = [t for t in tiffs if _tiff_id(t) in check_tiffs]
            print(f"\n  Reopening ASTRA grid picks for {len(check_paths)} CHECK TIFFs ...")
            if not run_astra_phase(flt, tiffs, out_dir, check_paths):
                sys.exit("ERROR: ASTRA re-picking did not complete.")

            # Delete CHECK rows and re-run calibration
            cal_csv = out_dir / "phase3" / f"F{flt}_cal.csv"
            _delete_tiff_rows(cal_csv, check_tiffs)

            print(f"\n  Re-running calibration on {len(check_paths)} fixed TIFFs ...\n")
            retry = run_phase3(flt, check_paths, out_dir)
            if retry["failed"] or retry["needs_picks"]:
                sys.exit("ERROR: Recalibration failed; resolve the errors before Phase 4.")

            # Re-print table and loop
            analyze_phase3_results(flt, out_dir)
            check_tiffs, check_details = print_phase3_cbd_table(flt, out_dir)

        if not check_tiffs:
            print(f"\n  All TIFFs GOOD.")

        _pause("Press ENTER to proceed to phase 4 ...")

    # -- Phase 4: Echo Extraction -----------------------------------------
    if args.resume_from <= 4:
        # If resuming from an earlier phase, the echoes CSV is stale (computed
        # from old calibration).  Delete it so Phase 4 re-processes all TIFFs.
        echo_csv = out_dir / "phase4" / f"F{flt}_echoes.csv"
        if args.resume_from < 4 and echo_csv.exists():
            echo_csv.unlink()
            print(f"  Cleared stale {echo_csv.name} (re-running after Phase 3 update)")

        print(f"\n  {'='*55}")
        print(f"  PHASE 4: Echo Extraction")
        print(f"  {'='*55}\n")

        phase4 = run_phase4(flt, tiffs, out_dir)
        if phase4["failed"]:
            sys.exit("ERROR: Phase 4 failed; resolve the errors before continuing.")

    print_flight_summary(flt, out_dir)

    print(f"\n  Flight F{flt} processing complete.\n")


if __name__ == "__main__":
    main()
''',
    'segment_ocr': r'''"""
segment_ocr.py — 7-segment structural digit recognition for LYRA
=================================================================
Instead of pixel-level NCC, identifies which of the 7 segments are lit
by measuring ink density in spatial zones. Position-invariant by design.

Production API (imported by detect_frames.py):
    from segment_ocr import recognize_frame_structural, SEGMENT_THRESHOLD

Prototype mode (run standalone for training/evaluation):
    python codes/segment_ocr.py [--diag TIFF_ID FRAME_IDX]

Segment layout:
     -a-
    |   |
    f   b
    |   |
     -g-
    |   |
    e   c
    |   |
     -d-
"""

from pathlib import Path
import sys
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from lyra import detect_frames, display_path, output_root
from build_digit_templates import (
    get_binary_text, find_digit_col_ranges, adjust_blobs_to_7,
    TMPL_W, TMPL_H,
)

# -- Production threshold (trained on F125, universal for Tek 465 CRT) -----
SEGMENT_THRESHOLD = 0.26

# -- Prototype config (only used when run as __main__) ---------------------
FLT = 125
RAW_DIR = Path(os.environ.get("LYRA_TIFF_DIR", ROOT / f"tiff/{FLT}"))
OUT_DIR = output_root(ROOT / "output") / f"F{FLT}/step0"
INDEX_CSV = output_root(ROOT / "output") / f"F{FLT}/phase1/F{FLT}_frame_index.csv"
TRAIN_TIFFS = ["7700", "7725", "8400", "8425"]
TEST_TIFFS = ["8300"]

# -- 7-segment truth table -------------------------------------------------
#                   a  b  c  d  e  f  g
SEGMENT_TABLE = {
    "0": np.array([1, 1, 1, 1, 1, 1, 0]),
    "1": np.array([0, 1, 1, 0, 0, 0, 0]),
    "2": np.array([1, 1, 0, 1, 1, 0, 1]),
    "3": np.array([1, 1, 1, 1, 0, 0, 1]),
    "4": np.array([0, 1, 1, 0, 0, 1, 1]),
    "5": np.array([1, 0, 1, 1, 0, 1, 1]),
    "6": np.array([1, 0, 1, 1, 1, 1, 1]),
    "7": np.array([1, 1, 1, 0, 0, 0, 0]),
    "8": np.array([1, 1, 1, 1, 1, 1, 1]),
    "9": np.array([1, 1, 1, 1, 0, 1, 1]),
}
SEG_NAMES = ["a", "b", "c", "d", "e", "f", "g"]

# -- Zone definitions (fractions of DIGIT bounding box, NOT full image) -----
# Each zone: (row_start_frac, row_end_frac, col_start_frac, col_end_frac)
# These are relative to the auto-detected ink bounding box within each blob.
ZONES = {
    "a": (0.00, 0.18, 0.15, 0.85),   # top horizontal bar
    "b": (0.10, 0.48, 0.60, 1.00),   # top-right vertical
    "c": (0.52, 0.90, 0.60, 1.00),   # bottom-right vertical
    "d": (0.82, 1.00, 0.15, 0.85),   # bottom horizontal bar
    "e": (0.52, 0.90, 0.00, 0.40),   # bottom-left vertical
    "f": (0.10, 0.48, 0.00, 0.40),   # top-left vertical
    "g": (0.42, 0.58, 0.15, 0.85),   # middle horizontal bar
}

# Ink detection threshold for bounding box
BBOX_INK_THRESH = 0.05


def find_digit_bbox(digit_img, orig_aspect=None):
    """
    Find the bounding box of the actual digit ink within the template image.
    Returns (top, bottom, left, right) in pixel coordinates.

    orig_aspect: original blob width / median blob width before resize.
        If < 0.5, this is a narrow digit (e.g. "1") and horizontal zones
        should use the full image width, not the ink bbox.
    """
    H, W = digit_img.shape
    row_proj = digit_img.mean(axis=1)
    col_proj = digit_img.mean(axis=0)

    ink_rows = np.where(row_proj > BBOX_INK_THRESH)[0]
    ink_cols = np.where(col_proj > BBOX_INK_THRESH)[0]

    if len(ink_rows) < 5 or len(ink_cols) < 3:
        return 0, H, 0, W

    top = int(ink_rows[0])
    bottom = int(ink_rows[-1])

    # For narrow digits: the blob crop was narrow, then stretched to TMPL_W.
    # Use full image width so horizontal zones don't all overlap the stroke.
    if orig_aspect is not None and orig_aspect < 0.55:
        left, right = 0, W
    else:
        left = int(ink_cols[0])
        right = int(ink_cols[-1])
        # Still enforce a minimum so moderate-width digits don't collapse
        ink_w = right - left
        min_w = int(0.60 * W)
        if ink_w < min_w:
            center = (left + right) // 2
            left = max(0, center - min_w // 2)
            right = min(W, left + min_w)

    return top, bottom, left, right


def get_zone_rects(digit_img, orig_aspect=None):
    """
    Compute zone pixel rectangles relative to the actual digit bounding box.
    Returns dict {seg_name: (r0, r1, c0, c1)} and the bbox tuple.
    """
    top, bottom, left, right = find_digit_bbox(digit_img, orig_aspect)
    h = max(bottom - top, 1)
    w = max(right - left, 1)

    rects = {}
    for seg, (r0f, r1f, c0f, c1f) in ZONES.items():
        r0 = int(top + r0f * h)
        r1 = int(top + r1f * h)
        c0 = int(left + c0f * w)
        c1 = int(left + c1f * w)
        rects[seg] = (r0, r1, c0, c1)
    return rects, (top, bottom, left, right)


def measure_segment_densities(digit_img, orig_aspect=None):
    """
    Measure ink density in each of the 7 zones.
    Zones are placed relative to the auto-detected digit bounding box,
    so they adapt to where the actual content sits in the image.
    digit_img: (TMPL_H, TMPL_W) float32 binary image (1=ink, 0=bg).
    orig_aspect: blob_width / median_blob_width (< 0.55 -> narrow digit).
    Returns (densities_dict, zone_rects, bbox).
    """
    rects, bbox = get_zone_rects(digit_img, orig_aspect)
    densities = {}
    for seg in SEG_NAMES:
        r0, r1, c0, c1 = rects[seg]
        zone = digit_img[r0:r1, c0:c1]
        densities[seg] = float(zone.mean()) if zone.size > 0 else 0.0
    return densities, rects, bbox


def classify_digit(densities, threshold):
    """
    Classify a digit from its segment densities.

    threshold: float (global) or dict {seg_name: float} (per-segment).

    Approach:
    1. Threshold densities -> binary segment vector
    2. Compute Hamming distance to each digit's truth pattern
    3. Among candidates with minimum Hamming distance, use MSE as tiebreaker

    Returns (best_digit, segment_vector, min_hamming).
    """
    dens_vec = np.array([densities[s] for s in SEG_NAMES])
    if isinstance(threshold, dict):
        th_vec = np.array([threshold[s] for s in SEG_NAMES])
    else:
        th_vec = np.full(7, threshold)
    seg_vec = (dens_vec >= th_vec).astype(int)

    # Score each candidate: (hamming, mse) — sort by hamming first, mse second
    candidates = []
    for d, truth in SEGMENT_TABLE.items():
        hamming = int(np.sum(seg_vec != truth))
        mse = float(np.mean((dens_vec - truth.astype(float)) ** 2))
        candidates.append((hamming, mse, d))

    candidates.sort(key=lambda x: (x[0], x[1]))
    best_hamming, best_mse, best_digit = candidates[0]

    return best_digit, seg_vec, best_hamming


def crop_and_resize_digit(binary, s, e, med_blob_w, pad=4):
    """
    Crop a digit blob from the binary image and resize to TMPL_W x TMPL_H.
    Narrow blobs (like "1") are zero-padded to the median width BEFORE
    resizing, so the original aspect ratio is preserved and the digit
    doesn't get stretched across the full template width.

    Returns (query_img, orig_aspect) or (None, aspect) if too narrow.
    """
    crop_d = binary[:, max(0, s - pad): e + pad + 1]
    if crop_d.shape[1] < 3:
        return None, 0.0

    blob_w = e - s
    orig_aspect = blob_w / med_blob_w if med_blob_w > 0 else 1.0

    # If narrow: pad to median width so resize preserves aspect ratio
    target_w = int(med_blob_w + 2 * pad)
    actual_w = crop_d.shape[1]
    if actual_w < 0.7 * target_w:
        deficit = target_w - actual_w
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        crop_d = np.pad(crop_d, ((0, 0), (pad_left, pad_right)),
                        mode="constant", constant_values=0)

    query = np.array(Image.fromarray(
        (crop_d * 255).astype(np.uint8)).resize(
        (TMPL_W, TMPL_H), Image.LANCZOS),
        dtype=np.float32) / 255.0
    return query, orig_aspect


# ===========================================================================
# Production API — imported by detect_frames.py
# ===========================================================================

def recognize_frame_structural(frame_crop, H, fw,
                               threshold=SEGMENT_THRESHOLD):
    """
    Recognize the 7-digit CBD number from a single frame using structural
    7-segment matching.

    Same interface as recognize_frame() in build_digit_templates.py, so
    step1 can swap between NCC and structural with minimal code change.

    Parameters
    ----------
    frame_crop : np.ndarray, uint8
        Cropped frame image (full height, frame width).
    H : int
        Full TIFF image height (needed by get_binary_text).
    fw : int
        Frame width in pixels.
    threshold : float
        Segment density threshold for ON/OFF classification.

    Returns
    -------
    recognized : str or None
        7-character string (e.g. "1250465") or None if blob detection fails.
    blobs : list of (int, int)
        Column ranges of detected digit blobs.
    confidences : list of int
        Per-digit Hamming distance (0 = perfect segment match, lower = better).
    """
    binary = get_binary_text(frame_crop, H, fw)
    blobs = find_digit_col_ranges(binary)

    if len(blobs) < 5:
        return None, blobs, []

    blobs = adjust_blobs_to_7(blobs)

    if len(blobs) != 7:
        return None, blobs, []

    blob_widths = [e - s for s, e in blobs]
    med_blob_w = float(np.median(blob_widths))

    recognized = ""
    confidences = []
    for s, e in blobs:
        query, orig_aspect = crop_and_resize_digit(binary, s, e, med_blob_w)
        if query is None:
            recognized += "?"
            confidences.append(7)  # worst possible
            continue
        densities, _, _ = measure_segment_densities(query, orig_aspect)
        pred, _, hamming = classify_digit(densities, threshold)
        recognized += pred
        confidences.append(hamming)

    return recognized, blobs, confidences


# ===========================================================================
# Prototype / training functions (used when run as __main__)
# ===========================================================================

def extract_digit_blobs(tiff_path, tiff_id, gt_cbds):
    """
    Extract all digit blobs from a TIFF with ground truth.
    Returns list of dicts with keys:
        tiff_id, frame_idx, pos, true_digit, digit_img, blob_coords
    """
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W = img_norm.shape

    frames = detect_frames(img_norm)
    widths = [r - l for l, r in frames]
    med_w = float(np.median(widths))

    results = []
    for fidx, (l, r) in enumerate(frames):
        if (r - l) < 0.75 * med_w:
            continue
        true_cbd = gt_cbds.get(fidx)
        if not true_cbd:
            continue
        true_str = f"{FLT:03d}{true_cbd}"

        crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
        fw = crop_u8.shape[1]
        binary = get_binary_text(crop_u8, H, fw)
        blobs = adjust_blobs_to_7(find_digit_col_ranges(binary))

        if len(blobs) != 7:
            continue

        # Compute median blob width for aspect ratio
        blob_widths = [e - s for s, e in blobs]
        med_blob_w = float(np.median(blob_widths))

        for pos, (s, e) in enumerate(blobs):
            query, orig_aspect = crop_and_resize_digit(
                binary, s, e, med_blob_w)
            if query is None:
                continue

            true_d = true_str[pos] if pos < len(true_str) else "?"
            results.append({
                "tiff_id": tiff_id,
                "frame_idx": fidx,
                "pos": pos,
                "true_digit": true_d,
                "digit_img": query,
                "orig_aspect": orig_aspect,
                "blob_coords": (s, e),
            })

    return results


def load_ground_truth():
    gt = {}
    with open(INDEX_CSV) as f:
        for row in csv.DictReader(f):
            tid = row["tiff_id"]
            fidx = int(row["frame_idx"])
            cbd = row.get("cbd", "")
            if tid not in gt:
                gt[tid] = {}
            if cbd:
                gt[tid][fidx] = cbd
    return gt


def find_tiff(tiff_id):
    matches = list(RAW_DIR.glob(f"*_{tiff_id:>07s}_*-reel_begin_end.tiff"))
    if not matches:
        matches = list(RAW_DIR.glob(f"*_{tiff_id}_*"))
    return matches[0] if matches else None


# -- Diagnostic figure for one frame ---------------------------------------
def make_frame_diagnostic(tiff_path, tiff_id, frame_idx, gt_cbds, threshold):
    """
    Detailed diagnostic for one frame showing the structural matching
    pipeline for every digit.
    """
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W = img_norm.shape

    frames = detect_frames(img_norm)
    l, r = frames[frame_idx]
    crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
    fw = crop_u8.shape[1]

    true_cbd = gt_cbds.get(frame_idx, "????")
    true_str = f"{FLT:03d}{true_cbd}"

    binary = get_binary_text(crop_u8, H, fw)
    blobs = adjust_blobs_to_7(find_digit_col_ranges(binary))

    n_digits = len(blobs)
    blob_widths = [e - s for s, e in blobs]
    med_blob_w = float(np.median(blob_widths)) if blob_widths else 1.0

    # -- Figure: 4 rows × n_digits columns ---------------------------------
    # Row 1: digit image with zone rectangles overlaid
    # Row 2: density bar chart per segment
    # Row 3: segment vector (lit/unlit) vs truth
    # Row 4: classification result
    fig, axes = plt.subplots(4, n_digits, figsize=(3.2 * n_digits, 14))
    if n_digits == 1:
        axes = axes[:, np.newaxis]
    fig.patch.set_facecolor("white")

    # Zone colors
    seg_colors = {
        "a": "#e41a1c", "b": "#377eb8", "c": "#4daf4a",
        "d": "#ff7f00", "e": "#984ea3", "f": "#a65628", "g": "#f781bf",
    }

    ocr_str = ""
    for col, (s, e) in enumerate(blobs):
        query, orig_aspect = crop_and_resize_digit(
            binary, s, e, med_blob_w)
        if query is None:
            ocr_str += "?"
            for row in range(4):
                axes[row, col].axis("off")
            continue

        true_d = true_str[col] if col < len(true_str) else "?"
        densities, rects_d, bbox = measure_segment_densities(
            query, orig_aspect)
        pred, seg_vec, hamming = classify_digit(densities, threshold)
        ocr_str += pred
        ok = pred == true_d

        # -- Row 1: digit image + zone overlays ----------------------------
        ax = axes[0, col]
        ax.imshow(query, cmap="gray", vmin=0, vmax=1, aspect="auto")
        # Show bounding box
        bt, bb, bl, br = bbox
        bbox_rect = patches.Rectangle(
            (bl, bt), br - bl, bb - bt,
            linewidth=1.5, edgecolor="white", facecolor="none",
            linestyle="--")
        ax.add_patch(bbox_rect)
        for seg_name in SEG_NAMES:
            r0, r1, c0, c1 = rects_d[seg_name]
            color = seg_colors[seg_name]
            rect = patches.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=1.5, edgecolor=color, facecolor=color,
                alpha=0.25)
            ax.add_patch(rect)
            # Label
            cx, cy = (c0 + c1) / 2, (r0 + r1) / 2
            ax.text(cx, cy, seg_name, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
        ax.set_title(f"Pos {col}, true='{true_d}'", fontsize=9)
        ax.axis("off")

        # -- Row 2: density bar chart --------------------------------------
        ax = axes[1, col]
        bars = [densities[s] for s in SEG_NAMES]
        bar_colors = [seg_colors[s] for s in SEG_NAMES]
        ax.bar(SEG_NAMES, bars, color=bar_colors, alpha=0.7, edgecolor="k",
               linewidth=0.5)
        ax.axhline(threshold, color="red", linewidth=1.5, linestyle="--",
                   label=f"theta={threshold:.2f}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_title("Segment densities", fontsize=8)

        # -- Row 3: segment vector vs truth --------------------------------
        ax = axes[2, col]
        truth_vec = SEGMENT_TABLE.get(true_d, np.zeros(7))
        x = np.arange(7)
        width = 0.35
        ax.bar(x - width/2, seg_vec, width, color="steelblue",
               label="detected", alpha=0.8)
        ax.bar(x + width/2, truth_vec, width, color="orange",
               label="truth", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(SEG_NAMES, fontsize=7)
        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([0, 1])
        ax.legend(fontsize=6, loc="upper right")
        # Mark mismatches
        for i in range(7):
            if seg_vec[i] != truth_vec[i]:
                ax.text(i, 1.2, "X", ha="center", fontsize=10,
                        color="red", fontweight="bold")
        ax.set_title("Segments: detected vs truth", fontsize=8)

        # -- Row 4: classification result ----------------------------------
        ax = axes[3, col]
        ax.axis("off")
        result_text = (
            f"Predicted: '{pred}'\n"
            f"True:      '{true_d}'\n"
            f"Hamming:   {hamming}\n"
            f"{'CORRECT' if ok else 'WRONG'}"
        )
        color = "green" if ok else "red"
        ax.text(0.5, 0.5, result_text, transform=ax.transAxes,
                fontsize=11, fontfamily="monospace",
                ha="center", va="center", color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow" if ok else "mistyrose",
                          edgecolor=color))

    fig.suptitle(
        f"TIFF {tiff_id}, Frame {frame_idx} — "
        f"7-segment structural matching\n"
        f"True: {true_str} | OCR: {ocr_str} | "
        f"threshold={threshold:.2f}",
        fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUT_DIR / f"diag_segment_{tiff_id}_fr{frame_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Diagnostic saved -> {display_path(out_path, ROOT)}")
    return ocr_str


# -- Per-segment threshold learning ----------------------------------------
def learn_per_segment_thresholds(all_digits):
    """
    Learn optimal threshold for each segment independently.
    For each segment, collect all (density, should_be_lit) pairs from
    training data, then sweep to find the threshold that minimizes errors.

    Returns dict {seg_name: optimal_threshold}.
    """
    # Collect densities grouped by segment and ON/OFF status
    seg_data = {s: {"on": [], "off": []} for s in SEG_NAMES}

    for d in all_digits:
        true_d = d["true_digit"]
        if true_d not in SEGMENT_TABLE:
            continue
        truth = SEGMENT_TABLE[true_d]
        densities, _, _ = measure_segment_densities(
            d["digit_img"], d.get("orig_aspect"))
        for i, seg in enumerate(SEG_NAMES):
            if truth[i] == 1:
                seg_data[seg]["on"].append(densities[seg])
            else:
                seg_data[seg]["off"].append(densities[seg])

    print("\n  Per-segment density statistics:")
    print(f"  {'seg':>3}  {'ON mean':>8}  {'ON std':>7}  "
          f"{'OFF mean':>9}  {'OFF std':>8}  {'best_th':>8}  {'sep':>5}")
    print("  " + "-" * 60)

    per_seg_th = {}
    for seg in SEG_NAMES:
        on_vals = np.array(seg_data[seg]["on"])
        off_vals = np.array(seg_data[seg]["off"])

        on_mean = on_vals.mean() if len(on_vals) else 0.5
        on_std = on_vals.std() if len(on_vals) else 0.1
        off_mean = off_vals.mean() if len(off_vals) else 0.1
        off_std = off_vals.std() if len(off_vals) else 0.1

        # Sweep threshold for this segment
        best_th_s = 0.20
        best_err = len(on_vals) + len(off_vals)
        for th in np.arange(0.05, 0.80, 0.01):
            err = (np.sum(on_vals < th) +     # false negatives
                   np.sum(off_vals >= th))     # false positives
            if err < best_err:
                best_err = err
                best_th_s = th

        # Separation: gap between on/off means in units of pooled std
        pooled_std = max(0.01, np.sqrt((on_std**2 + off_std**2) / 2))
        sep = (on_mean - off_mean) / pooled_std

        per_seg_th[seg] = best_th_s
        print(f"  {seg:>3}  {on_mean:>8.3f}  {on_std:>7.3f}  "
              f"{off_mean:>9.3f}  {off_std:>8.3f}  {best_th_s:>8.2f}  "
              f"{sep:>5.1f}")

    return per_seg_th


def sweep_threshold(all_digits):
    """
    Find optimal global threshold AND per-segment thresholds.
    Returns (best_global_threshold, per_seg_thresholds, accuracy_at_best).
    """
    # First: learn per-segment thresholds
    per_seg_th = learn_per_segment_thresholds(all_digits)

    # Evaluate per-segment approach
    correct_perseg = 0
    total = len(all_digits)
    for d in all_digits:
        densities, _, _ = measure_segment_densities(
            d["digit_img"], d.get("orig_aspect"))
        pred, _, _ = classify_digit(densities, per_seg_th)
        if pred == d["true_digit"]:
            correct_perseg += 1
    acc_perseg = correct_perseg / total if total > 0 else 0

    # Also sweep global for comparison
    thresholds = np.arange(0.05, 0.60, 0.01)
    best_acc_global = 0
    best_th_global = 0.20
    for th in thresholds:
        correct = 0
        for d in all_digits:
            densities, _, _ = measure_segment_densities(
                d["digit_img"], d.get("orig_aspect"))
            pred, _, _ = classify_digit(densities, th)
            if pred == d["true_digit"]:
                correct += 1
        acc = correct / total if total > 0 else 0
        if acc > best_acc_global:
            best_acc_global = acc
            best_th_global = th

    print(f"\n  Global best: theta={best_th_global:.2f} -> {best_acc_global:.1%}")
    print(f"  Per-segment:  -> {acc_perseg:.1%}")

    if acc_perseg >= best_acc_global:
        print("  Using per-segment thresholds")
        return best_th_global, per_seg_th, acc_perseg
    else:
        print("  Using global threshold")
        return best_th_global, best_th_global, best_acc_global


# -- Confusion analysis ----------------------------------------------------
def confusion_analysis(all_digits, threshold):
    """Print per-digit accuracy and common confusions."""
    from collections import Counter

    digit_counts = Counter()
    digit_correct = Counter()
    confusions = []

    for d in all_digits:
        densities, _, _ = measure_segment_densities(
            d["digit_img"], d.get("orig_aspect"))
        pred, seg_vec, hamming = classify_digit(densities, threshold)
        true_d = d["true_digit"]
        digit_counts[true_d] += 1
        if pred == true_d:
            digit_correct[true_d] += 1
        else:
            confusions.append({
                "true": true_d, "pred": pred,
                "hamming": hamming,
                "tiff": d["tiff_id"], "frame": d["frame_idx"],
                "pos": d["pos"],
                "densities": densities,
                "seg_vec": seg_vec,
            })

    th_label = (f"{threshold:.2f}" if isinstance(threshold, float)
                else "per-segment")
    print(f"\n  Per-digit accuracy (threshold={th_label}):")
    print(f"  {'digit':>5}  {'correct':>7}  {'total':>5}  {'acc':>6}")
    print("  " + "-" * 30)
    for digit in sorted(digit_counts.keys()):
        c = digit_correct[digit]
        t = digit_counts[digit]
        acc = c / t if t > 0 else 0
        marker = "" if acc == 1.0 else " <- ERRORS"
        print(f"  {digit:>5}  {c:>7}  {t:>5}  {acc:>6.1%}{marker}")

    total_correct = sum(digit_correct.values())
    total = sum(digit_counts.values())
    print(f"  {'ALL':>5}  {total_correct:>7}  {total:>5}  "
          f"{total_correct/total:.1%}")

    if confusions:
        print(f"\n  Confusions ({len(confusions)} errors):")
        for c in confusions:
            dens_str = " ".join(
                f"{s}={c['densities'][s]:.2f}" for s in SEG_NAMES)
            print(f"    TIFF {c['tiff']} fr{c['frame']} pos{c['pos']}: "
                  f"true='{c['true']}' pred='{c['pred']}' "
                  f"hamming={c['hamming']}")
            print(f"      densities: {dens_str}")
            print(f"      seg_vec:   {c['seg_vec'].tolist()}")

    return confusions


# -- Test evaluation with sequential constraint --------------------------
def recognize_tiff_structural(tiff_path, tiff_id, gt_cbds, threshold):
    """
    Run structural matching on all complete frames of a TIFF.
    Returns per-frame results for sequential constraint evaluation.
    """
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(tiff_path), dtype=np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    H, W = img_norm.shape

    frames = detect_frames(img_norm)
    widths = [r - l for l, r in frames]
    med_w = float(np.median(widths))

    frame_results = []
    for fidx, (l, r) in enumerate(frames):
        if (r - l) < 0.75 * med_w:
            continue  # partial frame
        true_cbd = gt_cbds.get(fidx)
        true_str = f"{FLT:03d}{true_cbd}" if true_cbd else None

        crop_u8 = (img_norm[:, l:r+1] * 255).clip(0, 255).astype(np.uint8)
        fw = crop_u8.shape[1]
        binary = get_binary_text(crop_u8, H, fw)
        blobs = adjust_blobs_to_7(find_digit_col_ranges(binary))

        if len(blobs) != 7:
            frame_results.append({
                "frame_idx": fidx, "true_str": true_str,
                "raw_ocr": None, "n_blobs": len(blobs),
            })
            continue

        blob_widths = [e - s for s, e in blobs]
        med_blob_w = float(np.median(blob_widths))

        ocr_str = ""
        for pos, (s, e) in enumerate(blobs):
            query, orig_aspect = crop_and_resize_digit(
                binary, s, e, med_blob_w)
            if query is None:
                ocr_str += "?"
                continue
            densities, _, _ = measure_segment_densities(query, orig_aspect)
            pred, _, _ = classify_digit(densities, threshold)
            ocr_str += pred

        frame_results.append({
            "frame_idx": fidx, "true_str": true_str,
            "raw_ocr": ocr_str, "n_blobs": len(blobs),
        })

    return frame_results


def evaluate_test_tiff(tiff_id, threshold):
    """
    Full evaluation on a test TIFF: raw accuracy + sequential constraint.
    Returns summary dict.
    """
    from build_digit_templates import apply_sequential_constraint

    gt = load_ground_truth()
    tiff_path = find_tiff(tiff_id)
    if not tiff_path:
        print(f"  ERROR: TIFF {tiff_id} not found")
        return None
    gt_cbds = gt.get(tiff_id, {})
    if not gt_cbds:
        print(f"  ERROR: no ground truth for TIFF {tiff_id}")
        return None

    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION: TIFF {tiff_id}")
    print(f"{'='*70}")

    frame_results = recognize_tiff_structural(
        tiff_path, tiff_id, gt_cbds, threshold)

    # -- Raw accuracy -----------------------------------------------------
    print(f"\n  Raw structural matching results:")
    print(f"  {'Fr':>3}  {'Raw OCR':>10}  {'True':>10}  {'Match':>6}")
    print("  " + "-" * 40)

    total_digits = 0
    correct_digits = 0
    total_frames = 0
    correct_frames = 0

    frame_indices = []
    raw_reads = []

    for fr in frame_results:
        fidx = fr["frame_idx"]
        raw = fr["raw_ocr"]
        true_s = fr["true_str"]

        if raw is None:
            print(f"  {fidx:>3}  {'(no blobs)':>10}  {true_s or '?':>10}  {'—':>6}")
            continue

        frame_indices.append(fidx)
        raw_reads.append(raw)

        if true_s:
            total_frames += 1
            frame_ok = (raw == true_s)
            if frame_ok:
                correct_frames += 1
            for i, (r, t) in enumerate(zip(raw, true_s)):
                total_digits += 1
                if r == t:
                    correct_digits += 1
            marker = "OK" if frame_ok else "MISS"
        else:
            marker = "no GT"

        print(f"  {fidx:>3}  {raw:>10}  {true_s or '?':>10}  {marker:>6}")

    raw_digit_acc = correct_digits / total_digits if total_digits > 0 else 0
    raw_frame_acc = correct_frames / total_frames if total_frames > 0 else 0

    print(f"\n  Raw digit accuracy:  {correct_digits}/{total_digits} "
          f"= {raw_digit_acc:.1%}")
    print(f"  Raw frame accuracy:  {correct_frames}/{total_frames} "
          f"= {raw_frame_acc:.1%}")

    # -- Sequential constraint --------------------------------------------
    print(f"\n  Applying sequential constraint (flight={FLT}) ...")
    corrected, n_anchors = apply_sequential_constraint(
        frame_indices, raw_reads, flight=FLT)

    print(f"  Anchor agreement: {n_anchors}/{len(frame_indices)} "
          f"raw reads match best-fit sequence")

    print(f"\n  {'Fr':>3}  {'Raw':>10}  {'Corrected':>10}  {'True':>10}  "
          f"{'Status':>8}")
    print("  " + "-" * 55)

    corrected_digits = 0
    corrected_frames = 0
    total_digits_corr = 0
    total_frames_corr = 0

    for i, (fidx, raw, corr) in enumerate(
            zip(frame_indices, raw_reads, corrected)):
        # Find ground truth for this frame
        true_s = None
        for fr in frame_results:
            if fr["frame_idx"] == fidx:
                true_s = fr["true_str"]
                break

        if true_s:
            total_frames_corr += 1
            frame_ok = (corr == true_s)
            if frame_ok:
                corrected_frames += 1
            for j, (c, t) in enumerate(zip(corr, true_s)):
                total_digits_corr += 1
                if c == t:
                    corrected_digits += 1

            raw_ok = (raw == true_s)
            if frame_ok and not raw_ok:
                status = "FIXED"
            elif frame_ok:
                status = "OK"
            elif raw_ok and not frame_ok:
                status = "BROKEN"
            else:
                status = "STILL BAD"
        else:
            status = "no GT"

        print(f"  {fidx:>3}  {raw:>10}  {corr:>10}  {true_s or '?':>10}  "
              f"{status:>8}")

    corr_digit_acc = (corrected_digits / total_digits_corr
                      if total_digits_corr > 0 else 0)
    corr_frame_acc = (corrected_frames / total_frames_corr
                      if total_frames_corr > 0 else 0)

    print(f"\n  After sequential constraint:")
    print(f"    Digit accuracy:  {corrected_digits}/{total_digits_corr} "
          f"= {corr_digit_acc:.1%}")
    print(f"    Frame accuracy:  {corrected_frames}/{total_frames_corr} "
          f"= {corr_frame_acc:.1%}")

    # -- Summary ----------------------------------------------------------
    print(f"\n  SUMMARY for TIFF {tiff_id}:")
    print(f"    Raw:       {raw_digit_acc:.1%} digit, "
          f"{raw_frame_acc:.1%} frame")
    print(f"    Corrected: {corr_digit_acc:.1%} digit, "
          f"{corr_frame_acc:.1%} frame")
    improvement = corr_digit_acc - raw_digit_acc
    if improvement > 0:
        print(f"    Sequential constraint improved digit accuracy by "
              f"+{improvement:.1%}")
    elif improvement < 0:
        print(f"    WARNING: Sequential constraint DECREASED digit accuracy "
              f"by {improvement:.1%}")
    else:
        print(f"    Sequential constraint had no effect on digit accuracy")

    return {
        "tiff_id": tiff_id,
        "raw_digit_acc": raw_digit_acc,
        "raw_frame_acc": raw_frame_acc,
        "corr_digit_acc": corr_digit_acc,
        "corr_frame_acc": corr_frame_acc,
        "n_anchors": n_anchors,
        "n_frames": total_frames,
    }


# -- Main -------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()

    # Parse optional diagnostic target
    diag_tiff = "8400"
    diag_frame = 7
    if "--diag" in sys.argv:
        idx = sys.argv.index("--diag")
        if idx + 2 < len(sys.argv):
            diag_tiff = sys.argv[idx + 1]
            diag_frame = int(sys.argv[idx + 2])

    # -- Step 1: extract all digit blobs from training TIFFs ---------------
    print("Extracting digit blobs from training TIFFs ...")
    all_digits = []
    for tid in TRAIN_TIFFS:
        tiff_path = find_tiff(tid)
        if not tiff_path:
            print(f"  WARNING: TIFF {tid} not found — skipping")
            continue
        gt_cbds = gt.get(tid, {})
        if not gt_cbds:
            print(f"  WARNING: no ground truth for TIFF {tid} — skipping")
            continue
        print(f"  TIFF {tid}: ", end="", flush=True)
        digits = extract_digit_blobs(tiff_path, tid, gt_cbds)
        all_digits.extend(digits)
        print(f"{len(digits)} digit blobs extracted")

    print(f"\n  Total training digits: {len(all_digits)}")

    # -- Step 2: learn thresholds --------------------------------------------
    best_th_global, best_th, best_acc = sweep_threshold(all_digits)

    # -- Step 3: confusion analysis at best threshold ----------------------
    confusions = confusion_analysis(all_digits, best_th)

    # -- Step 4: diagnostic figure for one frame ---------------------------
    print(f"\nGenerating diagnostic for TIFF {diag_tiff} frame {diag_frame}...")
    tiff_path = find_tiff(diag_tiff)
    gt_cbds = gt.get(diag_tiff, {})
    ocr = make_frame_diagnostic(
        tiff_path, diag_tiff, diag_frame, gt_cbds, best_th)

    print(f"\nTraining accuracy: {best_acc:.1%}")

    # -- Step 5: TEST on held-out TIFF(s) ---------------------------------
    print(f"\n{'#'*70}")
    print(f"  TESTING ON HELD-OUT TIFF(s): {TEST_TIFFS}")
    print(f"{'#'*70}")

    test_results = []
    for tid in TEST_TIFFS:
        result = evaluate_test_tiff(tid, best_th)
        if result:
            test_results.append(result)

    # -- Final comparison -------------------------------------------------
    if test_results:
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS")
        print(f"{'='*70}")
        print(f"  Training (TIFFs {', '.join(TRAIN_TIFFS)}):")
        print(f"    Raw digit accuracy: {best_acc:.1%}")
        print(f"\n  Test (TIFFs {', '.join(TEST_TIFFS)}):")
        for r in test_results:
            print(f"    TIFF {r['tiff_id']}: raw={r['raw_digit_acc']:.1%} -> "
                  f"corrected={r['corr_digit_acc']:.1%} "
                  f"(frame: {r['raw_frame_acc']:.1%} -> "
                  f"{r['corr_frame_acc']:.1%})")
            print(f"      Anchors: {r['n_anchors']}/{r['n_frames']}")

    print("\nDone.")
''',

}

import importlib.abc as _importlib_abc
import importlib.util as _importlib_util
import subprocess as _subprocess
import sys as _sys

_BUNDLE_PATH = Path(__file__).resolve()
_SCRIPT_DIR = _BUNDLE_PATH.parent


class _EmbeddedSubprocess:
    def __init__(self, active_script: str | None = None):
        self.active_script = active_script

    def run(self, command, *args, **kwargs):
        if not isinstance(command, (str, bytes, os.PathLike)):
            command = list(command)
            if len(command) > 1:
                script_path = Path(command[1])
                script_name = script_path.stem
                if script_name in _EMBEDDED_SCRIPT_SOURCES:
                    command[1:2] = [
                        str(_BUNDLE_PATH), "--_lyra-script", script_name,
                    ]
                elif script_path.resolve() == _BUNDLE_PATH and self.active_script:
                    command[1:2] = [
                        str(_BUNDLE_PATH), "--_lyra-script", self.active_script,
                    ]
        return _subprocess.run(command, *args, **kwargs)


class _EmbeddedSourceLoader(_importlib_abc.Loader):
    def __init__(self, name: str):
        self.name = name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = _EMBEDDED_SCRIPT_SOURCES[self.name]
        source_path = _SCRIPT_DIR / f"{self.name}.py"
        module.__file__ = str(source_path)
        module.__package__ = ""
        exec(compile(source, str(source_path), "exec"), module.__dict__)
        if "subprocess" in module.__dict__:
            module.subprocess = _EmbeddedSubprocess(self.name)


class _EmbeddedSourceFinder(_importlib_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _EMBEDDED_SCRIPT_SOURCES:
            return _importlib_util.spec_from_loader(
                fullname, _EmbeddedSourceLoader(fullname))
        return None


_sys.meta_path.insert(0, _EmbeddedSourceFinder())
_sys.modules["lyra"] = _sys.modules[__name__]


def _execute_embedded_script(name: str, arguments: list[str]) -> None:
    if name not in _EMBEDDED_SCRIPT_SOURCES:
        available = ", ".join(sorted(_EMBEDDED_SCRIPT_SOURCES))
        raise SystemExit(f"Unknown LYRA tool '{name}'. Available tools: {available}")

    previous_argv = _sys.argv
    previous_subprocess = _sys.modules.get("subprocess")
    _sys.argv = [str(_SCRIPT_DIR / f"{name}.py"), *arguments]
    _sys.modules["subprocess"] = _EmbeddedSubprocess(name)
    namespace = {
        "__name__": "__main__",
        "__file__": str(_SCRIPT_DIR / f"{name}.py"),
        "__package__": None,
    }
    try:
        exec(compile(_EMBEDDED_SCRIPT_SOURCES[name], namespace["__file__"], "exec"),
             namespace)
    finally:
        _sys.argv = previous_argv
        if previous_subprocess is None:
            _sys.modules.pop("subprocess", None)
        else:
            _sys.modules["subprocess"] = previous_subprocess


def main(argv: list[str] | None = None) -> None:
    arguments = list(_sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"--tool", "--_lyra-script"}:
        if len(arguments) < 2:
            raise SystemExit(f"{arguments[0]} requires a LYRA script name.")
        _execute_embedded_script(arguments[1], arguments[2:])
        return

    from run_flight import main as run_flight_main

    run_flight_main(arguments)


if __name__ == "__main__":
    main()
