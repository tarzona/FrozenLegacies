import numpy as np
from URSA_Angelo.functions import echo_detection as ed


def make_time_series(times_us, peaks):
    # create power_vals with low floor and Gaussian peaks at given times
    power = np.full_like(times_us, -100.0, dtype=float)
    for t, amp in peaks:
        power += amp * np.exp(-0.5 * ((times_us - t) / 0.2) ** 2)
    return power


def test_detect_double_tx_using_absolute_window():
    times = np.linspace(0, 6, 601)  # 0..6 μs, 0.01 step
    # create two transmitter components at 0.8 and 2.1 us
    power = make_time_series(times, [(0.8, 30.0), (2.1, 28.0)])
    # set config to a tx absolute window that contains both peaks
    cfg = {"processing_params": {"tx_search_start_us": 0.0, "tx_search_end_us": 3.0}}
    res = ed.detect_double_transmitter_pulse(None, None, power, times, cfg)
    assert isinstance(res, dict)
    assert res.get("recommended_tx_idx") is not None
    # recommended index should correspond approximately to later component (2.1us)
    rec_idx = res["recommended_tx_idx"]
    assert abs(times[rec_idx] - 2.1) < 0.1


def test_surface_detection_absolute_window_primary():
    times = np.linspace(0, 40, 4001)
    # create a surface candidate at 9.5 us with strong power
    power = make_time_series(times, [(9.5, 30.0), (30.0, 18.0)])
    cfg = {"processing_params": {"surface_search_start_us": 9.0, "surface_search_end_us": 11.0, "surface_peak_height_noise_std": 1.0, "noise_floor_window_start_us": 5.0, "noise_floor_window_end_us": 6.2}}
    tx_analysis = {"is_double_pulse": False, "recommended_tx_idx": 0}
    idx = ed.detect_surface_echo_adaptive(power, times, tx_analysis, cfg)
    assert idx is not None
    assert abs(times[idx] - 9.5) < 0.2


def test_bed_detection_absolute_window_primary():
    times = np.linspace(0, 80, 8001)
    # surface at 10us, bed at 35us
    power = make_time_series(times, [(10.0, 30.0), (35.0, 25.0)])
    cfg = {"processing_params": {"bed_search_start_us": 33.0, "bed_search_end_us": 37.0, "bed_min_time_after_surface_us": 5.0, "bed_min_power_db": -80}}
    # locate surface index
    surf_idx = np.argmax(np.exp(-0.5 * ((times - 10.0) / 0.2) ** 2) * 30.0)
    idx = ed.detect_bed_echo(power, times, surf_idx, px_per_us=1.0, config=cfg)
    assert idx is not None
    assert abs(times[idx] - 35.0) < 0.2


def test_bed_absolute_window_prefers_highest_power():
    times = np.linspace(0, 80, 8001)
    # create surface at 10us and two bed-like peaks inside the window at 23.8 and 25.0 us
    power = make_time_series(times, [(10.0, 30.0), (23.8, 20.0), (25.0, 26.0)])
    cfg = {"processing_params": {"bed_search_start_us": 23.0, "bed_search_end_us": 26.0, "bed_min_time_after_surface_us": 5.0, "bed_min_power_db": -80, "bed_select_mode": "highest_peak"}}
    surf_idx = np.argmax(np.exp(-0.5 * ((times - 10.0) / 0.2) ** 2) * 30.0)
    idx = ed.detect_bed_echo(power, times, surf_idx, px_per_us=1.0, config=cfg)
    assert idx is not None
    # should pick 25.0us (higher amplitude) not 23.8us
    assert abs(times[idx] - 25.0) < 0.2


def test_bed_absolute_window_ignores_min_time():
    times = np.linspace(0, 80, 8001)
    # surface at 10us and bed at 12us (inside user absolute window) --
    # bed_min_time_after_surface_us will be set larger than 12 in config
    power = make_time_series(times, [(10.0, 30.0), (12.0, 20.0)])
    cfg = {
        "processing_params": {
            "bed_search_start_us": 11.5,
            "bed_search_end_us": 12.5,
            # Set very large min time so an absolute candidate would normally fail
            "bed_min_time_after_surface_us": 40.0,
            "bed_min_power_db": -80,
            "bed_select_mode": "highest_peak",
        }
    }
    surf_idx = np.argmax(np.exp(-0.5 * ((times - 10.0) / 0.2) ** 2) * 30.0)
    idx = ed.detect_bed_echo(power, times, surf_idx, px_per_us=1.0, config=cfg)
    assert idx is not None
    # even though bed_min_time_after_surface_us is large, absolute window should win
    assert abs(times[idx] - 12.0) < 0.2


def test_bed_absolute_window_most_prominent_mode():
    times = np.linspace(0, 80, 8001)
    # Create two peaks where the slightly lower amplitude peak is more prominent in shape
    power = np.full_like(times, -100.0, dtype=float)
    # 23.8 is tall but narrow; 25.0 is slightly lower in amplitude but broader -> larger prominence
    power += 20.0 * np.exp(-0.5 * ((times - 23.8) / 0.05) ** 2)  # narrow
    power += 18.0 * np.exp(-0.5 * ((times - 25.0) / 0.3) ** 2)   # wider
    # add surface at 10us
    power += 30.0 * np.exp(-0.5 * ((times - 10.0) / 0.2) ** 2)
    cfg = {"processing_params": {"bed_search_start_us": 23.0, "bed_search_end_us": 26.0, "bed_min_time_after_surface_us": 5.0, "bed_min_power_db": -80, "bed_select_mode": "most_prominent"}}
    surf_idx = np.argmax(np.exp(-0.5 * ((times - 10.0) / 0.2) ** 2) * 30.0)
    idx = ed.detect_bed_echo(power, times, surf_idx, px_per_us=1.0, config=cfg)
    assert idx is not None
    # should pick the broader peak at 25.0us (more prominent) when in most_prominent mode
    assert abs(times[idx] - 25.0) < 0.25


if __name__ == '__main__':
    test_detect_double_tx_using_absolute_window()
    test_surface_detection_absolute_window_primary()
    test_bed_detection_absolute_window_primary()
    print('All tests passed')
