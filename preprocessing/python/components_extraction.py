"""
Feature Extraction for Mind Wandering Detection
Based on Grandchamp et al. methodology

Extracts:
1. Alpha power (8.5-12 Hz) at PO7, Pz, PO8, Fz
2. Theta power (4-8 Hz) at PO7, Pz, PO8, Fz  
3. ISPC (Inter-Site Phase Clustering) between electrode pairs

Adapted for 10-second epochs instead of 400ms windows
"""

import numpy as np
import mne
from scipy import signal
from pathlib import Path
import pandas as pd


def create_plateau_filter(sfreq, freq_range, transition_width=0.2):
    """
    Create plateau-shaped FIR filter using firls (least-squares)

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz)
    freq_range : tuple
        (low_freq, high_freq) for the passband
    transition_width : float
        Width of transition as proportion of band (default: 0.2 = 20%)

    Returns
    -------
    filter_kernel : array
        FIR filter coefficients

    Notes
    -----
    Filter design:
    - Passband: freq_range (gain = 1.0)
    - Transition bands: 20% of bandwidth
    - Stopband: 0 Hz and Nyquist (gain = 0.0)
    - Ensures at least 3 cycles of lowest frequency

    Example
    -------
    >>> # Alpha filter 8.5-12 Hz at 512 Hz sampling
    >>> kernel = create_plateau_filter(512, (8.5, 12))
    """

    low_freq, high_freq = freq_range
    bandwidth = high_freq - low_freq

    # Calculate transition widths (20% of bandwidth)
    transition = bandwidth * transition_width

    # Define frequency bands (normalized to Nyquist)
    nyquist = sfreq / 2.0

    # Frequency points for firls:
    # [0, low_trans_start, low_freq, high_freq, high_trans_end, nyquist]
    freqs = [
        0,                           # DC
        low_freq - transition,       # Lower transition start
        low_freq,                    # Passband start
        high_freq,                   # Passband end
        high_freq + transition,      # Upper transition end
        nyquist                      # Nyquist frequency
    ]

    # Ensure frequencies are within valid range
    freqs = np.clip(freqs, 0, nyquist)

    # Desired gains (plateau shape)
    gains = [0, 0, 1, 1, 0, 0]

    # Calculate filter length: at least 3 cycles of lowest frequency
    min_length = int(np.ceil(3 * sfreq / low_freq))
    # Make it odd for zero-phase filtering
    if min_length % 2 == 0:
        min_length += 1

    # Design filter using least-squares (equivalent to MATLAB firls)
    filter_kernel = signal.firls(min_length, freqs, gains, fs=sfreq)

    # Verify SSE (Sum of Squared Errors) < 1
    # Compute frequency response
    w, h = signal.freqz(filter_kernel, worN=8000, fs=sfreq)

    # Ideal response
    ideal = np.zeros_like(w)
    ideal[(w >= low_freq) & (w <= high_freq)] = 1.0

    # SSE
    sse = np.sum((np.abs(h) - ideal) ** 2)

    if sse >= 1.0:
        print(f"⚠️  Warning: Filter SSE = {sse:.4f} (should be < 1.0)")
        print(f"   Consider increasing filter length or adjusting parameters")
    else:
        print(f"✓ Filter designed: {low_freq}-{high_freq} Hz, SSE = {sse:.6f}")

    return filter_kernel


def apply_hilbert_transform(data, filter_kernel):
    """
    Apply band-pass filter and Hilbert transform

    Parameters
    ----------
    data : array, shape (n_epochs, n_channels, n_times)
        EEG data
    filter_kernel : array
        FIR filter coefficients

    Returns
    -------
    analytic_signal : array (complex), shape (n_epochs, n_channels, n_times)
        Hilbert-transformed data (complex analytical signal)

    Notes
    -----
    The analytical signal allows computation of:
    - Instantaneous power: |signal|^2
    - Instantaneous phase: angle(signal)
    """

    n_epochs, n_channels, n_times = data.shape

    # Initialize output (complex)
    analytic_signal = np.zeros(data.shape, dtype=complex)

    # Process each epoch and channel
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Extract signal
            sig = data[epoch_idx, ch_idx, :]

            # Apply FIR filter (zero-phase filtering)
            filtered = signal.filtfilt(filter_kernel, 1.0, sig)

            # Hilbert transform
            analytic = signal.hilbert(filtered)

            analytic_signal[epoch_idx, ch_idx, :] = analytic

    return analytic_signal


def compute_power(analytic_signal):
    """
    Compute instantaneous power from analytical signal

    Parameters
    ----------
    analytic_signal : array (complex)
        Hilbert-transformed data

    Returns
    -------
    power : array (real)
        Instantaneous power at each time point

    Notes
    -----
    Power = |signal|^2 = real^2 + imag^2
    """

    return np.abs(analytic_signal) ** 2


def compute_phase(analytic_signal):
    """
    Compute instantaneous phase from analytical signal

    Parameters
    ----------
    analytic_signal : array (complex)
        Hilbert-transformed data

    Returns
    -------
    phase : array (real)
        Instantaneous phase in radians [-π, π]
    """

    return np.angle(analytic_signal)


def compute_ispc(analytic_x, analytic_y, axis=-1):
    # MATLAB: datadiff = data1.*conj(data2)
    datadiff = analytic_x * np.conj(analytic_y)

    # MATLAB: angle(datadiff)
    phase_diff = np.angle(datadiff)

    # MATLAB: exp(1i*angle(...))
    unit_vectors = np.exp(1j * phase_diff)

    # MATLAB: mean(..., 2)
    mean_vector = np.mean(unit_vectors, axis=axis)

    # MATLAB: abs(...)
    ispc = np.abs(mean_vector)

    return ispc

def extract_features_from_epochs(epochs, channels_of_interest=None,
                                 alpha_range=(8.5, 12), theta_range=(4, 8),
                                 baseline_window=None, signal_window=None):
    """
    Extract all features from epochs

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs (should be -10 to 0 seconds)
    channels_of_interest : dict, optional
        Mapping of channel names to use
        Default: {'PO7': 'A10', 'Pz': 'A19', 'PO8': 'B7', 'Fz': 'C21'}
    alpha_range : tuple
        (low, high) frequencies for alpha band (default: 8.5-12 Hz)
    theta_range : tuple
        (low, high) frequencies for theta band (default: 4-8 Hz)
    baseline_window : tuple, optional
        (tmin, tmax) in seconds for baseline period
        Default: (-10, -9) - first second of the 10-second epoch
    signal_window : tuple, optional
        (tmin, tmax) in seconds for signal period
        Default: (-1, 0) - last second before stimulus

    Returns
    -------
    features_df : pd.DataFrame
        Features for each epoch with columns:
        - epoch_idx: Epoch index
        - label: On-Task (0) or Mind Wandering (1)
        - alpha_power_PO7_baseline: Alpha power at PO7 during baseline
        - alpha_power_PO7_signal: Alpha power at PO7 during signal
        - ... (similar for Pz, PO8, Fz)
        - theta_power_PO7_baseline: Theta power at PO7 during baseline
        - ... (similar for all channels)
        - ispc_PO7_Pz_alpha_baseline: ISPC between PO7-Pz alpha during baseline
        - ... (similar for all pairs and bands)

    Notes
    -----
    Adaptation from article:
    - Article: baseline (-400 to 0 ms), signal (0 to 600 ms)
    - Here: baseline (first second), signal (last second)
    - Rationale: Capture changes over 10-second meditation period
    """

    # Default channels (10-20 system equivalents)
    if channels_of_interest is None:
        channels_of_interest = {
            'PO7': 'A10',   # Left posterior occipital
            'Pz': 'A19',    # Parietal midline
            'PO8': 'B7',    # Right posterior occipital
            'Fz': 'C21'     # Frontal midline
        }

    # Default time windows
    if baseline_window is None:
        baseline_window = (-10, -9)  # First second
    if signal_window is None:
        signal_window = (-1, 0)      # Last second before stimulus

    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    print(f"Epochs: {len(epochs)}")
    print(f"Channels: {list(channels_of_interest.keys())}")
    print(f"Alpha band: {alpha_range[0]}-{alpha_range[1]} Hz")
    print(f"Theta band: {theta_range[0]}-{theta_range[1]} Hz")
    print(f"Baseline window: {baseline_window[0]} to {baseline_window[1]} s")
    print(f"Signal window: {signal_window[0]} to {signal_window[1]} s")
    print("="*70)

    # Get sampling frequency
    sfreq = epochs.info['sfreq']

    # Map channel names (handle different naming conventions)
    ch_names_mapping = {}
    for standard_name, biosemi_name in channels_of_interest.items():
        if biosemi_name in epochs.ch_names:
            ch_names_mapping[standard_name] = biosemi_name
        elif standard_name in epochs.ch_names:
            ch_names_mapping[standard_name] = standard_name
        else:
            print(f"⚠️  Warning: Channel {standard_name}/{biosemi_name} not found")

    if len(ch_names_mapping) == 0:
        raise ValueError("No channels of interest found in epochs")

    print(f"\n✓ Using channels: {ch_names_mapping}")

    # Select channels
    epochs_subset = epochs.copy().pick_channels([ch_names_mapping[k] for k in ch_names_mapping.keys()])

    # Get data
    data = epochs_subset.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs_subset.times

    # Get time indices for baseline and signal windows
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    signal_mask = (times >= signal_window[0]) & (times <= signal_window[1])

    print(f"\nBaseline samples: {np.sum(baseline_mask)}")
    print(f"Signal samples: {np.sum(signal_mask)}")

    # Create filters
    print("\n--- Creating Filters ---")
    alpha_filter = create_plateau_filter(sfreq, alpha_range)
    theta_filter = create_plateau_filter(sfreq, theta_range)

    # Apply filters and Hilbert transform
    print("\n--- Applying Filters and Hilbert Transform ---")
    print("Processing alpha band...")
    alpha_analytic = apply_hilbert_transform(data, alpha_filter)

    print("Processing theta band...")
    theta_analytic = apply_hilbert_transform(data, theta_filter)

    # Compute power
    print("\n--- Computing Power ---")
    alpha_power = compute_power(alpha_analytic)
    theta_power = compute_power(theta_analytic)

    # Compute phase
    print("--- Computing Phase ---")
    alpha_phase = compute_phase(alpha_analytic)
    theta_phase = compute_phase(theta_analytic)

    # Initialize features list
    features_list = []

    # Channel names (ordered)
    ch_list = list(ch_names_mapping.keys())

    # Extract features for each epoch
    print("\n--- Extracting Features per Epoch ---")
    for epoch_idx in range(len(epochs)):
        if epoch_idx % 50 == 0:
            print(f"Processing epoch {epoch_idx}/{len(epochs)}...")

        features = {
            'epoch_idx': epoch_idx,
            'label': epochs.events[epoch_idx, 2]  # 0 or 1
        }

        # === POWER FEATURES ===
        for ch_idx, ch_name in enumerate(ch_list):
            # Alpha power
            features[f'alpha_power_{ch_name}_baseline'] = np.mean(
                alpha_power[epoch_idx, ch_idx, baseline_mask]
            )
            features[f'alpha_power_{ch_name}_signal'] = np.mean(
                alpha_power[epoch_idx, ch_idx, signal_mask]
            )

            # Theta power
            features[f'theta_power_{ch_name}_baseline'] = np.mean(
                theta_power[epoch_idx, ch_idx, baseline_mask]
            )
            features[f'theta_power_{ch_name}_signal'] = np.mean(
                theta_power[epoch_idx, ch_idx, signal_mask]
            )

        # === ISPC FEATURES ===
        # All pairs of channels
        for i, ch1 in enumerate(ch_list):
            for j, ch2 in enumerate(ch_list):
                if i < j:  # Avoid duplicates (PO7-Pz same as Pz-PO7)
                    pair_name = f'{ch1}_{ch2}'
                    if epoch_idx == 0:  # Solo miramos la primera época para no inundar la consola
                        p1_debug = alpha_phase[epoch_idx, i, signal_mask]
                        p2_debug = alpha_phase[epoch_idx, j, signal_mask]

                        print(f"\n--- DEBUG ISPC {pair_name} ---")
                        print(f"Canal 1 ({ch1}) idx: {i} | Canal 2 ({ch2}) idx: {j}")
                        print(f"¿Son los datos idénticos?: {np.array_equal(p1_debug, p2_debug)}")
                        print(f"Varianza Fase 1: {np.var(p1_debug):.6f}")
                        print(f"Diferencia de fase (primeros 5): {(p1_debug - p2_debug)[:5]}")
                    # ----------------------------
                    # Alpha ISPC
                    # CRITICAL: Average vectors WITHIN window, then magnitude
                    # Baseline window
                    ispc_alpha_baseline = compute_ispc(
                        alpha_phase[epoch_idx, i, baseline_mask],
                        alpha_phase[epoch_idx, j, baseline_mask]
                    )
                    features[f'ispc_{pair_name}_alpha_baseline'] = ispc_alpha_baseline

                    # Signal window
                    ispc_alpha_signal = compute_ispc(
                        alpha_phase[epoch_idx, i, signal_mask],
                        alpha_phase[epoch_idx, j, signal_mask]
                    )
                    features[f'ispc_{pair_name}_alpha_signal'] = ispc_alpha_signal

                    # Theta ISPC
                    # Baseline window
                    ispc_theta_baseline = compute_ispc(
                        theta_phase[epoch_idx, i, baseline_mask],
                        theta_phase[epoch_idx, j, baseline_mask]
                    )
                    features[f'ispc_{pair_name}_theta_baseline'] = ispc_theta_baseline

                    # Signal window
                    ispc_theta_signal = compute_ispc(
                        theta_phase[epoch_idx, i, signal_mask],
                        theta_phase[epoch_idx, j, signal_mask]
                    )
                    features[f'ispc_{pair_name}_theta_signal'] = ispc_theta_signal

        features_list.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)

    print(f"\n✓ Feature extraction complete")
    print(f"  Total features per epoch: {len(features_df.columns) - 2}")  # -2 for epoch_idx and label
    print(f"  Total epochs: {len(features_df)}")

    return features_df

