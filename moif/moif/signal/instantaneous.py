import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.signal as signal
from scipy.interpolate import interp1d
import warnings

def extract_instantaneous_features(
    ecg_signal: np.ndarray, 
    eda_signal: np.ndarray, 
    fs_ecg: int = 700, 
    fs_eda: int = 4, 
    target_fs: int = 100
) -> pd.DataFrame:
    """
    Extract continuous, instantaneous physiological features at `target_fs`.
    Uses STFT/CWT principles for HRV and Deconvolution for EDA.
    Returns a DataFrame where each row is 1 sample at `target_fs`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 1. Determine total duration based on ECG
        duration_sec = len(ecg_signal) / fs_ecg
        target_length = int(np.floor(duration_sec * target_fs))
        
        # Create timestamps for alignment
        t_target = np.linspace(0, duration_sec, target_length, endpoint=False)
        
        # --- 2. EDA Processing (Upsampling & Deconvolution) ---
        # Clean EDA
        eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=fs_eda)
        
        # Interpolate to target_fs
        t_eda_orig = np.linspace(0, len(eda_signal)/fs_eda, len(eda_signal), endpoint=False)
        f_eda = interp1d(t_eda_orig, eda_cleaned, bounds_error=False, fill_value="extrapolate")
        eda_100hz = f_eda(t_target)
        
        # Neurokit phasic separation at high resolution
        eda_decomposed = nk.eda_phasic(eda_100hz, sampling_rate=target_fs)
        eda_tonic = eda_decomposed['EDA_Tonic'].values
        eda_phasic = eda_decomposed['EDA_Phasic'].values
        
        # --- 3. ECG / HRV Processing (Instantaneous Powers) ---
        # Clean and peak detection
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs_ecg)
        peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_ecg)
        rpeaks = info['ECG_R_Peaks']
        
        if len(rpeaks) > 10:
            # Calculate R-R Intervals (ms)
            rri_ms = np.diff(rpeaks) / fs_ecg * 1000.0
            rri_t = rpeaks[1:] / fs_ecg
            
            # Interpolate tightly to 100Hz continuously
            f_rri = interp1d(rri_t, rri_ms, kind='cubic', bounds_error=False, fill_value="extrapolate")
            rri_100hz = f_rri(t_target)
            
            # Calculate instantaneous powers directly from RRI (Approximate via 4Hz moving average before 100Hz interoplation to prevent huge STFT matrices)
            # This is a pragmatic optimization for 100Hz CWT
            t_orig_rri = rri_t
            
            # Use Welch/Lomb-Scargle or simple Bandpass for instantaneous power tracking
            try:
                # Direct bandpass filtering on the interpolated signal for HF and LF envelopes
                sos_lf = signal.butter(4, [0.04, 0.15], btype='bandpass', fs=target_fs, output='sos')
                sos_hf = signal.butter(4, [0.15, 0.40], btype='bandpass', fs=target_fs, output='sos')
                
                lf_band = signal.sosfiltfilt(sos_lf, rri_100hz)
                hf_band = signal.sosfiltfilt(sos_hf, rri_100hz)
                
                # Hilbert transform to get analytic signal envelope (Instantaneous power)
                inst_lf = np.abs(signal.hilbert(lf_band)) ** 2
                inst_hf = np.abs(signal.hilbert(hf_band)) ** 2
                
            except Exception as e:
                print(f"Filter extraction failed: {e}")
                inst_lf = np.full(target_length, np.nan)
                inst_hf = np.full(target_length, np.nan)
                
                # Align Spectrogram time axis back to absolute target timestamps
                inst_lf = interp1d(t, inst_lf, bounds_error=False, fill_value="extrapolate")(t_target)
                inst_hf = interp1d(t, inst_hf, bounds_error=False, fill_value="extrapolate")(t_target)
            
            except Exception as e:
                print(f"Time-Frequency extraction failed: {e}")
                inst_lf = np.full(target_length, np.nan)
                inst_hf = np.full(target_length, np.nan)
        else:
            rri_100hz = np.full(target_length, np.nan)
            inst_lf = np.full(target_length, np.nan)
            inst_hf = np.full(target_length, np.nan)
            
        # Compile as DataFrame
        df_feats = pd.DataFrame({
            'timestamp': t_target,
            'EDA_Tonic': eda_tonic,
            'EDA_Phasic': eda_phasic,
            'HRV_RRI': rri_100hz,
            'HRV_Inst_LF': inst_lf,
            'HRV_Inst_HF': inst_hf
        })
        
        return df_feats
