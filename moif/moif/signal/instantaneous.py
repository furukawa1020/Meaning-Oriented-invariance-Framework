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
            
            # Instantaneous Time-Frequency Analysis via overlapping STFT
            # 20-second window to capture LF appropriately (0.04Hz is ~25s period, but 20s captures enough power gradient)
            nperseg = min(target_fs * 20, len(rri_100hz) // 2)
            noverlap = nperseg - 1 # 1 sample step for instantaneous tracking
            
            try:
                f, t, Zxx = signal.spectrogram(rri_100hz, fs=target_fs, window='hann', 
                                               nperseg=nperseg, noverlap=noverlap)
                
                power = np.abs(Zxx)**2
                lf_mask = (f >= 0.04) & (f < 0.15)
                hf_mask = (f >= 0.15) & (f <= 0.4)
                
                inst_lf = np.sum(power[lf_mask, :], axis=0) if np.sum(lf_mask) > 0 else np.full(len(t), np.nan)
                inst_hf = np.sum(power[hf_mask, :], axis=0) if np.sum(hf_mask) > 0 else np.full(len(t), np.nan)
                
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
