import numpy as np
import neurokit2 as nk
import warnings

def extract_window_features(ecg_signal: np.ndarray, eda_signal: np.ndarray, fs_ecg: int = 700, fs_eda: int = 4) -> dict:
    """
    Extract advanced physiological features from a sliding window using neurokit2.
    
    Parameters:
        ecg_signal: 1D array of ECG data
        eda_signal: 1D array of EDA data
        fs_ecg: ECG sampling rate (default 700 for WESAD)
        fs_eda: EDA sampling rate (default 4 for WESAD)
        
    Returns:
        dict: Aggregated features for the window (e.g. HRV_RMSSD, EDA_Tonic_Mean)
    """
    features = {}
    
    # 1. ECG Processing (HRV)
    if ecg_signal is not None and len(ecg_signal) >= fs_ecg * 30: # Need at least 30s for meaningful HRV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Clean, find peaks, and compute HRV
                # using a lightweight approach to avoid heavy plotting/logging
                ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs_ecg)
                peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_ecg)
                
                # Compute HRV
                hrv_indices = nk.hrv(peaks, sampling_rate=fs_ecg, show=False)
                
                if 'HRV_RMSSD' in hrv_indices.columns:
                    features['HRV_RMSSD'] = float(hrv_indices['HRV_RMSSD'].iloc[0])
                else:
                    features['HRV_RMSSD'] = np.nan
                    
                if 'HRV_LFHF' in hrv_indices.columns:
                    features['HRV_LFHF'] = float(hrv_indices['HRV_LFHF'].iloc[0])
                else:
                    features['HRV_LFHF'] = np.nan
                    
                if 'HRV_MeanNN' in hrv_indices.columns:
                    features['HRV_MeanNN'] = float(hrv_indices['HRV_MeanNN'].iloc[0])
                else:
                    features['HRV_MeanNN'] = np.nan
            except Exception:
                features['HRV_RMSSD'] = np.nan
                features['HRV_LFHF'] = np.nan
                features['HRV_MeanNN'] = np.nan
    else:
        features['HRV_RMSSD'] = np.nan
        features['HRV_LFHF'] = np.nan
        features['HRV_MeanNN'] = np.nan

    # 2. EDA Processing (Tonic / Phasic Deconvolution)
    if eda_signal is not None and len(eda_signal) >= fs_eda * 10: # Need at least 10s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Clean, decompose into Phasic and Tonic
                eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=fs_eda)
                eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=fs_eda)
                
                # Get mean Tonic (SCL) level for the window
                tonic_mean = float(np.mean(eda_decomposed['EDA_Tonic']))
                
                # Get mean of Phasic (SCR) component
                phasic_mean = float(np.mean(np.abs(eda_decomposed['EDA_Phasic'])))
                
                # Find peaks in the window
                peak_signal, info = nk.eda_peaks(eda_decomposed['EDA_Phasic'], sampling_rate=fs_eda, 
                                                amplitude_min=0.1)
                
                num_peaks = int(np.sum(peak_signal['SCR_Peaks']))
                
                features['EDA_Tonic_Mean'] = tonic_mean
                features['EDA_Phasic_Mean'] = phasic_mean
                features['EDA_SCR_Peaks'] = float(num_peaks)
            except Exception:
                features['EDA_Tonic_Mean'] = np.nan
                features['EDA_Phasic_Mean'] = np.nan
                features['EDA_SCR_Peaks'] = np.nan
    else:
        features['EDA_Tonic_Mean'] = np.nan
        features['EDA_Phasic_Mean'] = np.nan
        features['EDA_SCR_Peaks'] = np.nan
        
    return features
