import numpy as np
import scipy.signal as signal
import pandas as pd
# import hrvanalysis as hrvana
import neurokit2 as nk
import matplotlib as plt
from scipy.signal import welch
import os
from tqdm import tqdm
from scipy import signal, interpolate
from scipy.stats import zscore


MESA_PPG_PATH_NO_PAD = '/dcs/large/u2212061/raw_ppg_signal/'
MESA_STAGE_PATH_NO_PAD = '/dcs/large/u2212061/raw_stage_ann/'
PRV_FEATURES = '/dcs/large/u2212061/PRV_features/'
PRV_FEATURES_21 = '/dcs/large/u2212061/PRV_features/Window_21/'
PRV_FEATURES_51 = '/dcs/large/u2212061/PRV_features/Window_51/'
PRV_FEATURES_101 = '/dcs/large/u2212061/PRV_features/Window_101/'
PRV_STAGE = '/dcs/large/u2212061/PRV_stage/'
PRV_STAGE_21 = '/dcs/large/u2212061/PRV_stage/Window_21/'
PRV_STAGE_51 = '/dcs/large/u2212061/PRV_stage/Window_51/'
PRV_STAGE_101 = '/dcs/large/u2212061/PRV_stage/Window_101/'


# window_sizes (list) = [21, 51, 101]
def create_sliding_windows(ppg_signal, window_sizes, step_size=1024):
    windows = {}
    center_indices = {}
    for size in window_sizes:
        size_in_points = size * 1024
        windows[size] = []
        center_indices[size] = []

        total_groups = len(ppg_signal) // step_size
        num_complete_windows = total_groups - size + 1
        
        # num_complete_windows = (len(ppg_signal) - size_in_points) // step_size + 1
        
        # Complete windows
        for i in range(num_complete_windows):
            start = i * step_size
            window = ppg_signal[start:start+size_in_points]
            windows[size].append(window)
            
            center_index = (start + size_in_points // 2) // 1024
            center_indices[size].append(center_index)
        
        windows[size] = np.array(windows[size])
        center_indices[size] = np.array(center_indices[size])
    
    return windows, center_indices


def find_ppi(segment, target_fs=34.13):
    # 0.5Hz - 2.0Hz
    nyquist_freq = 0.5 * target_fs
    low = 0.5 / nyquist_freq
    high = 2.0 / nyquist_freq
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_segment = signal.filtfilt(b, a, segment)
    
    # Peak detection
    peaks, _ = signal.find_peaks(filtered_segment, distance=0.5*target_fs)
    
    # PPI
    ppi = np.diff(peaks) / target_fs * 1000
    
    # z-score
    z_scores = np.abs(zscore(ppi))
    threshold = 2
    ppi_cleaned = ppi[z_scores < threshold]
    
    median_ppi = np.median(ppi_cleaned)
    ppi[z_scores >= threshold] = median_ppi
    
    return ppi


def extract_prv_features(pp_intervals):
    # 1. Mean PPI
    mean_ppi = np.mean(pp_intervals)
    
    # 2. SDPP
    sdpp = np.std(pp_intervals)
    
    # 3. SDSD
    sdsd = np.std(np.diff(pp_intervals))
    
    # 4-7. Frequency
    freq, psd = welch(pp_intervals, fs=1000/np.mean(pp_intervals))
    vlf = np.trapz(psd[(freq >= 0.0033) & (freq < 0.04)])
    lf = np.trapz(psd[(freq >= 0.04) & (freq < 0.15)])
    hf = np.trapz(psd[(freq >= 0.15) & (freq < 0.4)])

    if hf > 0:
        lf_hf_ratio = lf / hf
    else:
        lf_hf_ratio = 0.0

    # 8. Total power
    total_power = np.trapz(psd)
    
    return [mean_ppi, sdpp, sdsd, vlf, lf, hf, lf_hf_ratio, total_power]


def calculate_prv_features(windows_dict):
    all_features = {}
    for window_size, windows in windows_dict.items():
        features_list = []
        for window in windows:
            pp_intervals = find_ppi(window)
            features = extract_prv_features(pp_intervals)
            features_list.append(features)
        all_features[window_size] = np.array(features_list)
    return all_features


def extract_prv_stage(window_sizes, center_indices, ppg_stage):
    prv_stage_dict = {}
    for size in window_sizes:
        prv_stage_list = []
        for index in center_indices[size]:
            prv_stage_list.append(ppg_stage[index])
        prv_stage_dict[size] = prv_stage_list

    return prv_stage_dict


def prv_main():
    ppg_name_list = sorted(os.listdir(MESA_PPG_PATH_NO_PAD))
    window_sizes = [21]

    for ppg_name in tqdm(ppg_name_list, desc='PRV Calculating'):
        ppg_signal = np.loadtxt(os.path.join(MESA_PPG_PATH_NO_PAD, ppg_name))
        ppg_stage = np.loadtxt(os.path.join(MESA_STAGE_PATH_NO_PAD, ppg_name), dtype=int)
        
        # Sliding window
        windows, center_indices = create_sliding_windows(ppg_signal, window_sizes)

        prv_stage_dict = extract_prv_stage(window_sizes, center_indices, ppg_stage)

        for size, stage in prv_stage_dict.items():
            folder_name = f'Window_{size}'
            output_path_stage = os.path.join(PRV_STAGE, folder_name, ppg_name)
            df_stage = pd.DataFrame(stage, columns=['Stage'])
            df_stage.to_csv(output_path_stage, sep='\t', index=False)

        prv_features_dict = calculate_prv_features(windows)
        
        for size, features in prv_features_dict.items():
            folder_name = f'Window_{size}'
            output_path_signal = os.path.join(PRV_FEATURES, folder_name, ppg_name)
            
            column_names = ['mean_ppi', 'sdpp', 'sdsd', 'vlf', 'lf', 'hf', 'lf_hf_ratio', 'total_power']
            
            df_signal = pd.DataFrame(features, columns=column_names)
            # df_signal = pd.DataFrame(features)
            df_signal.to_csv(output_path_signal, sep='\t', index=False)

    print('All files processed successfully!')
    

prv_main()




