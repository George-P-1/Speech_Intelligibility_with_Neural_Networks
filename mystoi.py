"""
STOI
====
Computes Short-Time Objective Intelligibility (STOI) metric for speech intelligibility.
Intrusive method that requires clean and speech-in-noise audio files.
This metric has a monotonic relation with subjective speech intelligibility (actual listeners score).

How to use?
-----------
    1. Import the module
    2. Load clean and speech-in-noise audio files
    3. Compute STOI value using `compute_stoi()` function
        - Check function documentation for more details

Dependencies
------------
    1. Numpy
    2. Scipy - for stft

References
----------
[1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
    Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
    IEEE Transactions on Audio, Speech, and Language Processing, 2011.

"""

# Import libraries
import numpy as np
from scipy.signal import stft

# SECTION - Constants
SR = 10000                  # Sampling rate/frequency
# Preprocessing (TF-Decomposition) - Silent frame removal and frames related
TRUE_FRAME_LEN = 256        # number of samples of Hann-windowed frames
OVERLAP = int(0.5*256)      # 50% overlap =128 samples both sides of frame. So padding upto 512 samples 
FRAME_LEN = 512             # =256 + OVERLAP * 2. This frame length is used for STFT
ENERGY_RANGE = 40           # Speech dynamic range
# one-third octave related
OCTAVE_BANDS = 15           # Number of one-third octave bands used
LCF = 150                   # lowest center frequency. Estimates higher c.f. is approx. 4.3kHz
# Comparison related
FRAME_TIME = 384            # 386 or 384 milliseconds. Paper shows both numbers in first few pages.
ANALYSIS_FRAME_LEN = 30     # Number of frames which equals an analysis window of 384ms
EPS = np.finfo("float").eps # Epsilon, smallest positive float number.
# Clipping
BETA = -15                  # Lower signal-to-distortion (SDR) bound in decibels (dB)
#!SECTION

# SECTION - Functions

def compute_stoi(clean_audio, spin_audio, sampling_rate: int) -> float:
    """
    # Short-Time Objective Intelligibility
    Computes STOI by comparing a clean and speech-in-noise audio. 

    Arguments:
        clean_audio: clean speech (target_anechoic)
        spin_audio: Speech-in-Noise
        sampling_rate (int): Assumes both audios have same sampling rate

    Returns:
        stoi_val (float): A value that has monotonic relation with subjective speech intelligibility (actual listeners score). 
        
        Higher STOI value corresponds to better intelligibility.
    """
    # Check if the sampling rate is 10kHz
    # REVIEW - Maybe later the resampling can be done within compute_stoi() function
    if SR != sampling_rate:
        raise Exception("Sampling rate is not {}".format(SR))

    # NOTE - Remove silent frames
    clean_audio, spin_audio = remove_silent_frames(clean_audio, spin_audio, ENERGY_RANGE, TRUE_FRAME_LEN, OVERLAP)

    # TF Decomposition
    # NOTE - Short-Time Fourier Transform on both signals - to obtain DFT bins
    f_clean, t_clean, clean_stft = stft(clean_audio, sampling_rate, 'hann', TRUE_FRAME_LEN, OVERLAP, FRAME_LEN)
    f_spin, t_spin, spin_stft = stft(spin_audio, sampling_rate, 'hann', TRUE_FRAME_LEN, OVERLAP, FRAME_LEN)

    # Get the one-third octave band matrix and center frequencies
    obm, cfs = one_third_octaves(sampling_rate, FRAME_LEN, OCTAVE_BANDS, LCF)

    # NOTE - Group DFT bins into one-third octave bands
    # Using Equation 1 from the paper - Apply OBM boolean matrix to spectrogram (STFT)
    clean_tf_units = np.sqrt(np.matmul(obm, (np.square(np.abs(clean_stft)))))
    spin_tf_units = np.sqrt(np.matmul(obm, (np.square(np.abs(spin_stft)))))
    
    # NOTE - Create 3D matrices of temporal envelopes (analysis windows) for clean and spin audio
    # Use a bigger frame of 30 frames (within 481) and 15 bands which equals an analysis window of 384ms
    # Use Equation 2 from the paper - short-time temporal envelope
    clean_an_windows = []
    spin_an_windows = []
    for m in range(ANALYSIS_FRAME_LEN, clean_tf_units.shape[1] + 1): # Go through the length of time axis using frame. The +1 is to include the last point since for example, range(1, 10) doesn't include 10.
        clean_an_win = clean_tf_units[:, m - ANALYSIS_FRAME_LEN : m]  # Analysis window for clean audio
        spin_an_win = spin_tf_units[:, m - ANALYSIS_FRAME_LEN : m]   # Analysis window for spin audio  
        # There is no +1 like in Equation 2 like m - ANALYSIS_FRAME_LEN +1: m, because the index 'm' is not included in the notation [0:m] so [0:30] would have 30 elements not 31.  
        clean_an_windows.append(clean_an_win)
        spin_an_windows.append(spin_an_win)

    # Convert to numpy arrays
    clean_an_windows = np.array(clean_an_windows)
    spin_an_windows = np.array(spin_an_windows)
    
    # NOTE - Normalize and clipping the spin audio based on clean audio
    # Use Equation 3 from the paper
    # Normalization
    normalization_consts = (
        np.linalg.norm(clean_an_windows, axis=2, keepdims=True) /
        (np.linalg.norm(spin_an_windows, axis=2, keepdims=True) + EPS))
    # axis=2 means 3rd dimension. In this case that is time.
    # The keepdims=True is to keep the dimensions of the result same as the input. 
    
    spin_an_windows_normalized = spin_an_windows * normalization_consts
    
    # Clipping
    clip_value = 10 ** (-BETA / 20) # Convert from dB to linear scale.
    spin_an_windows_clipped = np.minimum(spin_an_windows_normalized, clean_an_windows * (1 + clip_value))
    # NOTE - I think the 1 in (1+clip_value) is a safe threshold in which speech is expected to be present and clearly heard. Adjusting this value could affect prediction.

    # REVIEW - There is another equation (Eq 5) that says that the SDR of clean speech is required to be greater than the SDR of the noise. Maybe add it as an if condition or mention in documentation.

    # NOTE - Compare to get intermediate intelligibility measure
    # Equation 5 from the paper - Sample Correlation Coefficient
    # Centering Data (Subtract mean vectors)
    spin_inter = spin_an_windows_clipped - np.mean(spin_an_windows_clipped, axis=2, keepdims=True)
    clean_inter = clean_an_windows - np.mean(clean_an_windows, axis=2, keepdims=True)
    # Normalize by dividing by their norms
    # This results in all vectors having unit length with different directions.
    spin_inter = spin_inter / (np.linalg.norm(spin_inter, axis=2, keepdims=True) + EPS)
    clean_inter = clean_inter / (np.linalg.norm(clean_inter, axis=2, keepdims=True) + EPS)
    # Correlation
    d_matrix = clean_inter * spin_inter

    J = clean_inter.shape[1]    # Number of bands
    M = clean_inter.shape[0]    # Total number of frames

    if J != OCTAVE_BANDS:
        raise Exception ("Error due to mismatch number of bands in computation of STOI value.")

    # NOTE - Average the correlation matrix over all bands and frames to get final STOI value
    d = np.sum(d_matrix) / (J * M)

    return d


def one_third_octaves(sr: int, frame_len: int, num_bands: int, lcf):
    """
    Creates center frequencies of bands of one-third octaves and a one-third octave band matrix. 

    Arguments:
        sr (int): Sampling rate
        frame_len (int): Number of samples in a frame of signal
        num_bands (int): Number of one-third octave bands
        lcf : Lowest center frequency

    Returns:
        (obm, cf) (np.ndarray, np.ndarray): Octave Band Matrix and Center frequencies

    """
    octave_kind = 3 # third octave
    
    # Create the frequency axis based on number of samples in a frame and sampling rate
    dft_bins = np.linspace(0, sr, frame_len + 1)
    # the +1 is because sr/2 should be a frequency in the array.
    dft_bins = dft_bins[0 : int(frame_len/2) + 1]    # Take only Nyquist frequencies
    # when taking fourier transform, the half result is regarding negative frequencies that
    # doesn't matter in the real signal. So the dft_bins only need Nyquist frequencies
    
    bands = np.array(range(num_bands))  # list for 15 bands
    cfs = np.power(2.0, (bands / octave_kind)) * lcf # Center frequencies of bands
    # REVIEW - The paper mentions highest center frequency is approx. 4.3kHz but it isn't when calculated.
    ube = lcf * np.power(2.0, (2 * bands + 1) / (2*octave_kind))  # upper band edge
    # NOTE - However, the upper band edge of highest band is approx. 4.3kHz
    lbe = lcf * np.power(2.0, (2 * bands - 1) / (2*octave_kind))  # lower band edge
    
    obm = np.zeros((num_bands, len(dft_bins)))  # Octave Band Matrix
    # rows are bands and columns are dft_bins. A tall matrix.

    for band in range(len(cfs)):
        dft_bin = np.argmin(np.square(dft_bins - lbe[band])) # index of smallest value/distance
        # distance between every DFT bin and a lower band edge at a time
        lbe[band] = dft_bins[dft_bin] # update lower band edge to the nearest DFT bin
        dft_bin_l_idx = dft_bin # store index of lowest distance

        dft_bin = np.argmin(np.square(dft_bins - ube[band])) # index of smallest distance
        ube[band] = dft_bins[dft_bin] # update upper band edge to the nearest DFT bin
        dft_bin_h_idx = dft_bin # store index of highest distance

        # Assign to the octave band matrix
        obm[band, dft_bin_l_idx:dft_bin_h_idx] = 1

    return obm, cfs


def reconstruct_signal(the_frames, overlap):
    """
    Reconstructs the signal from frames. 

    Arguments:
        the_frames (np.ndarray): Frames of signal
        overlap (int): Number of samples in overlap

    Returns:
        np.ndarray: Reconstructed signal
    """
    # Shape of the frames matrix
    num_frames, frame_len = the_frames.shape
    
    # Number of segments per frame. Basically each frame will span below number of segments.
    overlaps_in_frame = -(-frame_len // overlap) # Divide and round up

    # Pad the frames - to ensure that reshaping doesn't break
    # Pad so that length becomes overlaps_in_frame * overlap and add n=overlaps_in_frame frames
    signal = np.pad(the_frames, ((0, overlaps_in_frame), (0, overlaps_in_frame * overlap - frame_len)))

    # Reshape (modify dimensions) - each of those frames become smaller frames of length 'overlap'
    # In this case those smaller 2 frames are of length 128
    signal = signal.reshape((num_frames + overlaps_in_frame, overlaps_in_frame, overlap))

    # Transpose (rearrange dimensions) of signal into (overlaps_in_frame, num_frames+overlaps_in_frame, overlap)
    # After transposing, overlapping regions are aligned along a single axis (the row axis) so they can be summed later.
    signal = np.transpose(signal, [1, 0, 2])

    # Reshape (modify dimensions again) so that signal.shape = (overlaps_in_frame * (num_frames+overlaps_in_frame), overlap)
    signal = signal.reshape((-1, overlap))

    # Remove last n=overlaps_in_frame elements
    signal = signal[:-overlaps_in_frame]

    # Reshape to (segments, frame+segments-1, hop)
    signal = signal.reshape((overlaps_in_frame, num_frames + overlaps_in_frame - 1, overlap))

    # Sum overlapping rows column-wise to reconstruct the signal
    signal = np.sum(signal, axis=0)

    # find original length of signal
    end = (len(the_frames) - 1) * overlap + frame_len
    # Reshape to original length
    signal = signal.reshape(-1)[:end]

    return signal

# STUB
def remove_silent_frames(clean_audio, spin_audio, dyn_range, true_frame_len, overlap):
    """
    Removes silent frames from clean and spin audio.
    Removes all frames in both signals in which the energy in clean speech signal is lower than 40dB.

    Arguments:
        clean_audio: clean speech (target_anechoic)
        spin_audio: Speech-in-Noise
        dyn_range (int): Speech dynamic range
        frame_len (int): Number of samples in a frame of signal
        overlap (int): Number of samples in overlap

    Returns:
        (clean_audio, spin_audio) (np.ndarray, np.ndarray): 
        Clean and Spin audio after removing silent frames
    """
    # Create Hann window mask
    hann_window = np.hanning(true_frame_len) 
    # REVIEW - Should it be frame_len + 2? np.hanning(framelen + 2)[1:-1]?
    
    # Create an array of frames for both clean and spin audio
    clean_frames = []
    spin_frames = []
    for i in range(0, len(clean_audio) - true_frame_len, overlap): # 0 128 256 384 ...
        clean_frames.append(hann_window * clean_audio[i : i+true_frame_len])
        spin_frames.append(hann_window * spin_audio[i : i+true_frame_len])
    # Convert list to np.ndarray
    clean_frames = np.array(clean_frames)
    spin_frames = np.array(spin_frames)

    # Compute energy in dB for clean audio frames
    clean_energies = 20 * np.log10(np.linalg.norm(clean_frames, axis=1) + EPS)
    # The 20 * log10 is to convert energy to dB. axis=1 is to compute energy of each frame.
    # The norm is to compute energy of each frame. The EPS is to prevent log(0).    

    # Create mask to remove silent frames
    mask = clean_energies > np.max(clean_energies) - dyn_range  

    # Use mask on both clean and spin audio frames
    clean_frames = clean_frames[mask]
    spin_frames = spin_frames[mask]

    # Reconstruct clean and spin audio
    clean_no_sil = reconstruct_signal(clean_frames, overlap)
    spin_no_sil = reconstruct_signal(spin_frames, overlap)

    return clean_no_sil, spin_no_sil

#!SECTION