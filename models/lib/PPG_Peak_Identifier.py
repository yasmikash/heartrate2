import numpy as np
from wfdb import processing
from tensorflow.keras.models import load_model
from functools import partial
from importlib.resources import path

class PPG_PEAKS_NORMALIZER:

    def __init__(self, model, window_size, stride):
        self.model = load_model(model)
        self.window_s = window_size
        self.stride = stride

    def _partitioning_windows(self, ppg_signal):
        padding_signal = np.pad(ppg_signal,(self.window_s-self.stride, self.window_s), mode='edge')

        ppg_windows = []
        window_indexes = []

        padding_id = np.arange(padding_signal.shape[0])

        for window_id in range(0, len(padding_signal), self.stride):
            if window_id + self.window_s < len(padding_signal):
                ppg_windows.append(padding_signal[window_id:window_id+self.window_s])
                window_indexes.append(padding_id[window_id:window_id+self.window_s])

        ppg_windows = np.asarray(ppg_windows)
        ppg_windows = ppg_windows.reshape(ppg_windows.shape[0], ppg_windows.shape[1], 1)
        window_indexes = np.asarray(window_indexes)
        window_indexes = window_indexes.reshape(window_indexes.shape[0]*window_indexes.shape[1])

        return window_indexes, ppg_windows

    def _find_means(self, indices, values):
    
        assert(indices.shape == values.shape)
        comb = np.column_stack((indices, values))
        comb = comb[comb[:, 0].argsort()]
        split_on = np.where(np.diff(comb[:, 0]) != 0)[0]+1
        mean_values = [arr[:, 1].mean() for arr in np.split(comb, split_on)]
        mean_values = np.array(mean_values)

        return mean_values

    def _predict_means(self, window_indexes, preds, orig_len):
      
        preds = preds.reshape(preds.shape[0]*preds.shape[1])
        assert(preds.shape == window_indexes.shape)
        pred_mean = self._find_means(indices=window_indexes, values=preds)
        pred_mean = pred_mean[int(self.window_s-self.stride):(self.window_s-self.stride)+orig_len]

        return pred_mean

class SIGNAL_PEAKS_IDENTIFIER(PPG_PEAKS_NORMALIZER):
 
    def __init__(self, sampling_rate = 20, stride=250, window_size=500, threshold=0.01):
        self.model_path = "models/ppgmodel.h5"
        super().__init__(self.model_path, window_size, stride)
        self.iput_fs = sampling_rate
        self.threshold = threshold


    def _extract_predictions(self, ppg_signal, preds):
        assert(ppg_signal.shape == preds.shape)
        above_thresh = preds[preds > self.threshold]
        above_threshold_idx = np.where(preds > self.threshold)[0]
        correct_up = processing.correct_peaks(sig=ppg_signal,
                                              peak_inds=above_threshold_idx,
                                              search_radius=5,
                                              smooth_window_size=20,
                                              peak_dir='up')

        final_peaks = []

        for peak_id in np.unique(correct_up):
  
            points_in_peak = np.where(correct_up == peak_id)[0]
            if points_in_peak.shape[0] >= 5:
                final_peaks.append(peak_id)

        final_peaks = np.asarray(final_peaks)

        return final_peaks

    def identify_ppg_peaks(self, ppg_signal):
        signal = ppg_signal

        padded_indices, ppg_windows = self._partitioning_windows(ppg_signal=signal)
        normalize = partial(processing.normalize_bound, lb=-1, ub=1)
        ppg_windows = np.apply_along_axis(normalize, 1, ppg_windows)
        predictions = self.model.predict(ppg_windows, verbose=0)
        means_for_predictions = self._predict_means(window_indexes=padded_indices,preds=predictions,orig_len=signal.shape[0])

        predictions = means_for_predictions
        final_peaks = self._extract_predictions(ppg_signal=signal,preds=predictions)
        peaks_count = final_peaks

        return peaks_count
