import mne
from pylsl import StreamInlet, resolve_stream
import numpy as np

channel_names = ['C3', 'C4', 'CP3', 'CP4', 'F5', 'F6', 'PO3', 'PO4']
channel_types = ['eeg'] * len(channel_names)
sfreq = 256  # Adjust this based on your actual streams' sampling frequency
info = mne.create_info(channel_names, sfreq, channel_types)

data_buffer = []

try:
    print("Looking for EEG streams...")
    streams = resolve_stream('type', 'EEG')
    inlets = [StreamInlet(stream) for stream in streams]

    while True:
        for inlet in inlets:
            sample, timestamp = inlet.pull_sample(timeout=0.0)
            if sample:
                scaled_sample = np.array(sample) / 1000000
                data_buffer.append(scaled_sample)
except KeyboardInterrupt:
    print("Ending program and saving data...")
    # Transpose to match MNE's expected shape (channels x samples)
    data_array = np.array(data_buffer).T
    raw = mne.io.RawArray(data_array, info)

    fn_raw = "output.raw.edf"
    print(f"Saving raw data to {fn_raw}")
    mne.export.export_raw(fn_raw, raw, overwrite=True)

    filtered = raw.copy()
    filtered.filter(0.3, 35)
    filtered.notch_filter(freqs=[50,100])

    filtered.plot_psd(average=False)
    filtered.plot()
except Exception as e:
    print(f"An error occurred: {e}")