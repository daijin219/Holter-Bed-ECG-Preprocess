import mne
def read_edf_file(filename,read_fs=None):
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    data = raw.get_data()[0:-1,:].T
    data = data*1000
    if read_fs is not None:
        fs = raw.info['sfreq']
        return data,fs
    else:
        return data
