import os
import sys
import glob
import numpy as np
import tables
import matplotlib.pyplot as plt

msd_subset_path = "/Users/de1ukc/git/music-research/msd_dataset/MillionSongSubset"
# msd_subset_data_path = os.path.join(msd_subset_path, "data")
# msd_subset_addf_path = os.path.join(msd_subset_path, "AdditionalFiles")
assert os.path.isdir(msd_subset_path), "wrong path"  # sanity check

# You need to have the MSD code downloaded from GITHUB. https://github.com/tbertinmahieux/MSongsDB
msd_code_path = "/Users/de1ukc/music-analysis/MSongsDB-master"
assert os.path.isdir(msd_code_path), "wrong path"  # sanity check

sys.path.append(os.path.join(msd_code_path, "PythonSrc"))

import hdf5_getters as GETTERS


# we define this very useful function to iterate the files
def apply_to_all_files(basedir, func=lambda x: x, ext=".h5"):
    """
    From a base directory , go through all subdirectories , find all files with the given extension, apply the
    given function 'func' to all of them.
    If no 'func' is passed, we do nothing except counting. INPUT
    basedir - base directory of the dataset
    func - function to apply to all filenames ext - extension , .h5 by default
    RETURN
    number of files
    """
    cnt = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, "*" + ext))  # count files
        cnt += len(files)

        # apply function to all files
        for f in files:
            func(f)

    return cnt


def collect_dataset(basedir, ext=".h5"):
    beats: list[np.ndarray] = []
    tempos: list[float] = []
    segments_pitches_list = []
    mfccs_list = []

    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, "*" + ext))  # count files

        # apply function to all files
        for f in files:
            start_beats, tempo, segments_pitches, mfccs = get_data_from_file(h5_file_path=f)

            # plot_tempogram(start_beats)
            # plot_beatgram(start_beats)
            # plot_chromagram(segments_pitches=segments_pitches)
            # ploat_spec(mfccs=mfccs)

            beats.append(start_beats)
            tempos.append(tempo)
            segments_pitches_list.append(segments_pitches)
            mfccs_list.append(mfccs)

    return beats, tempos, segments_pitches_list, mfccs_list


def get_data_from_file(h5_file_path):
    h5 = tables.open_file(h5_file_path, mode='r')
    tempo = GETTERS.get_tempo(h5)
    beats_start = GETTERS.get_beats_start(h5)
    segment_pitches = GETTERS.get_segments_pitches(h5)
    mfccs = GETTERS.get_segments_timbre(h5)

    return beats_start, tempo, segment_pitches, mfccs


def plot_tempogram(beats_start):
    # Рассчитываем интервалы между ударами (beats)
    beat_intervals = np.diff(beats_start)

    plt.figure(figsize=(10, 6))
    plt.plot(beats_start[:-1], 60 / beat_intervals, label='Темпограмма')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Темп (ударов в минуту)')
    plt.title('Темпограмма')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_chromagram(segments_pitches):
    plt.figure(figsize=(10, 6))
    plt.imshow(segments_pitches.T, aspect='auto', origin='lower', cmap='magma')
    plt.xlabel('Сегменты')
    plt.ylabel('Питч (Хроматические признаки)')
    plt.title('Хромограмма')
    plt.colorbar(label='Нормализованные значения')
    plt.show()


def plot_beatgram(beats_start):
    plt.figure(figsize=(10, 6))
    plt.vlines(beats_start, ymin=0, ymax=1, color='r', alpha=0.75, label='Биты')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Уровень активности')
    plt.title('Битграмма')
    plt.legend()
    plt.grid(True)
    plt.show()


def ploat_spec(mfccs):
    # Plotting the MFCCs
    plt.figure(figsize=(12, 6))
    plt.imshow(mfccs.T, aspect='auto', origin='lower', cmap='hot')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time (in frames)')
    plt.title('MFCC Visualization')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


beats, tempos, segments_pitches_list, mfccs_list = collect_dataset(basedir=msd_subset_path)
print('number of songs', apply_to_all_files(basedir=msd_subset_path))
