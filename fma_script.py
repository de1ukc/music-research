import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List

TRACKS_FOLDER = './fma'
assert os.path.isdir(TRACKS_FOLDER), "wrong path"  # sanity check


def extract_tracks_from_folder(folder_path):
    track_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp3', '.wav')):
            track_paths.append(os.path.join(folder_path, filename))

    return track_paths


TRACKS_PATHS: List[str] = extract_tracks_from_folder(folder_path=TRACKS_FOLDER)


def extract_tempogram_and_onset_strength(track_path: str):
    y, sr = librosa.load(track_path)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=512)

    return tempogram, onset_env, sr


def plot_tempogram_and_onset_strength(tempogram, onset_env, sr, hop_length=512):
    ...


def get_all_tracks_tempograms() -> List[np.ndarray]:
    tempograms: List[np.ndarray] = []

    for track_path in TRACKS_PATHS:
        tempogram, onset_env, sr = extract_tempogram_and_onset_strength(track_path=track_path)

        # plot_tempogram_and_onset_strength(tempogram=tempogram, onset_env=onset_env, sr=sr)
        tempograms.append(tempogram)

    return tempograms


def extract_beatgram(track_path):
    y, sr = librosa.load(track_path, duration=40)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    return onset_env, beats, sr


def plot_beatgram(onset_env, beats, sr):
    plt.figure(figsize=(12, 4))

    # Отображение onset strength
    plt.subplot(2, 1, 1)
    times = librosa.times_like(onset_env, sr=sr)
    plt.plot(times, librosa.util.normalize(onset_env), label='Onset Strength')
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    plt.title('Onset Strength and Beats')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Отображение битграммы
    plt.subplot(2, 1, 2)
    hop_length = 512
    beatgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    librosa.display.specshow(beatgram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma')
    plt.title('Beatgram')
    plt.colorbar()
    plt.show()


def get_all_tracks_beatgrams() -> List[np.ndarray]:
    beatgrams: List[np.ndarray] = []
    for track_path in TRACKS_PATHS:
        onset_env, beatgram, sr = extract_beatgram(track_path=track_path)

        # plot_beatgram(onset_env=onset_env, beats=beatgram, sr=sr)
        beatgrams.append(beatgram)

    return beatgrams


def main():
    tempograms: List[np.ndarray] = get_all_tracks_tempograms()

    beatgrams = get_all_tracks_beatgrams()


if __name__ == "__main__":
    main()
