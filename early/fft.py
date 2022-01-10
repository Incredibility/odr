import pyaudio
import numpy as np
import time
import configparser
import math
from threading import Thread, Event
from tkinter import Tk, Label, PhotoImage, TclError
from librosa.beat import beat_track, tempo
from librosa.onset import onset_strength
from librosa.filters import mel, constant_q
from collections import OrderedDict
from scipy.signal import convolve
from scipy.ndimage.filters import maximum_filter1d

CONFIG_FILE_NAME = 'config'
DEFAULT_RATE_STR = 'default_rate'
CHUNK_STR = 'chunk'
N_FFT_STR = 'n_fft'
USE_SYSTEM_AUDIO_STR = 'use_system_audio'
ODF_SEC_STR = 'odf_sec'
BPM_POLL_SEC_STR = 'bpm_poll_sec'
BASE_BPM_STR = 'base_bpm'
STD_BPM_STR = 'std_bpm'
MIN_BPM_STR = 'min_bpm'
MAX_BPM_STR = 'max_bpm'
GIF_DIR_STR = 'gif_dir'
GIF_N_BEATS_STR = 'gif_n_beats'
WINFO_X_STR = 'winfo_x'
WINFO_Y_STR = 'winfo_y'
FORMAT = pyaudio.paFloat32

config = configparser.ConfigParser()


def default_config():
    """Fill default section of <config> with default values."""
    config[configparser.DEFAULTSECT] = {DEFAULT_RATE_STR: '48000',
                                        CHUNK_STR: '512',
                                        N_FFT_STR: '2048',
                                        USE_SYSTEM_AUDIO_STR: '1',
                                        ODF_SEC_STR: '9',
                                        BPM_POLL_SEC_STR: '3',
                                        BASE_BPM_STR: '120',
                                        STD_BPM_STR: '1',
                                        MIN_BPM_STR: '55',
                                        MAX_BPM_STR: '215',
                                        GIF_DIR_STR: 'kirby.gif',
                                        GIF_N_BEATS_STR: '1',
                                        WINFO_X_STR: '0',
                                        WINFO_Y_STR: '0'}


def read_config():
    """
    Read <config> values, creating a default config file if one does not exist.
    """
    if not config.read(CONFIG_FILE_NAME):
        default_config()
        with open(CONFIG_FILE_NAME, 'w') as configfile:
            config.write(configfile)


read_config()
section = config[configparser.DEFAULTSECT]
DEFAULT_RATE = section.getint(DEFAULT_RATE_STR, fallback=48000)             # sample rate to scale everything relatively by
CHUNK = section.getint(CHUNK_STR, fallback=512)                             # size of a recording chunk in number of samples, also hop length of stft
N_FFT = section.getint(N_FFT_STR, fallback=2048)                            # number of samples to fft for one odf frame
USE_SYSTEM_AUDIO = section.getboolean(USE_SYSTEM_AUDIO_STR, fallback=True)  # whether to use system audio or microphone
ODF_SEC = section.getfloat(ODF_SEC_STR, fallback=9)                         # length of odf in seconds
BPM_POLL_SEC = section.getfloat(BPM_POLL_SEC_STR, fallback=3)               # how often to poll for a new bpm
BASE_BPM = section.getfloat(BASE_BPM_STR, fallback=120)                     # peak for log normal bias
STD_BPM = section.getfloat(STD_BPM_STR, fallback=1)                         # standard deviation for log normal bias
MIN_BPM = section.getfloat(MIN_BPM_STR, fallback=55)                        # minimum bpm
MAX_BPM = section.getfloat(MAX_BPM_STR, fallback=215)                       # maximum bpm
GIF_DIR = section.get(GIF_DIR_STR, fallback='kirby.gif')                    # gif file to use
GIF_N_BEATS = section.getint(GIF_N_BEATS_STR, fallback=1)                   # number of beats in one loop of the gif
WINFO_X = section.getint(WINFO_X_STR, fallback=0)                           # gif window position x
WINFO_Y = section.getint(WINFO_Y_STR, fallback=0)                           # gif window position y

p = pyaudio.PyAudio()

if USE_SYSTEM_AUDIO:  # find a wasapi output device
    host_api_info = None
    try:
        host_api_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        # documentation https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio.get_host_api_info_by_type
        # says IOError, but IOError has been merged with (i.e. is an alias of) OSError since python 3.3
    except OSError:
        print('no WASAPI output device')
        exit()
    device_info = p.get_device_info_by_index(host_api_info['defaultOutputDevice'])
    device_channels = device_info['maxOutputChannels']
else:  # use default input device
    device_info = p.get_default_input_device_info()
    device_channels = device_info['maxInputChannels']
DEVICE_RATE = int(device_info['defaultSampleRate'])                         # actual sample rate of wasapi output device
MOD_CHUNK = int(2 ** round(np.log2(DEVICE_RATE * CHUNK / DEFAULT_RATE)))    # scale chunk size by default sample rate to the nearest power of 2 for efficient fft
MOD_N_FFT = int(2 ** round(np.log2(DEVICE_RATE * N_FFT / DEFAULT_RATE)))    # scale the number of samples to fft like for chunk size
N_CHUNK = int(DEVICE_RATE * ODF_SEC / MOD_CHUNK)                            # actual number of chunks in entire odf, close to number of seconds specified but rounded
N_FFT_OUT = MOD_N_FFT // 2 + 1                                              # frequency range of fft output


class StoppableThread(Thread):
    """Stoppable thread from https://stackoverflow.com/q/47912701.
    """

    def __init__(self):
        Thread.__init__(self)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()


def record_callback(in_data, _frame_count, _time_info, _status):
    """
    Callback for pyaudio stream. Append single odf entry calculated from new chunk-sized <in_data>.
    """
    # convert in_data buffer into a numpy array
    samples = np.frombuffer(in_data, np.float32)
    # copy current odf to odf_buffer but shifted back one spot for new odf frame value
    record_callback.odf_buffer[:-1] = record_callback.odf[1:]
    # zero the new frame
    record_callback.odf_buffer[-1] = 0
    # accumulate onset strength from each channel to the new frame
    for i in range(device_channels):
        # shift old frames back by MOD_CHUNK samples and copy on new samples
        record_callback.frames[i][:-MOD_CHUNK] = record_callback.frames[i][MOD_CHUNK:]
        record_callback.frames[i][-MOD_CHUNK:] = samples[i::device_channels]
        # compute magnitudes of each frequency and apply filter
        current = np.abs(np.fft.rfft(record_callback.window * record_callback.frames[i]))
        current = np.dot(record_callback.filter, current)
        # log compression to be closer to human hearing
        np.log10(current + 1, out=current)
        # apply maximum filter, based on Superflux algorithm
        maximum_filter1d(current, 3, output=current)
        # accumulate positive discrete derivative
        record_callback.odf_buffer[-1] += np.maximum(0, current - record_callback.previous[i]).sum()
        record_callback.previous[i] = current
    # swap newly computed odf buffer with odf
    record_callback.odf, record_callback.odf_buffer = record_callback.odf_buffer, record_callback.odf
    return None, pyaudio.paContinue

record_callback.odf = np.zeros(N_CHUNK)
record_callback.odf_buffer = np.empty(N_CHUNK)
record_callback.frames = np.zeros((device_channels, MOD_N_FFT))
record_callback.previous = np.zeros((device_channels, 128))
record_callback.window = np.hanning(MOD_N_FFT)
record_callback.filter = mel(sr=DEVICE_RATE, n_fft=MOD_N_FFT)


class DetectThread(StoppableThread):
    """Periodically (by <BPM_POLL_SEC>) fft <RecordThread>'s odf to find reasonable tempo estimate based on librosa's
    implementation.
    """

    def __init__(self, odf_source):
        StoppableThread.__init__(self)
        self.bpm = 120
        self.beat_offset = 0
        self._bpm_scale = 60 * DEVICE_RATE / MOD_CHUNK
        self._bpms = np.empty(N_CHUNK, np.float64)
        self._bpms[0] = np.inf
        self._bpms[1:] = 60.0 * DEVICE_RATE / (MOD_CHUNK * np.arange(1.0, N_CHUNK))
        self._log_prior = -0.5 * np.square((np.log2(self._bpms) - math.log2(BASE_BPM)) / STD_BPM)
        self._log_prior[:np.argmax(self._bpms < MAX_BPM)] = -np.inf
        self._odf_source = odf_source
        self._odf = np.empty(N_CHUNK)
        self.start()

    def run(self):
        while True:
            tp = time.perf_counter() + BPM_POLL_SEC
            if np.any(self._odf_source.odf):  # skip if odf is all zero
                tt = time.time()
                self._odf[:] = self._odf_source.odf

                # normalize by cutting off below mean?
                # global corr
                # corr = np.fft.irfft(np.square(np.abs(np.fft.rfft(np.maximum(0, self._odf - np.mean(self._odf)), n=2 * N_CHUNK + 1))))[:N_CHUNK]
                corr = np.fft.irfft(np.square(np.abs(np.fft.rfft(self._odf, n=2 * N_CHUNK + 1))))[:N_CHUNK]
                corr_min = np.min(corr)
                corr_normed = np.log1p((corr - corr_min) / (np.max(corr) - corr_min) * 1e6) + self._log_prior
                argmax = np.argmax(corr_normed)
                # print(f'confidence: {(corr[argmax] - np.median(corr)) / (corr[argmax] - corr_min)}')
                argmax_m1, argmax_p1 = argmax - 1, argmax + 1
                if corr[argmax_p1] == 0:
                    corr[argmax_p1] = np.finfo(corr.dtype).eps
                if corr[argmax_m1] == 0:
                    corr[argmax_m1] = np.finfo(corr.dtype).eps
                bpm = self._bpm_scale / (argmax + 0.5 * math.log(corr[argmax_p1] / corr[argmax_m1]) / math.log(
                    corr[argmax] * corr[argmax] / (corr[argmax_m1] * corr[argmax_p1])))
                # interpolate from normed corr?
                # bpm = self._bpm_scale / (argmax + 0.5 * math.log(corr_normed[argmax_p1] / corr_normed[argmax_m1]) / math.log(
                #     corr_normed[argmax] * corr_normed[argmax] / (corr_normed[argmax_m1] * corr_normed[argmax_p1])))
                # print(self._bpm_scale / argmax)
                # print(bpm, self._bpm_scale / argmax_p1, self._bpm_scale / argmax_m1)

                # odf_fft = np.fft.rfft(self._odf * np.hanning(N_CHUNK))
                # odf_fft_abs = np.abs(odf_fft)
                # odf_fft_abs[:int(MIN_BPM * N_CHUNK * MOD_CHUNK / DEVICE_RATE / 60.0)] = 0
                # odf_fft_abs[int(MAX_BPM * N_CHUNK * MOD_CHUNK / DEVICE_RATE / 60.0):] = 0
                # my_argmax = np.argmax(odf_fft_abs)
                # my_argmax_p1 = my_argmax + 1
                # my_argmax_m1 = my_argmax - 1
                # if odf_fft_abs[my_argmax_p1] == 0:
                #     odf_fft_abs[my_argmax_p1] += np.finfo(odf_fft_abs.dtype).eps
                # if odf_fft_abs[my_argmax_m1] == 0:
                #     odf_fft_abs[my_argmax_m1] += np.finfo(odf_fft_abs.dtype).eps
                # interpolated_my_argmax = my_argmax + 0.5 * math.log(odf_fft_abs[my_argmax_p1] / odf_fft_abs[my_argmax_m1]) / math.log(
                #     odf_fft_abs[my_argmax] * odf_fft_abs[my_argmax] / (odf_fft_abs[my_argmax_m1] * odf_fft_abs[my_argmax_p1]))
                # print(interpolated_my_argmax * 60.0 / (N_CHUNK * MOD_CHUNK / DEVICE_RATE))

                if MIN_BPM < bpm < MAX_BPM:
                    self.bpm = bpm
                    # print(self.bpm)

                # onset1 = onset_strength(y=record_callback.all_samples[::2], sr=DEVICE_RATE)
                # onset2 = onset_strength(y=record_callback.all_samples[1::2], sr=DEVICE_RATE)
                # print('librosa bpm:', tempo(sr=DEVICE_RATE, onset_envelope=onset1 + onset2), 'bpm2:', tempo(sr=DEVICE_RATE, onset_envelope=self._odf))

                # estimate beat locations to estimate beat offset
                period = round(60.0 * DEVICE_RATE / (MOD_CHUNK * self.bpm))
                norm = self._odf.std(ddof=1)
                if norm > 0:
                    self._odf /= norm
                try:
                    local_score = convolve(self._odf,
                                           np.exp(-0.5 * np.square(np.arange(-period, period + 1) * 32.0 / period)),
                                           'same')
                except ValueError:
                    if self._stop_event.wait(timeout=max(0.0, tp - time.perf_counter())):
                        print('stop_event detect_thread')
                        break
                    continue
                backlink = np.zeros_like(local_score, dtype=int)
                cumscore = np.zeros_like(local_score)
                window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)
                txwt = -100 * (np.square(np.log(-window / period)))

                first_beat = True
                for i, score_i in enumerate(local_score):
                    z_pad = np.maximum(0, min(-window[0], len(window)))
                    candidates = txwt.copy()
                    candidates[z_pad:] += cumscore[window[z_pad:]]
                    beat_location = np.argmax(candidates)
                    cumscore[i] = score_i + candidates[beat_location]
                    if first_beat and score_i < 0.01 * max(local_score):
                        backlink[i] = -1
                    else:
                        backlink[i] = window[beat_location]
                        first_beat = False
                    window += 1
                x_pad = np.pad(cumscore, [(1, 1)], mode='edge')
                inds1 = [slice(0, -2)]
                inds2 = [slice(2, x_pad.shape[0])]
                maxes = (cumscore > x_pad[tuple(inds1)]) & (cumscore >= x_pad[tuple(inds2)])
                med_score = np.median(cumscore[np.argwhere(maxes)])
                beats = [np.argwhere(cumscore * maxes * 2 > med_score).max()]
                while backlink[beats[-1]] >= 0:
                    beats.append(backlink[beats[-1]])
                beats = np.array(beats[::-1], dtype=int)
                smooth_boe = convolve(local_score[beats], np.hanning(5), 'same')
                valid = np.argwhere(smooth_boe > 0.5 * math.sqrt(np.square(smooth_boe).mean()))
                try:
                    beats = beats[valid.min():valid.max()].astype(np.float64) * MOD_CHUNK / DEVICE_RATE
                except ValueError:
                    beats = []

                # beats = beat_track(
                #     sr=DEVICE_RATE, onset_envelope=self._odf, hop_length=CHUNK, bpm=self.bpm, units='time')[1]
                if len(beats) > 0:
                    self.beat_offset = tt + np.median(beats % (60 / self.bpm))
                    # print(np.angle(odf_fft[my_argmax]), self.beat_offset)
                    # print((np.angle(odf_fft[my_argmax]) - np.pi / 2) / (2 * np.pi) % 1.0 * 60.0 / self.bpm, np.median(beats % (60 / self.bpm)))
            else:  # odf is all zero, so revert to BASE_BPM with zero offset
                self.bpm = BASE_BPM
                self.beat_offset = 0
            if self._stop_event.wait(timeout=max(0.0, tp - time.perf_counter())):
                print('stop_event detect_thread')
                break


class BeatThread(StoppableThread):
    """Play gif according to bpm and beat offset estimated by <DetectThread>."""

    def __init__(self, detect_thread, label_):
        StoppableThread.__init__(self)
        self._beat_n = 1
        self._tick_tock = 'tick'
        self._next_beat_time = 0
        self._last_beat_offset = detect_thread.beat_offset
        self._ids = []
        self._gif_beat = 0
        self._detect_thread = detect_thread
        self._label = label_
        self._frames = []
        self._n_frames = 0
        while True:  # load gif frames
            try:
                self._frames.append(PhotoImage(file=GIF_DIR, format=f'gif -index {self._n_frames}'))
                self._n_frames += 1
                print(self._n_frames)
            except TclError:
                break
        self.start()

    def run(self):
        self._next_beat_time = time.time() + (60 / self._detect_thread.bpm)
        while True:
            tp = time.perf_counter() + 1 / 60
            if time.time() >= self._next_beat_time:  # next beat has arrived
                print(f'{self._tick_tock} beat {self._beat_n}, ({self._detect_thread.bpm} bpm)')
                if self._gif_beat == GIF_N_BEATS:  # reached last beat, so reset current beat number
                    self._gif_beat = 0
                for k in self._ids:  # dequeue all upcoming frames
                    root.after_cancel(k)
                self._ids.clear()
                last_frame_time = 0
                for k in range(self._gif_beat * self._n_frames // GIF_N_BEATS,
                               (self._gif_beat + 1) * self._n_frames // GIF_N_BEATS):  # queue next frames
                    frame_time = GIF_N_BEATS * (
                            k - self._gif_beat * self._n_frames // GIF_N_BEATS) * 60000 / self._detect_thread.bpm / \
                                 self._n_frames
                    if frame_time - last_frame_time < 8:
                        continue
                    self._ids.append(root.after(int(frame_time), self._update, k))
                    last_frame_time = frame_time
                if self._detect_thread.beat_offset == self._last_beat_offset:  # same beat offset
                    self._next_beat_time += 60 / self._detect_thread.bpm
                else:  # different beat offset, so need to shift and correct
                    gap = (self._next_beat_time - self._detect_thread.beat_offset) % (60 / self._detect_thread.bpm)
                    if gap > 30 / self._detect_thread.bpm:
                        self._next_beat_time += 120 / self._detect_thread.bpm - gap
                    else:
                        self._next_beat_time += 60 / self._detect_thread.bpm - gap
                    self._last_beat_offset = self._detect_thread.beat_offset
                self._beat_n += 1
                self._tick_tock = 'tock' if self._tick_tock == 'tick' else 'tick'
                self._gif_beat += 1
            if self._stop_event.wait(timeout=max(0.0, tp - time.perf_counter())):
                print('destroying')
                root.quit()
                print('stop_event beat_thread')
                break

    def _update(self, index):
        """Change to the <index>th frame of gif."""
        self._label.configure(image=self._frames[index])


stream = p.open(format=FORMAT,
                channels=device_channels,
                rate=DEVICE_RATE,
                input=True,
                frames_per_buffer=MOD_CHUNK,
                input_device_index=device_info['index'],
                stream_callback=record_callback,
                as_loopback=USE_SYSTEM_AUDIO)

stream.start_stream()

threads = OrderedDict({'detect_thread': DetectThread(record_callback)})

root = Tk()
label = Label(root)
label.pack()

threads['beat_thread'] = BeatThread(threads['detect_thread'], label)


def toggle_show_window(_event=None):
    """Toggle the main window between shown and hidden."""
    toggle_show_window.show = not toggle_show_window.show
    root.overrideredirect(toggle_show_window.show)
    root.wm_attributes('-topmost', toggle_show_window.show)


def on_close():
    """Save window location to config, tell all threads to stop and wait for them to join."""
    root_winfo_x = root.winfo_x()
    root_winfo_y = root.winfo_y()
    if root_winfo_x != WINFO_X or root_winfo_y != WINFO_Y:
        if not config.read(CONFIG_FILE_NAME):
            default_config()
        config[configparser.DEFAULTSECT][WINFO_X_STR] = f'{root_winfo_x}'
        config[configparser.DEFAULTSECT][WINFO_Y_STR] = f'{root_winfo_y}'
        with open(CONFIG_FILE_NAME, 'w') as configfile:
            config.write(configfile)
    for thread in reversed(threads):
        threads[thread].stop()
        print(f'stopped {thread}')
    for thread in reversed(threads):
        threads[thread].join(timeout=1)
        print(f'returned {thread}')
    stream.stop_stream()
    stream.close()
    p.terminate()

# graph odf against predicted bpm pulses
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots()
# ln, = plt.plot(record_callback.odf)
# average, = plt.plot(np.zeros(N_CHUNK))

# def init():
#     ax.set_xlim(0, N_CHUNK)
#     return ln, average

# def update(i):
#     # cut off below mean?
#     average.set_ydata(np.full(N_CHUNK, np.mean(record_callback.odf)))
#     ax.set_ylim(0, np.max(record_callback.odf) + 1)
#     ln.set_ydata(record_callback.odf)
#     return ln, average

# ani = FuncAnimation(fig, update, init_func=init, blit=True)
# plt.show(block=False)

root.geometry(f'+{WINFO_X}+{WINFO_Y}')
toggle_show_window.show = False
toggle_show_window()
root.bind('<Button-1>', toggle_show_window)
root.wm_attributes('-transparentcolor', root.cget('bg'))
root.protocol('WM_DELETE_WINDOW', on_close)
root.mainloop()
