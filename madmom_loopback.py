from madmom.audio.signal import *
import samplerate


class StreamLoopbackWASAPI(object):
    """
    Like Stream, but prioritize the default WASAPI output device with loopback and tune for RNNBeatProcessor and DBNBeatTrackingProcessor.
    """

    def __init__(self, dtype=np.float32, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=FPS, use_system_audio=True):
        # import PyAudio here and not at the module level
        import pyaudio
        self.use_system_audio = use_system_audio
        if use_system_audio:
            try:
                # set attributes
                self.sample_rate = 44100  # apparently RNNBeatProcessor and/or DBNBeatTrackingProcessor models were trained on 44100 Hz
                self.num_channels = 1  # force mono
                self.dtype = dtype
                if frame_size:
                    self.frame_size = int(frame_size)
                if fps:
                    # use fps instead of hop_size
                    hop_size = self.sample_rate / float(fps)
                if int(hop_size) != hop_size:
                    raise ValueError(
                        'only integer `hop_size` supported, not %s' % hop_size)
                self.hop_size = int(hop_size)
                # init PyAudio
                self.pa = pyaudio.PyAudio()
                # find default WASAPI output device
                self.default_wasapi_output_device_info = self.pa.get_device_info_by_index(self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)['defaultOutputDevice'])
                # scale hop_size if device has a different sample rate
                if self.default_wasapi_output_device_info['defaultSampleRate'] != self.sample_rate:
                    if fps:
                        # use fps instead of hop_size
                        self.stream_hop_size = int(self.default_wasapi_output_device_info['defaultSampleRate'] / float(fps))
                    elif int(hop_size) != hop_size:
                        raise ValueError(
                            'only integer `hop_size` supported, not %s' % hop_size)
                    else:
                        self.stream_hop_size = int(self.hop_size * self.default_wasapi_output_device_info['defaultSampleRate'] / self.sample_rate)
                    # prepare resampler
                    self.resampler = samplerate.Resampler('linear')
                else:
                    self.stream_hop_size = self.hop_size
                # init a stream to read audio samples from
                self.stream = self.pa.open(rate=int(self.default_wasapi_output_device_info['defaultSampleRate']),
                                        channels=self.default_wasapi_output_device_info['maxOutputChannels'],
                                        format=pyaudio.paFloat32, input=True,
                                        frames_per_buffer=self.stream_hop_size,
                                        start=True,
                                        input_device_index=self.default_wasapi_output_device_info['index'],
                                        as_loopback=True)
            except OSError:
                print('no WASAPI output device')
                raise OSError
        else:
            # set attributes
            self.sample_rate = sample_rate
            self.num_channels = 1 if None else num_channels
            self.dtype = dtype
            if frame_size:
                self.frame_size = int(frame_size)
            if fps:
                # use fps instead of hop_size
                hop_size = self.sample_rate / float(fps)
            if int(hop_size) != hop_size:
                raise ValueError(
                    'only integer `hop_size` supported, not %s' % hop_size)
            self.hop_size = int(hop_size)
            # init PyAudio
            self.pa = pyaudio.PyAudio()
            # init a stream to read audio samples from
            self.stream = self.pa.open(rate=self.sample_rate,
                                    channels=self.num_channels,
                                    format=pyaudio.paFloat32, input=True,
                                    frames_per_buffer=self.hop_size,
                                    start=True)
        # create a buffer
        self.buffer = BufferProcessor(self.frame_size)
        # frame index counter
        self.frame_idx = 0
        # PyAudio flags
        self.paComplete = pyaudio.paComplete
        self.paContinue = pyaudio.paContinue

    def __iter__(self):
        return self

    def __next__(self):
        # get the desired number of samples (block until all are present)
        try:
            data = self.stream.read(self.stream_hop_size, exception_on_overflow=True)
        except IOError as e:
            print(e, 'stream buffer overflow')
        # convert it to a numpy array
        data = np.fromstring(data, 'float32').astype(self.dtype, copy=False)
        # perform some preprocessing if using custom loopback stream so that trained model is fed data in a shape it expects
        if self.use_system_audio:
            if self.default_wasapi_output_device_info['maxOutputChannels'] > 1:
                # convert data to mono
                data = np.mean(tuple(data[_::self.default_wasapi_output_device_info['maxOutputChannels']] for _ in range(self.default_wasapi_output_device_info['maxOutputChannels'])), axis=0)
            if self.default_wasapi_output_device_info['defaultSampleRate'] != self.sample_rate:
                # resample data
                data = self.resampler.process(data, self.sample_rate / self.default_wasapi_output_device_info['defaultSampleRate'])
        # buffer the data (i.e. append hop_size samples and rotate)
        data = self.buffer(data)
        # wrap the last frame_size samples as a Signal
        # TODO: check float / int hop size; theoretically a float hop size
        #       can be accomplished by making the buffer N samples bigger and
        #       take the correct portion of the buffer
        start = (self.frame_idx * float(self.hop_size) / self.sample_rate)
        signal = Signal(data[-self.frame_size:], sample_rate=self.sample_rate,
                        dtype=self.dtype, num_channels=self.num_channels,
                        start=start)
        # increment the frame index
        self.frame_idx += 1
        return signal

    next = __next__

    def is_running(self):
        return self.stream.is_active()

    def close(self):
        self.stream.close()
        # TODO: is this the correct place to terminate PyAudio?
        self.pa.terminate()

    @property
    def shape(self):
        """Shape of the Stream (None, frame_size[, num_channels])."""
        shape = None, self.frame_size
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels,)
        return shape


def process_stream(processor):
    """
    Like process_online, but operate specifically on StreamLoopbackWASAPI and close when needed.
    """
    # return immediately if should be closed
    if process_stream.close:
        return
    from madmom.processors import _process
    from madmom.audio.signal import Stream, FramedSignal
    # create a Stream with the given arguments, open a stream and start if not running already
    stream = StreamLoopbackWASAPI()
    if not stream.is_running():
        stream.start()
    # set arguments for online processing
    # Note: pass only certain arguments, because these will be passed to the
    #       processors at every time step (kwargs contains file handles etc.)
    process_args = {'reset': False}  # do not reset stateful processors
    # process everything frame-by-frame and close when needed
    for frame in stream:
        if process_stream.close:
            stream.close()
            break
        _process((processor, frame, process_args))