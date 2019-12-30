import array
import numpy as np

from .audio_segment import (
    AudioSegment,
)
from .utils import (
    db_to_float,
)

class AudioBuffer(object):
    """
    AudioBuffers support far fewer operations than AudioSegments but, because
    the AudioBuffer is a mutable object, those operations can be 
    implemented as in-place modification. These in-place operations can be
    more efficient when performing small modifications on large sections of
    audio.
    """
    def __init__(self, data, *args, **kwargs):
        self.sample_width = kwargs.pop("sample_width", None)
        self.frame_rate = kwargs.pop("frame_rate", None)
        self.channels = kwargs.pop("channels", None)

        if isinstance(data, AudioSegment):
            self.sample_width = data.sample_width
            self.frame_rate = data.frame_rate
            self.channels = data.channels
            data = data.get_array_of_samples()

        data = np.array(data)
        if data.dtype.kind == 'i':
            data = data / float(1 << ((data.dtype.itemsize * 8) - 1))
        data = data.reshape(-1, self.channels)
        assert(data.dtype.kind == 'f')

        self._data = data

    def __len__(self):
        """
        returns the length of this audio segment in milliseconds
        """
        return round(1000 * (self.frame_count() / self.frame_rate))

    def segment(self):
        """
        Convert the buffer to an AudioSegment.

        This code would be more efficient if it was migrated into the
        AudioSegment constructor.
        """

        # Convert from float to int16, flatten the channels and
        # (efficiently) convert to the array form that pydub "likes"
        # to import from.
        flattened = (self._data * 32768).astype(np.int16).reshape((-1,))
        data = array.array('h')
        data.frombytes(flattened.tobytes())

        return AudioSegment(
                data = data,
                sample_width = self.sample_width,
                frame_rate = self.frame_rate,
                channels = self.channels)

    def frame_count(self, ms=None):
        """
        returns the number of frames for the given number of milliseconds, or
            if not specified, the number of frames in the whole AudioSegment
        """
        if ms is not None:
            return ms * (self.frame_rate / 1000.0)
        else:
            return len(self._data)

    def fade(self, to_gain=0, from_gain=0, start=None, end=None,
             duration=None):

        # Convert start, end and duration to frame counts and fill
        # in the blanks
        if not start:
            start = end - duration
        if not end:
            end = start + duration
        start = max(int(min(self.frame_count(start), self.frame_count())), 0)
        end = max(int(min(self.frame_count(end), self.frame_count())), 0)
        duration = end - start

        # Apply the fade (if applicable)
        if duration > 0:
            gain_profile = \
                    np.logspace(from_gain/20, to_gain/20, num=duration) \
                        .repeat(self.channels) \
                        .reshape(-1, self.channels)
            self._data[start:end] *= gain_profile

        return self

    def fade_out(self, duration):
        return self.fade(to_gain=-120, duration=duration, end=len(self)+1)

    def fade_in(self, duration):
        return self.fade(from_gain=-120, duration=duration, start=0)

    def overlay(self, seg, position=0, gain_during_overlay=None):
        """
        Overlay the provided segment on to this segment starting at the
        specificed position and using the specfied looping beahvior.

        seg (AudioSegment or AudioBuffer):
            The audio segment or audio buffer to overlay on to this one.

        position (optional int):
            The position to start overlaying the provided segment in to this
            one.

        gain_during_overlay (optional int):
            Changes this segment's volume by the specified amount during the
            duration of time that seg is overlaid on top of it. When negative,
            this has the effect of 'ducking' the audio under the overlay.
        """
        if isinstance(seg, AudioSegment):
            # Resample and match number of channels *before* conversion
            if seg.channels != self.channels:
                seg = seg.set_channels(self.channels)
            if seg.frame_rate != self.frame_rate:
                seg = seg.set_frame_rate(self.frame_rate)

            buf = AudioBuffer(seg)
        else:
            # Audio buffers cannot be resampled... mUst match
            buf = seg

        start = int(self.frame_count(position))
        end = int(min(start + buf.frame_count(), self.frame_count()))

        if end > start:
            if gain_during_overlay:
                self._data[start:end] *= 10 ** (gain_during_overlay / 20)

            self._data[start:end] += buf._data[0:end-start]

        return self
