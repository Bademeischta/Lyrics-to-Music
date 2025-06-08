"""Pseudo-code post-processing utils."""
from midiutil import MIDIFile  # placeholder

class PostProcessor:
    def tokens_to_midi(self, tokens, path):
        midi = MIDIFile(1)
        # TODO: convert tokens to MIDI events
        midi.writeFile(open(path, 'wb'))

    def midi_to_wav(self, midi_path, wav_path):
        # TODO: call external synthesizer
        pass
