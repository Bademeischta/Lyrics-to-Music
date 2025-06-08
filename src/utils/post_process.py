"""Utility functions for turning decoder output into audio."""
from midiutil import MIDIFile
import subprocess

class PostProcessor:
    def tokens_to_midi(self, tokens, path):
        midi = MIDIFile(1)
        time = 0
        for t in tokens:
            midi.addNote(track=0, channel=0, pitch=int(t), time=time,
                         duration=1, volume=100)
            time += 1
        with open(path, 'wb') as f:
            midi.writeFile(f)

    def midi_to_wav(self, midi_path, wav_path):
        cmd = [
            "fluidsynth",
            "-ni",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            midi_path,
            "-F",
            wav_path,
            "-r",
            "44100",
        ]
        subprocess.run(cmd, check=True)
