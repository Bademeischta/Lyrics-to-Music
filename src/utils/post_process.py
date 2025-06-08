"""Post-processing utilities for converting tokens to audio."""
from midiutil import MIDIFile
import subprocess

class PostProcessor:
    def tokens_to_midi(self, tokens, path, tempo: int = 120):
        """Convert integer tokens to a simple monophonic MIDI file."""
        midi = MIDIFile(1)
        midi.addTempo(0, 0, tempo)
        time = 0
        for tok in tokens:
            pitch = int(tok)
            midi.addNote(0, 0, pitch, time, 0.5, 100)
            time += 0.5
        with open(path, "wb") as f:
            midi.writeFile(f)

    def midi_to_wav(self, midi_path, wav_path, soundfont: str = "/usr/share/sounds/sf2/FluidR3_GM.sf2"):
        """Render MIDI to audio using FluidSynth if installed."""
        cmd = ["fluidsynth", "-ni", soundfont, midi_path, "-F", wav_path, "-q"]
        subprocess.run(cmd, check=True)
