import os
import tempfile
import csv
from src.data.dataset import LyricsMusicDataset
import pretty_midi


def test_dataset_loading(tmp_path):
    csv_file = tmp_path / 'data.csv'
    midi_file = tmp_path / 'test.mid'
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    note = pretty_midi.Note(velocity=100, pitch=60, start=0, end=1)
    instrument.notes.append(note)
    pm.instruments.append(instrument)
    pm.write(str(midi_file))
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lyrics','midi'])
        writer.writeheader()
        writer.writerow({'lyrics':'hello','midi':str(midi_file)})
    ds = LyricsMusicDataset(str(csv_file))
    item = ds[0]
    assert item['lyrics'] == 'hello'
    assert item['midi'].instruments[0].notes[0].pitch == 60
