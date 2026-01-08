import generate_fsm as gf
import json

gf.changeRaga("jog")

phrases = [
    "XX " + p for p in json.load(open("./raga_data_jog/jog.json"))["new_phrases"]
]

gf.create_midi_file(
    phrases,
    gf.swar2midi_map,
    gf.MIDI_TONIC_NOTE,
    gf.MIDI_BPM,
    gf.OUTPUT_MIDI_FILE,
    gf.MIDI_TICKS_PER_BEAT,
)
