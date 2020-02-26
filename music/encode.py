# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:40:44 2020

Encoding and filtering input
midi files -> music21 streams -> numpy array -> text encodings

@author: srava
"""

import music21
import numpy as np

#import vocab
from vocab import *
from enum import Enum


BPB = 4 # beats per bar
TIMESIG = f'{BPB}/4' # default time signature
PIANO_RANGE = (21, 108)
VALTSEP = -1 # separator value for numpy encoding
VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array

SAMPLE_FREQ = 4
NOTE_SIZE = 128
DUR_SIZE = (10*BPB*SAMPLE_FREQ)+1 # Max length - 8 bars. Or 16 beats/quarternotes
MAX_NOTE_DUR = (8*BPB*SAMPLE_FREQ)


SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')

#1 Load the midi file
#2 Convert midi to Music21 stream
#3 Turn stream in into numpy chord array
#4 numpy array -> List[Timestep][NodeEncoding]

class MusicEncode():
    
    # Midi Encoding Method
    def midi_enc(self, midi_file, skip_last_rest=True):
        stream = self.file_stream(midi_file)
        chord_arr = self.stream_chordarr(stream)
        enc =  self.chordarr_npenc(chord_arr, skip_last_rest=skip_last_rest)
        vocab = MusicVocab.create()
        return self.npenc2idxenc(enc, vocab)
    
    #File to stream
    def file_stream(self, path):
        if isinstance(path, music21.midi.MidiFile): return music21.midi.translate.midiFileToStream(path)
        return music21.converter.parse(path)
    
    #Stream to array
    def stream_chordarr(self, stream, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
        """
        4/4 time
        note * instrument * pitch
        """
        
        highest_time = max(stream.flat.getElementsByClass('Note').highestTime, stream.flat.getElementsByClass('Chord').highestTime)
        maxTimeStep = round(highest_time * sample_freq)+1
        score_arr = np.zeros((maxTimeStep, len(stream.parts), NOTE_SIZE))
    
        def note_data(pitch, note):
            return (pitch.midi, int(round(note.offset*sample_freq)), int(round(note.duration.quarterLength*sample_freq)))
    
        for idx,part in enumerate(stream.parts):
            notes=[]
            for elem in part.flat:
                if isinstance(elem, music21.note.Note):
                    notes.append(note_data(elem.pitch, elem))
                if isinstance(elem, music21.chord.Chord):
                    for p in elem.pitches:
                        notes.append(note_data(p, elem))
                    
            # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
            notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
            for n in notes_sorted:
                if n is None: continue
                pitch,offset,duration = n
                if max_note_dur is not None and duration > max_note_dur: duration = max_note_dur
                score_arr[offset, idx, pitch] = duration
                score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding note
        return score_arr
    
    # Convert array to NP Encoding
    def chordarr_npenc(self, chordarr, skip_last_rest=True):
        # combine instruments
        result = []
        wait_count = 0
        for idx,timestep in enumerate(chordarr):
            flat_time = self.timestep2npenc(timestep)
            if len(flat_time) == 0:
                wait_count += 1
            else:
                # pitch, octave, duration, instrument
                if wait_count > 0: result.append([VALTSEP, wait_count])
                result.extend(flat_time)
                wait_count = 1
        if wait_count > 0 and not skip_last_rest: result.append([VALTSEP, wait_count])
        return np.array(result, dtype=int).reshape(-1, 2) # reshaping. Just in case result is empty
    
    # Note: not worrying about overlaps - as notes will still play. just look tied
    def timestep2npenc(self, timestep, note_range=PIANO_RANGE, enc_type=None):
        # inst x pitch
        notes = []
        for i,n in zip(*timestep.nonzero()):
            d = timestep[i,n]
            if d < 0: continue # only supporting short duration encoding for now
            if n < note_range[0] or n >= note_range[1]: continue # must be within midi range
            notes.append([n,d,i])
            
        notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)
        
        if enc_type is None: 
            # note, duration
            return [n[:2] for n in notes] 
        if enc_type == 'parts':
            # note, duration, part
            return [n for n in notes]
        if enc_type == 'full':
            # note_class, duration, octave, instrument
            return [[n%12, d, n//12, i] for n,d,i in notes] 
    
    
    # Convering np encodings into Tensors for use in model
        # single stream instead of note,dur
    def npenc2idxenc(self, t, vocab, seq_type=SEQType.Sentence, add_eos=False):
        "Transforms numpy array from 2 column (note, duration) matrix to a single column"
        "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
        if isinstance(t, (list, tuple)) and len(t) == 2: 
            return [self.npenc2idxenc(x, vocab) for x in t]
        t = t.copy()
        
        t[:, 0] = t[:, 0] + vocab.note_range[0]
        t[:, 1] = t[:, 1] + vocab.dur_range[0]
        
        prefix = self.seq_prefix(seq_type, vocab)
        suffix = np.array([vocab.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)
        return np.concatenate([prefix, t.reshape(-1), suffix])
    
    def seq_prefix(self, seq_type, vocab):
        if seq_type == SEQType.Empty: return np.empty(0, dtype=int)
        start_token = vocab.bos_idx
        if seq_type == SEQType.Chords: start_token = vocab.stoi[CSEQ]
        if seq_type == SEQType.Melody: start_token = vocab.stoi[MSEQ]
        return np.array([start_token, vocab.pad_idx])
    

if __name__ == '__main__':
    #print(midi_enc("bwv772.mid"));
    midi_file = "data/bwv772.mid"
    
    encode = MusicEncode()
    stream = encode.file_stream(midi_file)
    #stream.show('text')
    #stream.show()
    #stream.show('midi')

    chord_arr = encode.stream_chordarr(stream)
    print(chord_arr)
    ts1 = chord_arr[0].nonzero()
    c = music21.chord.Chord(ts1[1].tolist())
    print(ts1)
    print(c)


    enc = encode.chordarr_npenc(chord_arr, True)
    print(enc.shape)
    print(enc)
    
    print('Chord encoding size: ', np.prod(chord_arr.shape), 'Note encoding size:', np.prod(enc.shape))
    
    n = enc[0:1]
    print(n)
    c = enc[1:4]
    print(c)
    print(enc[:8])
    
    vocab = MusicVocab.create()
    
    idxenc = encode.npenc2idxenc(enc, vocab)
    print(idxenc.shape)
    print(idxenc)
    print(idxenc[0:5])
    print(idxenc[0:5].shape)
    print(vocab.stoi.items())
    
    