import os
import subprocess
import unidecode
import numpy as np
import pandas as pd
import torch
import array
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pydub import AudioSegment
from keras.preprocessing.sequence import pad_sequences
from transformers import Wav2Vec2Tokenizer


class AudioDataset(Dataset):
    """Audio dataset"""

    def __init__(self, transcription_file, root_dir, MAX_LEN):
        """
        :param transcription_file: Path to the text transcription.
        :param root_dir: Directory containing audio files.
        :param MAX_LEN: Maximum number of characters for the output. We trunk or pad output to this length 
        """
        self.transcriptions = pd.read_csv(transcription_file, sep="\t")
        self.root_dir = root_dir
        self.MAX_LEN = MAX_LEN
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")


    def __len__(self):
        return len(self.transcriptions)


    def clean_annotation(self, text):
        """
        Function to clean annotation text
        We need to uppercase the text, replace space with | and remove punctuation except '
        """
        text = text.upper()
        # remove accents
        text = unidecode.unidecode(text) 
        # output word
        output = ""
        # output vector
        vect_output = []
        for char in text:
            if char in list(self.tokenizer.get_vocab().keys()):
                output += char
            elif char == " ":
                output += "|"
        for char in output:
            vect_output.append(self.tokenizer.get_vocab()[char])
        vect_output = np.squeeze(pad_sequences([vect_output], maxlen=self.MAX_LEN, dtype="long", value=0, truncating="post", padding="post"), 0)
        return vect_output


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tokenizer = self.tokenizer
        
        # temp for english
        #root_dir = os.path.dirname(self.root_dir)
        audio_file_name = os.path.join(self.root_dir, self.transcriptions.iloc[idx, 1])
        audio_file_name_mp3 = audio_file_name+".mp3"
        audio_file_name_wav = audio_file_name+".wav"
        #audio_file_name_mp3 = audio_file_name
        #audio_file_name_wav = audio_file_name[0:-3]+"wav"
        # convert mp3 to wav
        sound = AudioSegment.from_mp3(audio_file_name_mp3)
        sound = sound.set_frame_rate(16000)
        sound.export(audio_file_name_wav, format="wav")
        audio, _ = sf.read(audio_file_name_wav)
        os.remove(audio_file_name_wav)
        input_values = tokenizer(audio, max_length=350000, padding="max_length", return_tensors="pt").input_values
        input_values = torch.squeeze(input_values, 0)
        annotation = self.transcriptions.iloc[idx, 2]

        sample = {'audio': input_values, 'annotation': self.clean_annotation(annotation)}
        return sample
