from os import path
import numpy as np
import torch
import array
from torch.utils.data import TensorDataset, DataLoader
#import soundfile as sf
from pydub import AudioSegment
from transformers import Wav2Vec2Tokenizer


class AudioDataset(Dataset):
    """Audio dataset"""

    def __init__(self, transcription_file, root_dir):
        """
        :param transcription_file: Path to the text transcription.
        :param root_dir: Directory containing audio files.
        """
        self.transcriptions = pd.read_csv(transcription_file)
        self.root_dir = root_dir
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    def __len__(self):
        return len(self.transcriptions)

    def clean_annotation(self, text):
        """
        Function to clean annotation text
        We need to uppercase the text, replace space with | and remove punctuation except '
        """
        text = text.upper()
        output = ""
        for char in text:
            if char in list(self.tokenizer.get_vocab().keys()):
                output += char
            elif char == " ":
                output += "|"
        return output


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

        audio_file_name = os.path.join(self.root_dir, self.transcriptions.iloc[idx, 1])
        audio_file_name_mp3 = audio_file_name+".mp3"
        audio_file_name_wav = audio_file_name+".wav"
        # convert mp3 to wav
        sound = AudioSegment.from_mp3(audio_file_name_mp3)
        sound.export(audio_file_name_wav, format="wav")
        audio, _ = sf.read(audio_file_name_wav)
        os.remove(audio_file_name_wav)
        input_values = tokenizer(audio, padding=True, return_tensors="pt").input_values
        annotation = self.transcriptions.iloc[idx, 2]

        sample = {'audio': input_values, 'annotation': clean_annotation(annotation)}

        return sample
