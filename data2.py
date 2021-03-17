import os
import re
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
from transformers import Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


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
        self.tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)


    def __len__(self):
        return len(self.transcriptions)


    def clean_annotation(self, text):
        """
        Function to clean annotation text
        We need to uppercase the text, replace space with | and remove punctuation except '
        """

        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
        text = re.sub(chars_to_ignore_regex, '', text).upper()

        # output word
        output = ""
        for char in text:
            if char == " ":
                output += "|"
            else:
                output += char
        return output


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_name = os.path.join(self.root_dir, self.transcriptions.iloc[idx, 1])
        audio_file_name_mp3 = audio_file_name+".mp3"
        audio_file_name_wav = audio_file_name+".wav"
        # convert mp3 to wav
        sound = AudioSegment.from_mp3(audio_file_name_mp3)
        sound = sound.set_frame_rate(16000)
        sound.export(audio_file_name_wav, format="wav")
        audio, _ = sf.read(audio_file_name_wav)
        os.remove(audio_file_name_wav)

        #audio
        batch = self.processor.pad(audio, padding=True, max_length=350000, sampling_rate=16000, return_tensors="pt")

        #annotation
        annotation = self.clean_annotation(self.transcriptions.iloc[idx, 2])
        with self.processor.as_target_processor():
            labels_batch =self.processor.pad(annotation, padding=True, max_length=100, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch
