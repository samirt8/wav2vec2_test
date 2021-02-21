import numpy as np
import torch
import array
from torch.utils.data import TensorDataset, DataLoader
import soundfile as sf
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

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

        audio_file_name = os.path.join(self.root_dir, self.transcriptions.iloc[idx, 0])
        audio, _ = sf.read(audio_file_name)
        input_values = tokenizer(audio, padding=True, return_tensors="pt").input_values
        annotation = self.transcriptions.iloc[idx, 1:]
        sample = {'audio': input_values, 'annotation': annotation}

        return sample
