from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC


class ASR_CTC(nn.Module):

    def __init__(self):
        super(ASR_CTC, self).__init__()
        self.wav2VecForCTC = Wav2VecForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    def forward(self, x):
        x = self.wav2VecForCTC(x)
        return x