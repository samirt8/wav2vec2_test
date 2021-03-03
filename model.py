from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC


class ASR_CTC(nn.Module):

    def __init__(self):
        super(ASR_CTC, self).__init__()
        self.wav2Vec2Tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base')
        self.wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')
        self.nb_labels = len(self.wav2Vec2Tokenizer.get_vocab())

    def forward(self, x):
        x = self.wav2Vec2ForCTC(x).logits
        #x = nn.functional.log_softmax(self.wav2Vec2ForCTC(x).logits)
        #x = x.view(-1, self.nb_labels)
        return x
