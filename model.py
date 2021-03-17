from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC


class ASR_CTC(nn.Module):

    def __init__(self):
        super(ASR_CTC, self).__init__()
        #self.wav2Vec2Tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base')
        #self.wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')
        #self.nb_labels = len(self.wav2Vec2Tokenizer.get_vocab())
        #self.wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')
        self.tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]",
                                                      word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53",
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            gradient_checkpointing=True,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer)
        )


    def forward(self, x):
        x = self.wav2Vec2ForCTC(x)
        #x = self.wav2Vec2ForCTC(x).logits
        #x = nn.functional.log_softmax(self.wav2Vec2ForCTC(x).logits)
        #x = x.view(-1, self.nb_labels)
        return x
