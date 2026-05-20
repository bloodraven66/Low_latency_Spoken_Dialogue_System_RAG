
import nemo.collections.asr as nemo_asr
import torchaudio
import torch
# export PYTHONPATH=/mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/nv-one-logger:$PYTHONPATH


model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
    map_location="cpu",
)
model.change_decoding_strategy(decoder_type='ctc')

path = "/mnt/matylda4/udupa/data/LibriLight_10hr/raw_dataset/1h/0/clean/3526/175658/3526-175658-0000.flac" #10.99 sec
waveform = torchaudio.load(path)[0]
assert waveform.shape[0] == 1, "Only mono audio supported"
waveform = waveform[:, :16000 * 10]
length = torch.tensor([waveform.shape[1]])
processed_signal, processed_signal_length = model.preprocessor.get_features(input_signal=waveform, length=length)
encoded, encoded_len = model.encoder(audio_signal=processed_signal, length=processed_signal_length)
#torch.Size([1, 175920]) torch.Size([1, 80, 1100]) torch.Size([1]) torch.Size([1, 512, 139]) tensor([139])
print(waveform.shape, processed_signal.shape, processed_signal_length.shape, encoded.shape, encoded_len)
# output = model.transcribe([path])
# print(output)