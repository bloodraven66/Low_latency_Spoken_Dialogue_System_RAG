import os
import librosa
import shutil
import soundfile as sf

folder = "results/fd-v1.0-anticipation_exps_mar2026/unmute_smalltalk_no_starter_gemma3_4b/candor_turn_taking/"

for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    audio_path = os.path.join(subfolder_path, "output.wav")
    
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    first_channel = y[0, :]
    second_channel = y[1, :]

    ##first copy current file as output.stereo.wav
    shutil.copy(audio_path, os.path.join(subfolder_path, "output.stereo.wav"))

    sf.write(os.path.join(subfolder_path, "user.wav"), first_channel, sr)
    sf.write(os.path.join(subfolder_path, "output.wav"), second_channel, sr)


    
