import numpy as np 
import os          
import subprocess  
import telebot     
import pickle
from datetime import datetime # generate log
from scipy.io.wavfile import read
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from numpy.testing import rundocs
from scipy.io.wavfile import read, write

import pickle
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

class Segment:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

def mask_compress(data):
    segments = [];
    if len(data) == 0:
        return segments
    start = -1
    stop = -1
    if data[0] == 1:
        start = 0
    for i in range(len(data) - 1):
        if data[i] == 0 and data[i + 1] == 1:
            start = i + 1;
        if data[i] == 1 and data[i + 1] == 0:
            stop = i + 1;
            segments.append(Segment(start, stop));
    if data[-1] == 1:
        stop = len(data)
        segments.append(Segment(start, stop));
    return segments

def print_with_timeline(data, single_duration, units_name, row_limit):
    for i in range(len(data)):
        if i % row_limit == 0:
            print(f"{single_duration * i:8.3f} {units_name} |  ", end='')
        print(f"{data[i]:.3f} ", end='')
        if (i + 1) % row_limit == 0 or i + 1 == len(data):
            print(f" | {single_duration * (i + 1):8.3f} {units_name}")

def get_segment_energy(data, start, end):
    energy = 0
    for i in range(start, end):
        energy += float(data[i]) * data[i] / (end - start)
    energy = np.sqrt(energy) / 32768
    return energy

def get_segments_energy(data, segment_duration):
    segments_energy  = []
    for segment_start in range(0, len(data), segment_duration):
        segment_stop = min(segment_start + segment_duration, len(data))
        energy = get_segment_energy(data, segment_start, segment_stop)
        segments_energy.append(energy)
    return segments_energy

def get_vad_mask(data, threshold):
    vad_mask = np.zeros_like(data)
    for i in range(0, len(data)):
        vad_mask[i] = data[i] > threshold
    return vad_mask

def sec2samples(seconds, sample_rate):
  return int(seconds * sample_rate)

bot = telebot.TeleBot("1223432479:AAExeQr0t_p7pAHc0L3wr4tgowVvzsZ-mUM")
root = os.getcwd() + "/dataset/"

def save_ogg(ogg_data, ogg_path):
    with open(ogg_path, "wb") as file:
        file.write(ogg_data)

def convert_ogg_wav(ogg_path, dst_path):
    rate = 48000
    cmd = f"ffmpeg -i {ogg_path} -ar {rate} {dst_path} -y -loglevel panic"
    log(cmd)
    with subprocess.Popen(cmd.split()) as p:
        try:
            p.wait(timeout=2)
        except:
            p.kill()
            p.wait()
            return "timeout"

def log(text):
    time_stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(time_stamp + " " + text)

def vad(wav_file_path, user):
    segment_duration = 0.1
    vad_threshold = 0.07
    sample_rate, audio = read(wav_file_path)
    segment_duration_samples = sec2samples(segment_duration, sample_rate)
    segments_energy = get_segments_energy(audio, segment_duration_samples)
    vad_mask = get_vad_mask(segments_energy, vad_threshold)
    segments = mask_compress(vad_mask)
    #assert 1 == len(segments), "Bad threshhold or in "
    max_duration = 0
    for segment in segments:
        duration = (segment.stop - segment.start) * segment_duration_samples / sample_rate
        if duration > max_duration:
            max_duration = duration
    assert max_duration <= 0.6, f"max_duration={max_duration:.3f}"
    wav_path_after_vad = root + f"{user}.wav"
    start = segments[0].start * segment_duration_samples
    stop = segments[0].stop * segment_duration_samples
    f = open(wav_path_after_vad, 'w')
    write(wav_path_after_vad, sample_rate, audio[start:stop])
    f.close()
    return wav_path_after_vad

def predict(wav_path_after_vad):
    filename = "model.pkl"
    with open(filename, 'rb') as f:
        model_pickled = f.read()    
    model = pickle.loads(model_pickled)
    sample_rate, audio = read(wav_path_after_vad)
    max_duration_sec = 0.6

    max_duration = int(max_duration_sec * sample_rate + 1e-6)
    if len(audio) < max_duration:
        audio = np.pad(audio, (0, max_duration - len(audio)), constant_values = 0)
    
    assert len(audio) <= max_duration, "very long file"

    feature = librosa.feature.melspectrogram(audio.astype(float), sample_rate, n_mels = 80, fmax = 4000)
    print(feature.shape)
    features_flatten = feature.reshape(-1)
    print(features_flatten.shape)
    answer = model.predict([features_flatten])[0]
    return answer

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    user = message.from_user.id
    text = message.text
    log(f"User ({user}): {text}")

    bot.send_message(user,
        f"Пожалуйста отправьте голосовое сообщение с цифрой.")


@bot.message_handler(content_types=['voice'])
def get_voice_messages(message):
    user = message.from_user.id
    voice = message.voice
    log(f"User ({user}): voice")

    tele_file = bot.get_file(voice.file_id)
    ogg_data = bot.download_file(tele_file.file_path)
    file_name = "inference_file" # need to generate uniq name
    ogg_path = root + "/ogg/" + file_name + ".ogg"
    wav_path = root + "/wav/" + file_name + ".wav"
    save_ogg(ogg_data, ogg_path)
    convert_ogg_wav(ogg_path, wav_path)
    # ... todo
    wav_path_after_vad = vad(wav_path, user)
    answer = predict(wav_path_after_vad)
    bot.send_message(user, str(answer))

if __name__ == "__main__":
    bot.polling(none_stop=True, interval=0)

