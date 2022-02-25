import librosa
from librosa.display import specshow
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os


def byte2Bits(byte):
  bits = [0,]*8
  n = int.from_bytes(byte, byteorder="big")
  # print(n)

  for i in range(8):
    bits[7-i] = n%2
    n = n//2
  return bits

def bits2Byte(bits):
  n = 0
  #print(bits)
  for i in range(8):
    n += bits[i] * 2**(7-i)
  return n

def file2Bits(filename):
  with open(filename, "rb") as f:
    bits = []
    byte = f.read(1)
    while byte != b"":
        newbits = byte2Bits(byte)
        bits = bits + newbits
        # print(bits)
        byte = f.read(1)
    return bits

def bits2File(bits, filename):
  with open(filename, "wb") as f:
    for i in range(0, len(bits), 8):
      bits_ = bits[i:i+8]
      n = bits2Byte(bits_)
      byte = n.to_bytes(1, byteorder="big")
      f.write(byte)


def showSpectrogram(spectrogram_data, sr):
  specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='log')
  plt.ylim(0, 2000)
  plt.show()

def spectrogram2AudioSignal(spectrogram):
    #return librosa.istft(spectrogram)
    return librosa.core.spectrum.griffinlim(spectrogram)

def signal2Spectrogram(signal):
  return np.abs(librosa.core.spectrum.stft(signal))



def expandBits(bits):
    result = []
    for bit in bits:
        result += [bit,]*2
    return result

def unexpandBits(bits):
    # secretBits = [1 if b >= 0.5 else 0 for b in bits]
    result = []
    for i in range(0, len(bits), 2):
        mean = (bits[i] + bits[i+1])/2
        if mean >= 0.5:
            bit = 1
        else:
            bit = 0
        result.append(bit)

    return result

def inverseBits(bits):
    finalb = []
    for b in bits:
        if b == 1:
            finalb.append(0)
        else:
            finalb.append(1)
    return finalb


def meanBits(mean):
    if mean >= 0.5:
        return 1
    else:
        return 0
def inverseMeanBits(mean):
    if mean >= 0.5:
        return 0
    else:
        return 1

def checkBits(bits1, bits2):
    # secretBits = [1 if b >= 0.5 else 0 for b in bits]
    result = []
    for i in range(0, len(bits1)):
        mean1 = meanBits(bits1[i])
        mean2 = inverseMeanBits(bits2[i])
        if mean1 == mean2:
            result.append(mean1)
        else:
            if abs(bits1[i]-0.5) > abs(bits2[i]-0.5):
                result.append(mean1)
            else:
                result.append(mean2)

    return result

######

def encode(audioFilename, secretFilename, outputFilename):
    sig, fs = librosa.core.load(audioFilename, sr=8000, mono=False)

    if sig.shape[0] < 2:
        print("No 2 channels")


    abs_spectrogram = signal2Spectrogram(sig)
    secretBits = file2Bits(secretFilename)
    #secretBits = expandBits(secretBits)
    maxSize = abs_spectrogram[0].shape[1] - (abs_spectrogram[0].shape[1] % 8)
    if len(secretBits) > maxSize:
        exit("Length too big")
    secretBits += [0,] * (abs_spectrogram[0].shape[1] - len(secretBits))

    changed_spectrogram1 = changeSpectrogram(abs_spectrogram[0], secretBits, fs)

    changed_spectrogram2 = changeSpectrogram(abs_spectrogram[1], inverseBits(secretBits[::-1]), fs)

    audio_signal1 = spectrogram2AudioSignal(changed_spectrogram1)
    audio_signal2 = spectrogram2AudioSignal(changed_spectrogram2)

    print("Audio signal converted to spectrogram")

    sf.write("auxfile1.wav", audio_signal1, fs, 'PCM_24')
    sf.write("auxfile2.wav", audio_signal2, fs, 'PCM_24')



    # print(abs_spectrogram.shape)
    # # print(audio_signal, audio_signal.shape)
    left_channel = AudioSegment.from_wav("auxfile1.wav")
    right_channel = AudioSegment.from_wav("auxfile2.wav")
    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)

    os.remove("auxfile1.wav")
    os.remove("auxfile2.wav")

    stereo_sound.export(out_f = outputFilename,
                       format = "wav")




def changeSpectrogram(abs_spectrogram, secretBits, fs):
    # print(secretBits[0:16])

    for i in range(-15, 15):
        abs_spectrogram[1000+i] = secretBits

    #showSpectrogram(abs_spectrogram, fs)


    return abs_spectrogram


def decode(audioFilename):
    sig, fs = librosa.core.load(audioFilename, sr=8000, mono=False)
    print(sig.shape)

    abs_spectrogram1 = signal2Spectrogram(sig)[0]
    abs_spectrogram2 = signal2Spectrogram(sig)[1]

    secretBits1 = abs_spectrogram1[1000]
    secretBits2 = abs_spectrogram2[1000][::-1]
    #secretBits = unexpandBits(secretBits)
    secretBits = checkBits(secretBits1, secretBits2)
    # print(abs_spectrogram[1000][8:16])
    # print(secretBits[8:16])
    # print('\n'*5)
    secretBits = secretBits[:-(len(secretBits)%8)]
    bits2File(secretBits, "file2.txt")


if __name__ == "__main__":
    encode("SineWaveMinus16.wav", "file.txt", "output.wav")
    print("After encoding")
    decode("output.wav")
