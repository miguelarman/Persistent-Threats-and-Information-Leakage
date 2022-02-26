import librosa
from librosa.display import specshow
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os

REDUNDANCY = 2
PADDINGWIDTH = 15
BYTESFORLENGTHOFFILENAME = 1
BYTESFORLENGTHOFSECRET = 2


def byte2Bits(byte):
  return int2Bits(int.from_bytes(byte, byteorder="big"))

def int2Bits(int):
  bits = [0,]*8
  n = int
  # print(n)

  for i in range(8):
    bits[7-i] = n%2
    n = n//2
  return bits

def bits2Byte(bits):
  n = 0
  # print(bits)
  for i in range(8):
    n += bits[i] * 2**(7-i)
  return n

def bits2Int(bits):
  return bits2Byte(bits)

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

def length2Bits(length, nbytes):
    # print(length)
    bits = [0,]*8*nbytes
    n = length
    for i in range(8*nbytes):
        bits[8*nbytes-1-i] = n%2
        n = n//2
    # print(bits)
    return bits

def bits2Length(bits, nbytes):
    # print(bits)
    n = 0
    for i in range(nbytes):
        n_ = bits2Int(bits[i*8:(i+1)*8])
        n *= 2**8
        n += n_
    return n


def filename2Bits(filename):
    bits = []
    for byte in filename.encode('utf-8'):
        newbits = int2Bits(byte)
        bits = bits + newbits
    return bits

def bits2Filename(bits):
    # print(bits, len(bits))
    filename = []
    for i in range(0, len(bits), 8):
        b = bits[i:i+8]
        # print(i, b)
        b = bits2Byte(b)
        filename.append(b)
    return bytearray(filename).decode('utf-8')



def showSpectrogram(spectrogram_data, sr):
  specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='log')
  # plt.ylim(0, 2000)

def spectrogram2AudioSignal(spectrogram):
  return librosa.core.spectrum.griffinlim(spectrogram)

def signal2Spectrogram(signal):
  return np.abs(librosa.core.spectrum.stft(signal))



def expandBits(bits):
    result = []
    for bit in bits:
        result += [bit,] * REDUNDANCY
    return result

def invertBits(bits):
    return [1-b for b in bits[::-1]]

def checkAndUnexpandBits(bits):
    # bits[1] are already inverted and modified
    result = []
    # print(bits[0][:16])
    # print(bits[1][:16])
    for i in range(0, len(bits[0]) - len(bits[0])%REDUNDANCY, REDUNDANCY):
        mean1 = sum([bits[0][j] for j in range(i, i + REDUNDANCY)]) / REDUNDANCY
        mean2 = sum([bits[1][j] for j in range(i, i + REDUNDANCY)]) / REDUNDANCY

        if abs(mean1-0.5) > abs(mean2-0.5):
            if mean1 > 0.5:
                bit = 1
            else:
                bit = 0
        else:
            if mean2 > 0.5:
                bit = 1
            else:
                bit = 0

        result.append(bit)

    return result




######

def encode(audioFilename, secretFilename, outputFilename, frequencies):
    try:
        sig, fs = librosa.core.load(audioFilename, mono=False)#, sr=8000)
    except:
        print("Error opening file")
        return
    print(f"Sampling rate is {fs}")

    if sig.shape[0] != 2:
        print("Necessary to use a stereo file")
        exit()

    abs_spectrogram = signal2Spectrogram(sig)
    print("Spectrogram created")

    totalBits = abs_spectrogram.shape[1]
    availableBytes = int(totalBits / 8 / REDUNDANCY)
    print(f"{availableBytes} bytes available")

    try:
        secretBits = file2Bits(secretFilename)
    except:
        print("Error opening file")
        return
    bytesToBeWriten = (BYTESFORLENGTHOFFILENAME + len(secretFilename) + BYTESFORLENGTHOFSECRET + int(len(secretBits) / 8)) * REDUNDANCY
    print(f"Bytes to be written: {bytesToBeWriten}")

    if bytesToBeWriten > availableBytes:
        print("Not enough space")
        exit()


    # Format:
    # 4 bytes with length of filename + filename as bits
    # 4 bytes with length of secret + secret bits
    fullSecretBits = length2Bits(len(secretFilename), BYTESFORLENGTHOFFILENAME) + filename2Bits(secretFilename) + \
                     length2Bits(int(len(secretBits) / 8), BYTESFORLENGTHOFSECRET) + secretBits

    expandedSecretBits = expandBits(fullSecretBits)

    print(f"Bytes expanded: from {int(len(secretBits)/8)} to {int(len(expandedSecretBits)/8)}")

    # maxSize = abs_spectrogram.shape[1] - (abs_spectrogram.shape[1] % 8)
    # if len(expandedSecretBits) > maxSize:
    #     exit("Length too big")

    # print(secretBits[0:16])REDUNDANCY * 4 * 8 + lengthOfFilename * REDUNDANCY + REDUNDANCY * 4 * 8
    # expandedSecretBits += [0,] * (abs_spectrogram.shape[1] - len(expandedSecretBits))

    invertedExpandedSecretBits = invertBits(expandedSecretBits)

    print("Writting bits to frequencies...", end='\r')
    for fr in frequencies: # TODO: Decide whether to write the same to all or append the message
        for i in range(-PADDINGWIDTH, PADDINGWIDTH):
            abs_spectrogram[0][fr+i][:len(expandedSecretBits)] = expandedSecretBits
            abs_spectrogram[1][fr+i][-len(expandedSecretBits):] = invertedExpandedSecretBits

    print("Writting bits to frequencies... OK")

    # showSpectrogram(abs_spectrogram, fs)

    print("Converting spectrogram to audio signal... ", end='\r')
    audio_signal1 = spectrogram2AudioSignal(abs_spectrogram[0])
    print("Converting spectrogram to audio signal... 1/2", end='\r')
    audio_signal2 = spectrogram2AudioSignal(abs_spectrogram[1])
    print("Converting spectrogram to audio signal... OK!")

    # # print(sig, sig.shape)
    # print(abs_spectrogram.shape)
    # # print(audio_signal, audio_signal.shape)

    # write output
    sf.write("auxfile1.wav", audio_signal1, fs, 'PCM_24')
    sf.write("auxfile2.wav", audio_signal2, fs, 'PCM_24')

    left_channel = AudioSegment.from_wav("auxfile1.wav")
    right_channel = AudioSegment.from_wav("auxfile2.wav")
    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)

    os.remove("auxfile1.wav")
    os.remove("auxfile2.wav")

    stereo_sound.export(out_f = outputFilename, format = "wav")

    print(f"Audio signal written to {outputFilename}")

def decode(audioFilename):
    try:
        sig, fs = librosa.core.load(audioFilename, mono=False)#, sr=8000)
    except:
        print("Error opening file")
        return
    print(f"Sampling rate is {fs}")

    if sig.shape[0] != 2:
        print("Necessary to use a stereo file")
        exit()

    abs_spectrogram = signal2Spectrogram(sig)
    print("Spectrogram created")

    for fr in frequencies:
        secretBits = abs_spectrogram[0][fr], invertBits(abs_spectrogram[1][fr])
        print(f"Read {len(secretBits[0]), len(secretBits[1])} bits")

        secretBits = checkAndUnexpandBits(secretBits)
        print(f"Bits unexpanded to {len(secretBits)} bits")

        # TODO: Calculate average or append each frequency

    # print(secretBits[:8])

    secretBits = secretBits[:-(len(secretBits) % 8)]

    lengthOfFilename = bits2Length(  secretBits[                                                                     : (BYTESFORLENGTHOFFILENAME                                                       )*8], BYTESFORLENGTHOFFILENAME)
    filename         = bits2Filename(secretBits[(BYTESFORLENGTHOFFILENAME                                        )*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename                                      )*8])
    lengthOfSecret   = bits2Length(  secretBits[(BYTESFORLENGTHOFFILENAME+lengthOfFilename                       )*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET               )*8], BYTESFORLENGTHOFSECRET)
    secretBits       =               secretBits[(BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET)*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET+lengthOfSecret)*8]

    print(f"Length of filename: {lengthOfFilename}")
    print(f"Filename: {filename}")
    print(f"Length of secret: {lengthOfSecret}")

    bits2File(secretBits, filename)
    print(f"Secret written to {filename} ({lengthOfSecret} bits)")


if __name__ == "__main__":

    # TODO:
    frequencies = [1000,]

    code = input("Choose between encode or decode: ")
    if code == "encode":
        audioFile = input("Filename of the audio cover file: ")
        secretFile = input("Filename of the file to hide: ")
        outputFile = input("Filename of the output file: ")
        encode(audioFile, secretFile, outputFile)
        # pass
    elif code == "decode":
        file = input("What file do you wish to decode? ")
        decode(file)
        # pass
    else:
        print("Code not accepted")
    # print("After encoding")
