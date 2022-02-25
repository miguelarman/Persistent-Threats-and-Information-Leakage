import librosa
from librosa.display import specshow
import numpy as np
import soundfile as sf

REDUNDANCY = 4
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

def length2Bits(length, nbytes): # 4 bytes
  bits = [0,]*8*nbytes
  n = length
  for i in range(8*nbytes):
    bits[8*nbytes-1-i] = n%2
    n = n//2
  return bits

def bits2Length(bits, nbytes):
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

def unexpandBits(bits):
    # secretBits = [1 if b >= 0.5 else 0 for b in bits]
    result = []
    for i in range(0, len(bits) - len(bits)%REDUNDANCY, REDUNDANCY):
        mean = sum([bits[j] for j in range(i, i + REDUNDANCY)]) / REDUNDANCY

        if mean >= 0.5:
            bit = 1
        else:
            bit = 0
        result.append(bit)

    return result




######

def encode(audioFilename, secretFilename, outputFilename):
    try:
        sig, fs = librosa.core.load(audioFilename)#, sr=8000)
    except:
        print("Error opening file")
        return
    print(f"Sampling rate is {fs}")

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

    for i in range(-PADDINGWIDTH, PADDINGWIDTH):
        abs_spectrogram[1000+i][:len(expandedSecretBits)] = expandedSecretBits

    print("Bits written to all frequencies")

    # showSpectrogram(abs_spectrogram, fs)

    print("Converting spectrogram to audio signal... ", end='\r')
    audio_signal = spectrogram2AudioSignal(abs_spectrogram)
    print("Converting spectrogram to audio signal... OK!")

    # # print(sig, sig.shape)
    # print(abs_spectrogram.shape)
    # # print(audio_signal, audio_signal.shape)

    # write output
    sf.write(outputFilename, audio_signal, fs, 'PCM_24')
    print(f"Audio signal written to {outputFilename}")

def decode(audioFilename):
    try:
        sig, fs = librosa.core.load(audioFilename)#, sr=8000)
    except:
        print("Error opening file")
        return
    print(f"Sampling rate is {fs}")

    abs_spectrogram = signal2Spectrogram(sig)
    print("Spectrogram created")

    secretBits = abs_spectrogram[1000]
    print(f"Read {len(secretBits)} bits")
    secretBits = unexpandBits(secretBits)
    print(f"Bits unexpanded to {len(secretBits)} bits")
    secretBits = secretBits[:-(len(secretBits) % 8)]

    filename         = bits2Filename(secretBits[(BYTESFORLENGTHOFFILENAME                                        )*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename                                      )*8])
    lengthOfFilename = bits2Length(  secretBits[                                                                     : (BYTESFORLENGTHOFFILENAME                                                       )*8])
    lengthOfSecret   = bits2Length(  secretBits[(BYTESFORLENGTHOFFILENAME+lengthOfFilename                       )*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET               )*8])
    secretBits       =               secretBits[(BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET)*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET+lengthOfSecret)*8]

    print(f"Length of filename: {lengthOfFilename}")
    print(f"Filename: {filename}")
    print(f"Length of secret: {lengthOfSecret}")

    bits2File(secretBits, filename)
    print(f"Secret written to {filename} ({lengthOfSecret} bits)")


if __name__ == "__main__":
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
