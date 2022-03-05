import librosa
from librosa.display import specshow
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os
import math
import random
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import getpass

# REDUNDANCY = 2
PADDINGWIDTH = 5
BYTESFORLENGTHOFFILENAME = 1
BYTESFORLENGTHOFSECRET = 2
KEY = b'1234123412341234'
FREQUENCIESOPTIONS = [1020, 1010, 1000, 990, 980, 970, 960, 950]

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
  plt.ylim(0, 2000)
  plt.show()

def spectrogram2AudioSignal(spectrogram):
  return librosa.core.spectrum.griffinlim(spectrogram)

def signal2Spectrogram(signal):
  return np.abs(librosa.core.spectrum.stft(signal))



def expandBits(bits, redundancy):
    result = []
    for bit in bits:
        result += [bit,] * redundancy
    return result

def invertBits(bits):
    return [1-b for b in bits]

def checkAndUnexpandBits(bits, redundancy):
    # bits[1] are already inverted and modified
    result = []
    # print(bits[0][:16])
    # print(bits[1][:16])
    for i in range(0, len(bits[0]) - len(bits[0])%redundancy, redundancy):
        mean1 = sum([bits[0][j] for j in range(i, i + redundancy)]) / redundancy
        mean2 = sum([bits[1][j] for j in range(i, i + redundancy)]) / redundancy

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

def cipherFile(filename, key):
    with open(filename, "rb") as input:
        data = input.read()
        cipher = AES.new(key, AES.MODE_CBC, iv=key)
        ciphered = cipher.encrypt(pad(data, AES.block_size))
        print("Data ciphered")

    with open(filename+".cipher", "wb") as output:
        output.write(ciphered)
        print("Ciphered data writen to temporal file")

def uncipherFile(filename, key):
    with open(filename+".cipher", "rb") as input:
        data = input.read()
        cipher = AES.new(key, AES.MODE_CBC, iv=key)
        print("Read ciphered data")

    with open(filename, "wb") as output:
        output.write(unpad(cipher.decrypt(data), AES.block_size))
        print("Data unciphered and writen")

######

def encode(audioFilename, secretFilename, outputFilename, redundancy, key, frequencies=FREQUENCIESOPTIONS):
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

    totalBits = abs_spectrogram.shape[-1]
    numFreqs = len(frequencies)

    availableBytes = math.ceil(totalBits / 8 / redundancy) * numFreqs
    availableBytesFreq = math.ceil(totalBits / 8)
    print(f"{availableBytes} bytes available")

    try:
        cipherFile(secretFilename, key)
    except:
        print("Error opening file")
        return
    secretBits = file2Bits(secretFilename+".cipher")
    bytesToBeWriten = (BYTESFORLENGTHOFFILENAME + len(secretFilename) + BYTESFORLENGTHOFSECRET + int(len(secretBits) / 8)) * redundancy
    print(f"Bytes to be written: {bytesToBeWriten}")

    numSplit = math.ceil(bytesToBeWriten / availableBytesFreq)
    #!!
    availableBytes = math.ceil(totalBits / 8) * numFreqs

    if bytesToBeWriten > availableBytes:
        print("Not enough space")
        exit()


    # Format:
    # 4 bytes with length of numFeqs + numfreqs
    # 4 bytes with length of filename + filename as bits
    # 4 bytes with length of secret + secret bits
    fullSecretBits = length2Bits(len(secretFilename), BYTESFORLENGTHOFFILENAME) + filename2Bits(secretFilename) + \
                     length2Bits(int(len(secretBits) / 8), BYTESFORLENGTHOFSECRET) + secretBits + [0,]*8

    expandedSecretBits = expandBits(fullSecretBits, redundancy)
    print(f"Bytes expanded: from {int(len(secretBits)/8)} to {int(len(expandedSecretBits)/8)}")

    invertedExpandedSecretBits = invertBits(expandedSecretBits)

    print("Writing bits to frequencies...", end='\r')

    splitedBits = []
    splitedinv = []
    #Split the data in the num of frequencies
    BitsFreq = totalBits
    for i in range(0, (BitsFreq*numSplit), BitsFreq):
        if i == ((BitsFreq * numSplit) - BitsFreq):
            splitedBits.append(expandedSecretBits[i:])
            splitedinv.append(invertedExpandedSecretBits[i:])
        else:
            splitedBits.append(expandedSecretBits[i: (i + BitsFreq)])
            splitedinv.append(invertedExpandedSecretBits[i:(i + BitsFreq)])

    #print("bytestotales " + str(len(invertedExpandedSecretBits)) + " totalBits " + str(totalBits))
    for i in range(numSplit):
        for j in range(-PADDINGWIDTH, PADDINGWIDTH):
            abs_spectrogram[0][frequencies[i]+j][:len(splitedBits[i])] = splitedBits[i]
            abs_spectrogram[1][frequencies[len(frequencies)-i-1]+j][:len(splitedinv[i])] = splitedinv[i]


    print("Writing bits to " + str(numSplit) + " frequencies... OK")

    #showSpectrogram(abs_spectrogram)

    newpid = os.fork()
    print("Converting spectrogram to audio signal... ", end='\r')
    if newpid == 0:
        audio_signal1 = spectrogram2AudioSignal(abs_spectrogram[0])
        sf.write("auxfile1.wav", audio_signal1, fs, 'PCM_24')
        os._exit(0)
    else:
        #print("Converting spectrogram to audio signal... 1/2", end='\r')
        audio_signal2 = spectrogram2AudioSignal(abs_spectrogram[1])
        sf.write("auxfile2.wav", audio_signal2, fs, 'PCM_24')
        os.wait()
    print("Converting spectrogram to audio signal... OK!")


    # write output

    left_channel = AudioSegment.from_wav("auxfile1.wav")
    right_channel = AudioSegment.from_wav("auxfile2.wav")
    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)

    os.remove("auxfile1.wav")
    os.remove("auxfile2.wav")

    stereo_sound.export(out_f = outputFilename, format = "wav")

    print(f"Audio signal written to {outputFilename}")

def decode(audioFilename, key, frequencies=FREQUENCIESOPTIONS):
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

    secrbits = []
    invbits = []
    for fr in frequencies:
        secrbits = np.append(secrbits, abs_spectrogram[0][fr])

    for fr in frequencies[::-1]:
        invbits = np.append(invbits, abs_spectrogram[1][fr])

    secretBits = secrbits, invertBits(invbits)



    print(f"Read {len(secretBits[0]), len(secretBits[1])} bits")

    secretBits = checkAndUnexpandBits(secretBits, redundancy)
    print(f"Bits unexpanded to {len(secretBits)} bits")

        # TODO: Calculate average or append each frequency

    # print(secretBits[:8])

    lengthOfFilename = bits2Length( secretBits[:(BYTESFORLENGTHOFFILENAME)*8], BYTESFORLENGTHOFFILENAME)
    filename         = bits2Filename(secretBits[(BYTESFORLENGTHOFFILENAME                                        )*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename                                      )*8])
    lengthOfSecret   = bits2Length(  secretBits[(BYTESFORLENGTHOFFILENAME+lengthOfFilename                       )*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET               )*8], BYTESFORLENGTHOFSECRET)


    secretBits       =  secretBits[(BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET)*8 : (BYTESFORLENGTHOFFILENAME+lengthOfFilename+BYTESFORLENGTHOFSECRET+lengthOfSecret)*8]

    print(f"Length of filename: {lengthOfFilename}")
    print(f"Filename: {filename}")
    print(f"Length of secret: {lengthOfSecret}")

    bits2File(secretBits, filename+"_decoded.cipher")
    uncipherFile(filename+"_decoded", key)
    print(f"Secret written to {filename}_decoded ({lengthOfSecret} bits)")


if __name__ == "__main__":

    code = input("Choose between encode or decode: ")

    if code == "encode":
        audioFile  = input("Filename of the audio cover file: ")
        secretFile = input("Filename of the file to hide: ")
        outputFile = input("Filename of the output file: ")
        redundancy = input("Redundancy of each byte (the greater, the more robust it is, but the less capacity it has): ")
        redundancy = int(redundancy)
        key        = getpass.getpass(prompt="Key for the cipher: ")

        if len(key) != 16:
            print("Key must be 16 characters")
            exit()
        key = key.encode('utf-8')

        encode(audioFile, secretFile, outputFile, redundancy, key)

    elif code == "decode":
        file       = input("What file do you wish to decode? ")
        redundancy = input("Redundancy of each byte used in encoding: ")
        redundancy = int(redundancy)
        key        = getpass.getpass(prompt="Key used for the cipher: ")

        if len(key) != 16:
            print("Key must be 16 characters")
            exit()
        key = key.encode('utf-8')

        decode(file, key)

    else:
        print("Code not accepted")