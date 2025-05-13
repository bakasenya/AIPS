import speech_recognition as sr
import pyaudio
import wave
import threading
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize PyAudio and recognizer
p = pyaudio.PyAudio()
recognizer = sr.Recognizer()
stop_words = set(stopwords.words('english'))

# Function to read and save audio to a file
def read_audio(stream, filename):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    seconds = 10
    frames = []
    
    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

    stream.stop_stream()
    stream.close()

# Convert audio to text
def convert(i):
    if i >= 0:
        sound = 'record' + str(i) + '.wav'
        with sr.AudioFile(sound) as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Converting Audio To Text...")
            audio = recognizer.listen(source)
        
        try:
            value = recognizer.recognize_google(audio)
            os.remove(sound)  # Remove the audio file after conversion
            
            # Save the result to a text file
            with open("test.txt", "a") as f:
                f.write(value + " ")
                
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Error with Speech Recognition service: {e}")

# Tokenizing and removing stopwords from text
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    return [word for word in word_tokens if word.lower() not in stop_words]

# Compare common words between speech and question files
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    return list(a_set.intersection(b_set))

# Function to save audio and convert simultaneously
def save_and_convert(i):
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, frames_per_buffer=1024, input=True)
    filename = f'record{i}.wav'
    read_audio(stream, filename)
    convert(i-1)  # Convert previous audio while recording the current one

# Main function to handle the recording and conversion process
def main():
    for i in range(30//10):  # Number of total seconds to record (3 chunks of 10 seconds)
        t1 = threading.Thread(target=save_and_convert, args=[i])
        t1.start()
        t1.join()

    # After recording and converting, process the text
    with open("test.txt", "r") as file:
        speech_data = file.read()
        
    filtered_speech = remove_stopwords(speech_data)
    
    # Read and filter question data
    with open("paper.txt", "r") as file:
        question_data = file.read()
        
    filtered_questions = remove_stopwords(question_data)
    
    # Compare speech and questions for common words
    comm = common_member(filtered_questions, filtered_speech)
    print(f'Number of common elements: {len(comm)}')
    print(comm)

if __name__ == '__main__':
    main()
