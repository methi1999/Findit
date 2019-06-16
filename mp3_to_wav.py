#Convert music files to .wav format and store in the same folder

from pydub import AudioSegment
import os

#directory of input songs
music_dir = '.'

#file formats to convert into wav
file_formats = ['m4a', 'mp3']

for file in os.listdir(music_dir):
	if file.split('.')[-1] in file_formats:
		print("Working on file", file)
		new_name = ''.join(file.split('.')[:-1])+'.wav'
		AudioSegment.from_file(file).export(new_name, format="wav")
