#Generates a random batch of audio clips from a directory fo songs for testing

import subprocess
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import time

from Shazam import Shazam
from Song import Song

"""
trim the song using soX
input is the path_to_song, output directory, start of the clip and the duration
"""

def trim_s(song_pth, target_folder, start, duration):

	#new clip has the following name: <name>_start_duration.wav
	new_name = target_folder+song_pth.split('.')[0].split('/')[-1]+'_'+str(start)+'_'+str(duration)+'.wav'
	#use soX to clip the file
	subprocess.call(['./sox', song_pth, new_name, 'trim', str(start), str(duration)])

"""
creates batch of clips to test on
songs_path is the directory which contains the original songs
target_path is the folder where the clips are to be dumped
"""

def create_clips(num_clips, songs_path='audio/', target_path='test/'):

	#parameters which define the lower and upper bounds for the clips
	lower_length, upper_length = 2, 12
	#maximum starting point to clip the audio. Should be lesser than min(duration_of_songs)-upper-length
	start_max = 90
	songs_dir = [x for x in os.listdir(songs_path) if x != '.DS_Store']
	total_clips = len(songs_dir)
	#random batch of start points and the lengths of the clips
	start = np.random.randint(0, high=start_max, size=num_clips)
	lengths = np.random.randint(lower_length, high=upper_length, size=num_clips)

	chosen_songs = np.random.choice(total_clips, num_clips)
	#clip and save the audio files
	for i in range(num_clips):
		trim_s(songs_path+songs_dir[chosen_songs[i]], target_path, start[i], lengths[i])

"""
Do the actual testing by calling the Shazam model on each clip and store the results
Each prediction is classified into 3 categories:
1. correct with time: if the songs is correctly idenitifed along with the point in time at which the clip was trimmed
2. correct: song is correctly identified but the time is wrong
3. wrong: incorrect prediction about the song
set fresh start = True if the directory of clips needs to be cleaned before generating a new batch 
"""
def batch_testing(fresh_start=True, songs_path='audio/', data_path='data/', clips_path='test/'):
	
	#clip files have structure: <name-of-song>_start_dur.wav
	num_clips = 200
	#parameters for the Shazam app
	window_size = 1024
	downsample_factor = 2
	#maximum tolerance for identifying if the point in time at which the clip was trimmed is correct
	time_delta = 1

	#delete existing clips and make a new folder
	if fresh_start:
		shutil.rmtree(clips_path)
		os.mkdir(clips_path)

	#create clips to test
	create_clips(num_clips)
	#initialise the app
	app = Shazam(songs_path, data_path)

	#store results
	stats = {}
	#to track time
	start_time = time.time()
	#total duration of clips. Used for tracking perofromance of the app
	clips_dur = 0

	for song in os.listdir(clips_path):

		if song == '.DS_Store':
			continue

		#identify song, duration and starting point
		song_name = '_'.join(song.split('_')[:-2])
		start, duration = int(song.split('_')[-2]), int(song.split('_')[-1].split('.')[0])
		#update the total duration of clips
		clips_dur += duration

		#for each duration, store a dictionary of the 3 aforementioned categories 
		if duration not in stats.keys():
			stats[duration] = {'correct_time':0, 'correct':0, 'wrong':0}

		#test
		to_test = app.create_song(clips_path+song, try_dumped=True, is_target=True)
		print("Testing on", song_name)
		song_pred, time_pred = app.compare_song(to_test)
		
		#update statistics
		if song_pred == song_name and abs(time_pred-start) < time_delta:
			stats[duration]['correct_time'] += 1
		elif song_pred == song_name:
			stats[duration]['correct'] += 1
		else:
			stats[duration]['wrong'] += 1

	#marks end of prediction phase
	time_taken = time.time() - start_time

	#plot the stastics
	#green - correct with time, blue - correct, red - wrong
	print(stats, time_taken)
	fig, ax = plt.subplots()
	ct, c, w = 0,0,0

	for dur, vals in stats.items():
		egs = sum(vals.values())
		for t_, num in vals.items():
			percent = 100*num/egs
			if t_ == 'correct_time':
				ax.scatter(dur, percent, c='g')
				ct += num
			elif t_ == 'correct':
				ax.scatter(dur, percent, c='b')
				c += num
			else:
				ax.scatter(dur, percent, c='r')
				w += num

	title = "time_taken: " + str(round(time_taken,4)) + "; len_clips: " + str(clips_dur) + 's'
	des = "Correct_time:"+str(ct)+" Correct:"+str(c)+" Wrong:"+str(w)
	plt.title(title+'\n'+des)
	plt.xlabel("Time duration in s")
	plt.ylabel("Percent success")
	plt.grid(True)
	plt.show()

if __name__ == '__main__':

	batch_testing()