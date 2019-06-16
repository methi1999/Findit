import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import operator

from Song import Song
"""
The main driver which creates database of songs, stores their audio fingerprint and compares it with an input clip
"""
class Findit():

	def __init__(self, audio_path, data_path):

		self.audio_path = audio_path #contains music .wav files
		self.data_path = data_path #path for storing temporary data such as hashes and spectograms
		self.keep_coeff = 1 #Above mean to keep in spectogram
		# self.delta = 2 #Frequerncy bins difference tolerance level. Only for brute force search
		self.keep_targets_ratio = 0.8 #Number of common targets threshold during final filtering
		self.overlap_factor = 0.5 #percentage of overlap between the windows during FFT calculation
		self.window_size = 1024 
		self.sampling_freq = 22100 #since most audible enrgy is below 10.05 KHz (by Nyquist)
		self.round_off = 6 #Round of the floating point numbers

		#database.pkl contains the audio fingerprint for each song in the form of a hash table 

		if os.path.exists(data_path+'database.pkl'):
			with open(data_path+'database.pkl', 'rb') as f:
				self.database = pickle.load(f)
			print("Loaded database from pickle dump")
		else:
			self.database = {}
			self.num_songs = 0

		self.anchor_delay = 3 #point which acts as an anchor for a target zone of size self.target_zone
		self.target_zone = 5 #size of target zone

#Dump spectogram pickle dump for each song in the folder self.data_path
	def make_spectogram_database(self):

		for song_name in os.listdir(self.audio_path):
			
			if song_name == '.DS_Store' or os.path.exists(self.data_path+'data_'+song_name.split('/')[-1].split('.')[0]):
				continue
			
			s = self.create_song(self.audio_path+song_name, try_dumped=True, is_target=False)
			s.dump(base_pth=self.data_path, dump_data=False)

#creates and returns a sound object for Findit by filling in thee required hyperparameters
	def create_song(self, filename, try_dumped, is_target=False):

		s = Song(filename, self.window_size, self.keep_coeff, try_dumped=True, sampling_freq=self.sampling_freq, overlap=self.overlap_factor, is_target=is_target)
		return s

	"""
	Creates the hash of one song and stores it in the data/ directory
	is_target = True while testing the app. It does not store the pickle dump since we won't need it in the future
		  	= False for audio files in the database
	A couple is defined as the following key-value pair: key = (anchor freq, point freq, delta time), value = (absolute time of anchor, song id)
	"""

	def hash_one_song(self, song, is_target):

		print("Working on",song.name)

		#check if filtered spectogram exists. if not, call the function
		try:
			x = song.filtered
			print("Found filtered spectogram for",song.name,"in pickle dump")
		except:
			x = song.fft_and_mask(plot_spec=False, plot_filtered=False)
			if not is_target:
				song.dump(base_pth=self.data_path, dump_data=False)

		#x has structure: {'time_slice':list_of_preserved_frequencies: e.g.: '1':[freq, freq2, etc.], '2':[freq1, freq2, etc.]}

		time_res = song.time_res
		#number the points starting with smaller time_slice_no and smaller_freq
		numbered_pts = {}
		i = 0
		for time in sorted(x.keys()):
			for freq in sorted(x[time]):
				numbered_pts[i] = (time, freq)
				i += 1

		song.total_pts = i
		anchor = 0
		print("Total freq points in song",song.name,"are",song.total_pts)
		#database is stored asa dictionary with: key = (anchor freq, point freq, delta time), value = (absolute time of anchor, song id)
		total_couples = 0

		#if it is an audio file to be added to the database, we use song_name as song_id
		if not is_target:

			#for the first (anchor_delay) points, the anchor point is the first point
			freq_anchor, time_anchor = numbered_pts[0][1], numbered_pts[0][0]

			for start in range(self.anchor_delay):
				freq_pt, time_pt = numbered_pts[start][1], numbered_pts[start][0]
				cur_key = (freq_anchor, freq_pt, round((time_pt-time_anchor)*time_res, self.round_off))
				cur_val = (round(time_anchor*time_res, self.round_off), song.name)
				total_couples +=1 

				if cur_key not in self.database.keys():
					self.database[cur_key] = []
				self.database[cur_key].append(cur_val)

			#eahc point will act as an anchor for 5 points in the spectogram with strong frequencies
			for anchor in range(song.total_pts-self.target_zone-self.anchor_delay+1):
				
				freq_anchor, time_anchor = numbered_pts[anchor][1], numbered_pts[anchor][0]
				#target zone stretches from anchor_point+anchor_delay to anchor_point+anchor_delay+target_zone_suze
				for target_pt in range(anchor+self.anchor_delay, anchor+self.anchor_delay+self.target_zone):
					freq_pt, time_pt = numbered_pts[target_pt][1], numbered_pts[target_pt][0]
					cur_key = (freq_anchor, freq_pt, round((time_pt-time_anchor)*time_res, self.round_off))
					cur_val = (round(time_anchor*time_res, self.round_off), song.name)
					total_couples +=1 

					if cur_key not in self.database.keys():
						self.database[cur_key] = []
					self.database[cur_key].append(cur_val)

			song.total_couples = total_couples
			print("Total couples for song",song.name,'are',total_couples)

		else:

			#Identical to the if block, except we don't store the couples in the database but we return it as a database since we want to test the input
			freq_anchor, time_anchor = numbered_pts[0][1], numbered_pts[0][0]

			to_return = {}
			total_couples = 0 

			for start in range(self.anchor_delay):
				freq_pt, time_pt = numbered_pts[start][1], numbered_pts[start][0]
				cur_key = (freq_anchor, freq_pt, round((time_pt-time_anchor)*time_res, self.round_off))
				cur_val = round(time_anchor*time_res, self.round_off)

				if cur_key not in to_return.keys():
					to_return[cur_key] = []
				to_return[cur_key].append(cur_val)
				total_couples +=1 

			for anchor in range(song.total_pts-self.target_zone-self.anchor_delay+1):
				
				freq_anchor, time_anchor = numbered_pts[anchor][1], numbered_pts[anchor][0]
			
				for target_pt in range(anchor+self.anchor_delay, anchor+self.anchor_delay+self.target_zone):
					freq_pt, time_pt = numbered_pts[target_pt][1], numbered_pts[target_pt][0]
					cur_key = (freq_anchor, freq_pt, round((time_pt-time_anchor)*time_res, self.round_off))
					cur_val = round(time_anchor*time_res, self.round_off)

					if cur_key not in to_return.keys():
						to_return[cur_key] = []
					to_return[cur_key].append(cur_val)
					total_couples +=1

			print("Total couples for song",song.name,'are',total_couples) 

			return to_return
	
#Hash all the songs in the audio/ folder and store it as a pkl file
	def hash_database(self):

		for songname in sorted(os.listdir(self.audio_path)):
			if songname == '.DS_Store':
				continue
			cur_song = self.create_song(self.audio_path+songname, try_dumped=True)
			self.hash_one_song(cur_song, is_target=False)
			print("Hashed",cur_song.name)

		with open(self.data_path+'database.pkl', 'wb') as f:
			pickle.dump(self.database, f)

		print("Dumped hash database")

#Return only those couples in the database whose song_id matches the names in the list of desired song_names
	def filter_database_song(self, song_names):

		to_return = {}
		for name in song_names:
			to_return[name] = {}

		for freq, details in self.database.items():

			for option in details:
				time, name = option[0], option[1]
				if name in song_names:
					if freq not in to_return[name].keys():
						to_return[name][freq] = []
					to_return[name][freq].append(time)
		return to_return

	"""
	Compares the input audio with all the songs int he database and returns the best match
	Always returns the best match. Can be configured to return 'No match' if matching notes/similarity fall below a threshold
	"""
	def compare_song(self, song):

		#Hash is of the structure: key = (anchor freq, point freq, delta time) value = (absolute time of anchor, song id)
		cur_song_hash = self.hash_one_song(song, is_target=True)
		
		#Hash the audio files in the database
		if len(self.database) == 0:
			self.hash_database()

		#Find couples from database which match couples in target audio
		matching = []
		
		for key in cur_song_hash.keys():
			if key in self.database.keys():
				matching.append((key[0], self.database[key]))
				
		
		#Weed out non-target zones couples since if there is a common target zone
		#the point must appear atleast (target_zone_size) number of times since every point(excluding at the start and the end) is part of (target_zone) number of zones
		hits = {}
		for matching_couple in matching:
			anchor_freq = matching_couple[0]
			for time_id in matching_couple[1]:
				if (anchor_freq, time_id) not in hits.keys():
					hits[(anchor_freq, time_id)] = 0
				hits[(anchor_freq, time_id)] += 1
		
		filtered_hash = [key for key, vals in hits.items() if vals >= self.target_zone]
		
		#Find number of common target zones
		common_targets = {}
		for filtered_pair in filtered_hash:
			song_name = filtered_pair[1][1]
			if song_name not in common_targets.keys():
				common_targets[song_name] = 0
			common_targets[song_name] += 1
		
		#If number of target zones are above the (number of zones in the current audio clip)*(a coefficient)

		songs_to_consider = []
		for name, com_target in common_targets.items():
			if com_target >= song.total_pts*self.keep_targets_ratio:
				songs_to_consider.append(name)

		#We could end the search and declare 'no match found' or look for the best option by sorting the number of common notes in descending order and trying the best three
		if len(songs_to_consider) == 0:
			print("Couldn't reach threshold. Weak suggestions:")
			sorted_x = sorted(common_targets.items(), key=operator.itemgetter(1), reverse=True)
			songs_to_consider = [x[0] for x in sorted_x[:3]]
		
		#Check for time coherency at this stage. Find a delta per song which has maximizes number of instances of:
		#delta = time of anchor in song - time of anchor in input clip
		final_songs_data = self.filter_database_song(songs_to_consider)
		coherent_notes = {}
		# print(songs_to_consider)
		
		for song_name, sdict in final_songs_data.items():
			deltas = {}
			for freq_pair, anchor_times in cur_song_hash.items():
				for anchor_time in anchor_times:
					if freq_pair not in sdict.keys():
						continue
					target_times = sdict[freq_pair]
					for target_time in target_times:
						diff = round(target_time-anchor_time, self.round_off)
						if diff not in deltas:
							deltas[diff] = 0
						deltas[diff] += 1
			#Keep the delta which iccurs the maximum number of times
			best_delta = max(deltas, key=deltas.get)
			coherent_notes[song_name] = (best_delta, deltas[best_delta])

		#Check for the song which has the maximum number of time coherent notes.
		most_notes, best_song = 0, '.'
		for song_name, deltas in coherent_notes.items():
			if deltas[1] > most_notes:
				most_notes = deltas[1]
				best_song = song_name

		print("\n***Best song:", best_song, "with delta in seconds = ", coherent_notes[best_song][0],'***\n')

		return (best_song, coherent_notes[best_song][0])


if __name__ == "__main__":

	#make Findit object
	app = Findit(audio_path='audio/', data_path='data/')
	#create song object from input path
	song = app.create_song('test/A_Sky_Full_of_Stars_12_3.wav', try_dumped=True, is_target=True)
	#Find best match
	app.compare_song(song)

	# song.fft_and_mask(plot_spec=True, plot_filtered=True) #Used for visualising the spectogram
	


"""
Brute force method which uses the sliding window approach to compare the input target zone with all songs in the database
Extremely slow!
	def brute_force_compare_filtered(self, song1, song2):
		#Assuming x has more time slices than y (x is original, y is input)
		#Both have structure: x[time_slice] = [freq1, freq2, etc.]
		
		if song1.time_res != song2.time_res:
			print("Time resolution of both songs is unequal. Sliding window yet to be implmeneted")
			# exit(0)
			return (0,0)

		try:
			x = song1.filtered
			print("Found filtered for",song1.name,"in pickle dump")
		except:
			x = song1.fft_and_mask()
			song1.dump()
		try:
			y = song2.filtered
			print("Found filtered for",song2.name,"in pickle dump")
		except:
			y = song2.fft_and_mask()
			song2.dump()

		print("\nBrute-Force comparison between",song1.name, "and",song2.name," \n")
		if len(x) < len(y):
			x,y = y,x

		lx, ly = len(x), len(y)

		overlap = {}
		for start in range(lx-ly+1):
			matches, total = 0, 0
			for i in range(ly):
				if start+i not in x.keys() or i not in y.keys():
					continue
				x_cur, y_cur = x[start+i], y[i]
				total += len(x_cur) + len(y_cur)
				for x_freq in x_cur:
					for y_freq in y_cur:
						if abs(x_freq-y_freq) < self.delta:
							matches += 2
			if total == 0:
				overlap[start] = -1
			else:	
				overlap[start] = matches/total

		# plt.scatter(np.array(list(overlap.keys()))*song1.time_res, overlap.values())
		idx = np.argmax(list(overlap.values()))
		print("Most probable overlap between",song1.name,"and",song2.name,"is at time t = ",idx*song1.time_res,'with strength = ',overlap[idx])
		# plt.show()
		return (idx*song1.time_res, overlap[idx])


	def brute_search(self, target_song):

		base_pth = self.audio_path

		comparisons = {}

		target = Song(base_pth+target_song, self.window_size, keep_coeff = self.keep_coeff, sampling_freq=self.sampling_freq)
		maximum = -1
		name = '.'
		best_time = 0

		for song_name in os.listdir(base_pth):
			
			if song_name != target_song and song_name != '.DS_Store':

				current = Song(base_pth+song_name, self.window_size, keep_coeff = self.keep_coeff, sampling_freq=self.sampling_freq)
				time, strength = self.brute_force_compare_filtered(current, target)
				comparisons[song_name] = (time, strength)
				if strength > maximum:
					maximum = strength
					name = song_name
					best_time = time

		print("BEST MATCH:",name,"at time t = ",best_time)
		print('\n',comparisons)
"""