from cmath import *

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile
import pickle
import os
import subprocess

"""
The song class contains various parameters such as sampling frequency, the data, the filtered spectogram, etc.

window_size: window used during calculation of the DFT
keep_coeff: a coefficient which is multiplied by the mean of the filtered spectogram to keep only the strong frequencies
sampling_freq: sampling frequency of the audio file. Converted to 22.1KHz since most of the nergy is below 10.05 KHz
overlap_factor: the factor by which the windows overlap while calcualting the DFT. 
				A value of 0 indciates that the windows slide without any overlap between them.

"""

class Song():

	def __init__(self, filename, window_size, keep_coeff, try_dumped, sampling_freq, overlap, is_target):

		self.name = filename.split('/')[-1].split('.')[0]
		print("**** Working on the audio file:",self.name,'****')

		#parametrs
		self.keep_coeff = keep_coeff
		self.window_size = window_size
		self.sampling_freq = sampling_freq
		self.overlap_factor = overlap
		self.time_res = self.window_size*(1-self.overlap_factor)/self.sampling_freq

		#load filtered spectogram from pickle dump if it exists
		if os.path.exists('data/data_'+self.name) and try_dumped:

			print("Loading data from pickle dump...")
			with open('data/data_'+self.name, 'rb') as f:
				data = pickle.load(f)

			self.filtered = data['filtered']
			
			try:
				self.data = data['data']	
			except:
				pass
				
		else:
			#IO and normalise
			self.io(filename, sampling_freq, is_target)
		
	#read the .wav file using Scipy, convert to the desired sampling frequency using soX and also convert stereo to mono
	def io(self, filename, sampling_freq, is_target):

		cur_sampling_freq, self.data = wavfile.read(filename)
		
		#resample
		if cur_sampling_freq != self.sampling_freq:
			print("Resampling from", cur_sampling_freq, 'to', self.sampling_freq)
			newname = filename.split('.')[-2]+'_resampled.wav'
			subprocess.call(['./sox', filename, '-r', str(self.sampling_freq), newname])
			cur_sampling_freq, self.data = wavfile.read(newname)

			#delete the resampled file since the target audio is not to be stored for further use
			if is_target:
				os.remove(newname)
			else:
				os.rename(newname, filename)

		#normalise
		self.data = self.data/16284

		#Convert stereo to mono by averaging the channels
		if isinstance(self.data[0], (list, tuple, np.ndarray)):
			self.data = np.mean(self.data, axis=1)
			print("Converted stereo to mono by averaging")

		print("Max freq by Nyquist = ", self.sampling_freq//2)

	def fft_and_mask(self, plot_spec, plot_filtered):

		#Calculate fft and plot spectogram
		# x,y,z have shape: freq_bins x windows
		x, y, self.fft = self.fft_and_spectogram(plot=plot_spec, window_func='hamming')

		#filter the spectogram
		self.filtered = self.mask_spectogram(x, y, self.fft, plot=plot_filtered)

		return self.filtered

	"""
	Crucial stage of keeping only the strongest frequencies is done in this function
	The psycho-acoustic logarithmic model is kept in mind while ignoring the weaker frequencies.
	For each time slice, we first calculate the strongest points in the logarithmic band.
	Then, we drop the ones which are wekaer than the (mean of the points along it's time slice)*keep_coeff
	Finally, we also calculate the mean of the strongest points across time in a band and drop those points whose magnitude is below this mean
	"""
	def mask_spectogram(self, x, y, fft, plot):

		print("Filtering...")

		freq_bins, time_slices = x.shape

		#define the logarithmic bands
		if freq_bins == 256:
			logarithmic_bands = [(0,32), (32,64), (64,128), (128,256)]
			print("Freq. bins for 256 bins are:", logarithmic_bands)
		elif freq_bins == 512:
			logarithmic_bands = [(0,20), (20,60), (60,120), (120,240), (240,512)]
			print("Freq. bins for 512 bins are:", logarithmic_bands)
		elif freq_bins == 1024:
			logarithmic_bands = [(0,64), (64,256), (256,512), (512,1024)]
			print("Freq. bins for 1024 bins are:", logarithmic_bands)
		else:
			print("Unknown # freq_bins")
			exit(0)
		
		nbands = len(logarithmic_bands)
		#stores the final frequency points for each time slice in a dictionary with key=time_slice and value=list of strong frequencies
		filtered_spectogram = {}
		band_across_time_mean = np.zeros((time_slices, nbands))

		#iterate over each time slice
		for time_slice in range(time_slices):
			
			temp_spectro = []
			#store temporary spectograms which need to be filtered further
			#contains list of (time_of_occurence, frequency_bin, logarithmic_band_index)
			slice_mean = 0
			
			for i in range(nbands):
				band = logarithmic_bands[i]
				fft_slice = fft[band[0]:band[1], time_slice]
				#find strongest frequency point in corresponding band
				idx = np.argmax(fft_slice)
				band_across_time_mean[time_slice][i] = fft_slice[idx]
				
				slice_mean += fft_slice[idx]
				temp_spectro.append((time_slice, band[0]+idx, fft_slice[idx], i))

			slice_mean /= nbands
			for poss in temp_spectro:
				
				if not poss[0] in filtered_spectogram.keys():
					filtered_spectogram[poss[0]] = []		
				
				if poss[2] >= slice_mean*self.keep_coeff:
					filtered_spectogram[poss[0]].append((poss[1], poss[2], poss[3]))

		#Drop points which are weaker than the mean across time in it's band
		band_across_time_mean = np.mean(band_across_time_mean, axis=0)
		
		for time_slice, data in filtered_spectogram.items():
			final_f = []
			for band in data:
				f_bin, mag, band_no = band[0], band[1], band[2]
				if mag >= band_across_time_mean[band_no]:
					final_f.append(f_bin)
					
			filtered_spectogram[time_slice] = final_f

		#scatter plot of the filtered spectogram
		if plot:
			to_plot_x, to_plot_y = [], []
			for key, val in filtered_spectogram.items():
				plt.scatter(np.array([key]*len(val))*self.time_res, val, marker='x', c='r')

			for log_band in logarithmic_bands:
				plt.axhline(y=log_band[0], color='k', linestyle='-')
			plt.ylim(0, freq_bins)
			plt.show()

		return filtered_spectogram

#Calcualte the FFT and plot the spectogram
	def fft_and_spectogram(self, plot, window_func):

		if window_func == 'hamming':
			print("Using Hamming window")

		#num_windows = total parts whose DFT needs to be calcualted
		num_windows = len(self.data)//(self.window_size*(1-self.overlap_factor))
		freq_resolution = self.sampling_freq/self.window_size
		freq_bins = self.window_size//2
		

		#Create grid for spectogram
		x, y = np.meshgrid(np.arange(0, len(self.data)/self.sampling_freq, self.time_res), np.linspace(0, freq_resolution*freq_bins, num=self.window_size//2))
		
		fft_values = np.zeros(x.shape)

		print("Calculating dft with following parameters:")
		print("freq. resolution =", freq_resolution)
		print('total dft windows =', num_windows)
		print('total frequency bins =',freq_bins)
		
		start, count = 0, 0

		#slide the window considering the overlap factor and calculate DFT of parts
		while count < num_windows:
			
			cur_slice = self.data[start: start+self.window_size]		
			cur_fft = fft(cur_slice, window_func, self.window_size)
			fft_values[:, count] = cur_fft
			start += int((1-self.overlap_factor)*self.window_size)
			count += 1

		#plot the spectogram if set to True
		if plot:

			fig, ax = plt.subplots()
			c = ax.pcolormesh(x, y/1000, fft_values, cmap='BrBG')

			ax.set_title('FT')
			ax.set_xlabel("Time in seconds")
			ax.set_ylabel("Frequency in KHz")

			fig.colorbar(c, ax=ax)

			plt.show()

		return x, y, fft_values

	"""
	dump the filtered spectogram of the complete audio file
	set dump_data=True if the actual contents also need to be dumped
	"""
	def dump(self, base_pth, dump_data):

		#No need to save the data since we can always read the wav file
		try:
			to_dump = {'name':self.name, 'filtered':self.filtered}
		except:
			self.fft_and_mask(plot_spec=False, plot_filtered=False)
			to_dump = {'name':self.name, 'filtered':self.filtered}

		if dump_data:
			to_dump['data'] = self.data

		with open(base_pth+'data_'+self.name, 'wb') as f:
			pickle.dump(to_dump, f)

		print("Successfully Dumped", self.name)
	
	#downsampling technique leads to aliasing. Not used
	def average_downsample(self, factor):

		print("Downsampling by a factor of",factor)

		new_length = len(self.data)//factor
		if len(self.data)%factor != 0:
			new_length += 1
		to_return = np.zeros((new_length))

		for i in range(new_length):
			to_return[i] = np.mean(self.data[factor*i:factor*(i+1)])

		self.data = to_return
		self.sampling_freq = self.sampling_freq/factor
		print("New max freq by Nyquist = ", self.sampling_freq/2)

def recursive_fft(x):
		
		N=len(x)
		if N==1: 
			return x
 
		even = recursive_fft([x[k] for k in range(0,N,2)])
		odd = recursive_fft([x[k] for k in range(1,N,2)])
 
		M=N//2
		l=[ even[k] + exp(-2j*pi*k/N)*odd[k] for k in range(M) ]
		r=[ even[k] - exp(-2j*pi*k/N)*odd[k] for k in range(N-M) ]
 
		return l+r

def fft(x, window_func, window_size):

	l = len(x)
	#Add padding in case length of data is not a power of two
	if l != window_size:

		print("Adding padding of",window_size-l)
		temp = np.zeros((window_size))
		temp[:l] = x
		x = temp

	#Use hamming window function if specified
	if window_func == 'hamming':
		hamming = signal.hamming(window_size)
		x = np.multiply(hamming, x)

	#Calculate dft
	dft = recursive_fft(x)
	#Return only half the samples since the remaining are conjugates
	mods = [abs(x) for x in dft[:len(x)//2]]
	return 20*np.log10(mods)