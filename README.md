# Introduction

Findit is a Python program which can detect the song being played by comparing it with a local database of songs. </br>
[How Shazam Works](http://coding-geek.com/how-shazam-works/) is a brilliant article which discusses the approach from scratch. The code is a direct implementation of the above article (barring a few conceptual changes such as an overlapping window). </br>
The user can go through each and every step of the pipeline, visualise the intermediate results and get a feel for the complete approach which is used as the basic pipeline by major commercial applications such as Shazam!

Here is a spectogram of 3-second audio clip from 'A Sky Full of Stars'
![](img/spectogram.png?raw=true "Spectogram")
And here's a filtered version which only keeps the strongest frequencies
![](img/filtered.png?raw=true "Discretized Spectogram")
On running it on a test of 200 clips, here are the results:
![](img/200_test.png?raw=true "Results")

# Requirements

The program requires:
1. Python 3.x
2. [soX](http://sox.sourceforge.net) is a powerful open-source audio processing application. It is used for resampling the audio and trimming random clips for testing. Download the executable, rename it to 'sox' and place it in the parent folder.
3. PyDub, SciPy, NumPy and Matplotlib

# Usage

1. Place your audio files, which form the database, in the audio/ directory.
2. In the 'Findit.py' script, enter the target audio path and run the program
3. test_maker.py can generate random clips from the audio database and automatically run the clips through Findit and generate the results
