# PythonExam -> MusicSolution 2000

A. We want to make a music recommendation app that will use machine learning algorithms to recommend different bands/albums/genres etc.
   based on the users listening preferences. 

B. We will use either the Spotify web API or Shazam API, which are free to use. 

C. Collect data from music databases. We will try to use the Panda and Mumpy libraries for the cleaning of our data, like removing duplicates, removing errors and missing items. 

D. We will visualize the data to see which genres are most popular among different user groups. (Maybe using Streamlit for frontend).
   We want to give the user access to some statistics and information about the different music they are recommended -ish.
   We will use matplotlib for the visualization of our data and statistics. 

E. We will use collaborative filtering algorithms to recommend music to users based on the bands/genres/artist/albums/tracks the user chooses as their favorite.

# Short description

We wanted to make a music recommendation app in python, using various libraries and technologies we have learned throughout this semester. The app we made should make recommendations based on either the user input or the users preferences on genres. As a user you can select recommendations based on other users, or you can type in your favourite artists and get recommendations. If that's not good enough you can get recommendations by your favourite genre. 

# List of used technologies

Streamlit for visualization
pandas for reading csv files and data modeling
sklearn for algorithm nearestNeighbour and data cleaning
plotly for genre plots
implicit for algoritm ALS
matplotlib for data plots
scipy for matrix

# Installation guide & user guide

Clone the project -> cd into musicSolution folder. In the folder install the nedded dependencies with pip install ->  After installation is completed, go to console and type in streamlit run app.py. Make sure you are in musicSolution folder! 

# Status & list of challenges

We have implemented what we wanted to do in the first place. We wanted to make a recommender app based on collaborative filtering, with the use of algorithms which we believe we did in our application. We did mention that we wanted to use a music API to fetch the data, but instead we went with different music datasets, that we combined instead, and made some cleaning on using pandas. We thought this was better suited to us, and our needs. One of the challenges we faced was using the correct algorithm, and also to train a model to do exactly what we wanted it to do. We didnt know exactly how we should handle it, therefore we imported the necessary algorithms from implicit and sklearn instead of building our own from scrath. 


