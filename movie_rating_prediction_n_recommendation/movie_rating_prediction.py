import pandas as pd
import numpy as np
import random
import math
from operator import itemgetter

# Movie Data from MovieLens
# ratings.csv: including 671 users and 100k ratings
# links.csg: including IMDBID and TMDBID of 9125 movies

rating_f = 'ml-latest-small/ratings.csv'
link_f   = 'ml-latest-small/links.csv'

rating_df = pd.read_csv(rating_f, sep = ',')
link_df   = pd.read_csv(link_f, sep = ',')
rating_df = pd.merge(rating_df, link_df, on = ['movieId'])

print(rating_df.userId.unique().shape)
print(rating_df.movieId.unique().shape)
print(rating_df.imdbId.unique().shape)
print(rating_df.info())

# Build movie ratings matrix
rating_matrix = np.zeros((rating_df.userId.unique().shape[0],max(rating_df.movieId)))
for row in rating_df.itertuples():
    rating_matrix[row[1] - 1, row[2] - 1] = row[3]
rating_matrix = rating_matrix[:,:9000] #get first 9000 movies

# Build train and test matrix
train_matrix = rating_matrix.copy()
test_matrix  = np.zeros(rating_matrix.shape)

for i in range(rating_matrix.shape[0]):
    rating_index = np.random.choice(rating_matrix[i].nonzero()[0], size = 10, replace = True)
    train_matrix[i, rating_index] = 0.0
    test_matrix[i, rating_index] = rating_matrix[i, rating_index]
print(train_matrix.shape, test_matrix.shape)

# Calculate user similarity matrix
scalar = np.array([np.log(2 + train_matrix.T.sum(axis = 1))]).T
similarity_user = train_matrix.dot(train_matrix.T / scalar) + 1e-9
norms = np.array([np.sqrt(np.diagonal(train_matrix.dot(train_matrix.T)))]) + 1e-9
similarity_user = (similarity_user / (norms * norms.T))


# Calculate movie similarity matrix
scalar = np.array([np.log(2 + train_matrix.sum(axis = 1))]).T
similarity_mv = train_matrix.T.dot(train_matrix / scalar) + 1e-9
norms = np.array([np.sqrt(np.diagonal(train_matrix.T.dot(train_matrix)))]) + 1e-9 
similarity_mv = (similarity_mv / (norms * norms.T))

# User based ratings prediction
from sklearn.metrics import mean_squared_error
prediction = similarity_user.dot(train_matrix) / np.array([np.abs(similarity_user).sum(axis = 1)]).T
prediction = prediction[test_matrix.nonzero()]
test_vector = test_matrix[test_matrix.nonzero()]
rmse = math.sqrt(mean_squared_error(prediction, test_vector))

print('rmse: {0}'.format(rmse))

# Item based ratings prediction
prediction = similarity_mv.dot(train_matrix.T) / np.array([np.abs(similarity_mv).sum(axis = 1)]).T
prediction = prediction.T[test_matrix.nonzero()]
test_vector = test_matrix[test_matrix.nonzero()]
rmse = math.sqrt(mean_squared_error(prediction, test_vector))

print('rmse: {0}'.format(rmse))


# Validation by movie posters downloaded from Movie Database
import requests
import json
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML

#Get posters from Movie Database by API
headers = {'Accept': 'application/json'}
params = {'api_key': ''}
response = requests.get('http://api.themoviedb.org/3/configuration',params = params, headers = headers)
response = json.loads(response.text)
base_url = response['images']['base_url'] + 'w185'

def get_poster(imdb, base_url):
    #query themovie.org API for movie poster path.
    imdb_id = 'tt0{0}'.format(imdb)
    movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(imdb_id)
    response = requests.get(movie_url, params = params, headers = headers)
    try:
        file_path = json.loads(response.text)['posters'][0]['file_path']
    except:
        file_path = ''
    return base_url + file_path


# n_display : Number of most similar movies recommended
# test_mv_idx: index of test movie 
n_display = 5
test_mv_idx = 0

idx_to_mv = {}
for row in link_df.itertuples():
    idx_to_mv[row[1] - 1] = row[2]
    
mv = [idx_to_mv[x] for x in np.argsort(similarity_mv[test_mv_idx])[:-(n_display * 2)-1:-1]]
mv = filter(lambda imdb: len(str(imdb)) == 6, mv)
mv = list(mv)[:n_display]

URL = [0] * len(mv)
for i, m in enumerate(mv):
    URL[i] = get_poster(m, base_url)

images = ''
for i in range(n_display):
    images+="<img style='width: 185px; margin: 0px; float: left; border: 1px solid black;' src='%s'/>" % URL[i]
display(HTML(images))





