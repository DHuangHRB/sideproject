import pandas as pd
import numpy as np
import random
import math
from operator import itemgetter
import requests
import json
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML
import urllib
import os.path
import time

link_f   = 'ml-latest-small/links.csv'
link_df   = pd.read_csv(link_f, sep = ',')

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

idx_to_mv = {}
for row in link_df.itertuples():
    idx_to_mv[row[1] - 1] = row[2]
 
# Download 1k posters
dir_poster = './posters/'
max_num_poster = 1000

mvs = [0] * len(idx_to_mv)
for i in range(len(mvs)):
    if i in idx_to_mv.keys() and len(str(idx_to_mv[i])) == 6:
        mvs[i] = idx_to_mv[i]
mvs = list(filter(lambda imdb:imdb != 0, mvs))
total_mvs = len(mvs)

URL = [0] * max_num_poster
URL_IMDB = {'url':[],'imdb':[]}

i = 0
for m in mvs:
    if(os.path.exists(dir_poster + str(i) + '.jpg')):
        #print('Skip downloading exists jpg: {0}.jpg'.format(dir_poster + str(i)))
        i += 1
        continue
    URL[i] = get_poster(m, base_url)
    if(URL[i] == base_url):
        #print('Bad imdb id: {0}'.format(m))
        mvs.remove(m)
        continue
    #print('No.{0}: Downloading jpg(imdb {1}) {2}'.format(i, m, URL[i]))
    time.sleep(1)
    urllib.request.urlretrieve(URL[i], dir_poster + str(i) + '.jpg')
    URL_IMDB['url'].append(URL[i])
    URL_IMDB['imdb'].append(m)
    i += 1
    if len(URL_IMDB['imdb']) > max_num_poster:
        break

image = [0] * max_num_poster
x     = [0] * max_num_poster
prediction_file = 'prediction_result.csv'

def saveArraytoFile(array):
    np.savetxt(prediction_file, array, delimiter = ",")

def loadFiletoArray(file):
    if not os.path.exists(file):
        return None
    return np.genfromtxt(file, delimiter = ',')
	
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage

# Extract latent features of posters by using pretrained CNN - VGG16
train_deep = loadFiletoArray(prediction_file)
if train_deep is None:
    for i in range(max_num_poster):
        image[i] = kimage.load_img(dir_poster + str(i) + '.jpg', target_size = (224,224))
        x[i] = kimage.img_to_array(image[i])
        x[i] = np.expand_dims(x[i], axis = 0)
        x[i] = preprocess_input(x[i])

    model = VGG16(include_top = False, weights = 'imagenet')
    prediction = [0] * max_num_poster
    train_deep = np.zeros([max_num_poster, 25088]) # 7 * 7 * 512 features
    for i in range(max_num_poster):
        prediction[i] = model.predict(x[i]).ravel()
        train_deep[i, :] = prediction[i]
    saveArraytoFile(train_deep)

# Calculate movie similarity matrix based on latent features of posters
train_deep = loadFiletoArray(prediction_file)
similarity_deep = train_deep.dot(train_deep.T)
norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
similarity_deep = (similarity_deep/ (norms * norms.T))

# Validation by movie posters downloaded from Movie Database
n_display   = 5
test_mv_idx = 0
mv = [x for x in np.argsort(similarity_deep[test_mv_idx])[:-n_display-1:-1]]

images = ''
for i in range(len(mv)):
    images+="<img style='width: 185px; margin: 0px; float: left; border: 1px solid black;' src={0}.jpg />".format(dir_poster + str(mv[i]))
display(HTML(images))





