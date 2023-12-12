import os
import numpy as np
import requests
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision
from PIL import Image
from .glove import GloVe
from tqdm import tqdm

AD_DIRECTORY = os.path.dirname(__file__)

def obtain_vector(inputs, glove):
    vector_im = glove.embedding.get(inputs)
    if vector_im is None:
          vector_im = glove.embedding.get(inputs.lower())
    if vector_im is None:
          vector_im = glove.embedding.get(inputs.title())
    if vector_im is None:
          vector_im = glove.embedding.get(inputs.upper())
    return vector_im

def generate_glove():
    print("Generating glove similarity...")
    #download glove file
    os.makedirs("./knowledge", exist_ok=True)
    glove_file = './knowledge/glove.840B.300d.txt'
    if not os.path.exists(glove_file):
        print("Downloading glove files...")
        print("")
        print("Gonna take a while")
        print("")
        url_path = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
        r = requests.get(url_path)
        with open("./knowledge/glove.840B.300d.zip","wb") as f:
            f.write(r.content)
        filename = './knowledge/glove.840B.300d.zip'
        fz = zipfile.ZipFile(filename, 'r')
        for file in fz.namelist():
            fz.extract(file, './knowledge/.')
        if os.path.exists(filename):
              os.remove(filename)

    glove = GloVe('./knowledge/glove.840B.300d.txt')
    filepath = os.path.join(AD_DIRECTORY, "label_name.txt")
    vec_list = []
    vec_list_np = []
    cos_similarity = np.zeros((1000,1000))

    index = 0
    
    #the labels could be a word or a phrase with multi words
    #we also tested on average of every words
    #But we assume the last word should be more important, so in our final version
    #we assign a higher weight to last word in a phrase and average the fornt words

    #w2v for last word of multi-words
    for line in tqdm(open(filepath)):
      a = line.strip('\n')

      b = a.split(',')
      cnt = 0
      vector = torch.zeros(300)
      vec_front = torch.zeros(300)
      vec_b_average = torch.zeros(300)
      cnt_b = 0
      for i in range(len(b)):
            b[i] = b[i].lstrip()
            c = b[i].split(' ')
            if obtain_vector(c[-1], glove) is not None:
              vec_b_average += obtain_vector(c[-1], glove)
              cnt_b += 1
      if cnt_b == 0:
            print('index ', index,' generatint word_vector failure')
            continue
      vec_b_average = vec_b_average / cnt_b

      for i in range(len(b)):
            b[i] = b[i].lstrip()
            c = b[i].split(' ')
            cnt_f = 0
            for j in range(len(c) - 1):
                  if obtain_vector(c[j], glove) is not None:
                        vec_front += obtain_vector(c[j], glove)
                        cnt_f += 1
            if obtain_vector(c[-1], glove) is not None:
                  vec_back =obtain_vector(c[-1], glove)
            else:
                  vec_back = vec_b_average
            if cnt_f == 0:
                  vector += vec_back
            else:
                  vector += (vec_front / cnt_f )* 0.1 + vec_back * 0.9
            cnt += 1

      vector = torch.div(vector,cnt) 
  
      vec_list_np.append(np.array(vector))
      vec_list.append(vector)
      index += 1


    vec_list_np_stacked = np.stack(vec_list_np)
    vec_list_torch = torch.from_numpy(vec_list_np_stacked)

    cos_similarity = F.cosine_similarity(vec_list_torch[None, :], vec_list_torch[:, None], dim=-1)
    cos_similarity = cos_similarity.numpy()

#     np.save('./knowledge/golve_vec_list', np.array(vec_list_np))
#     for i in range(len(vec_list)):
#       for j in range(len(vec_list)):
#         cos_similarity[i,j] = F.cosine_similarity(vec_list[i], vec_list[j],dim=0).type(torch.half)
#         if i != j:
#           cos_similarity[i,j] = cos_similarity[i,j]
#     cos_similarity = np.array(cos_similarity)
    
    
    np.save(os.path.join(AD_DIRECTORY, "imagenet_cos_similarity_glove"), cos_similarity)
    print("Glove cos_similarity finished")
    return cos_similarity