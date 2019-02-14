# -*- coding: utf-8 -*-

import json
import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

problem2id = json.load(open('problems.jsn','r',encoding='utf8'))
vectors    = np.load('problemVecs.npy')

positions  = Isomap(n_neighbors=5,n_components=2).fit_transform(vectors)


plt.figure(figsize = (10,10))
for name,i in problem2id.items():
    if name in ['pad','unk']:
        continue
    plt.plot(positions[i,0],positions[i,1])
    plt.text(positions[i,0]+.1 , positions[i,1]+.1 , name)