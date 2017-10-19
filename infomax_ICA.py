#coding: utf-8
#Description:
#Removing occular artefacts using compound based infomax algorithm 
#that uses Independent Component Analysis Algorithm
#From: Andac Demir
#Date: Sep 15 2017

#STEP 1:
#Generate the data that will be tested later:

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

np.random.seed(0)  # set seed for reproducible results
n_samples = 4000
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: sawtooth signal
S = np.c_[s1, s2, s3]

#S has a shape 4000 rows and 3 colums
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # Standardize data

# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

#Step 2:
#Recovering of the uncontaminated signals from the multivariate signal
# compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Get the estimated sources
A_ = ica.mixing_  # Get estimated mixing matrix

plt.figure(figsize=(12, 8))
models = [X, S, S_]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA estimated sources',
         ]
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.tight_layout()
plt.show()
