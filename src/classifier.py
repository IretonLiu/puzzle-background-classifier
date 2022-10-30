
from gmm import GaussianMixtureModel
import numpy as np


class Classifier:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def maximum_a_posteriori(self, likelihoods):
        """ map for binary classification """

        prior = [1-np.sum(self.labels)/len(self.labels),
                 np.sum(self.labels)/len(self.labels)]

        return np.array([likelihoods[i].score_samples(self.data)*prior[i] for i in range(2)])
