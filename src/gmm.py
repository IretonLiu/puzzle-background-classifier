from ftplib import error_reply
import os
import numpy as np
from scipy.stats import multivariate_normal, norm


class GaussianMixtureModel():
    """
    Gaussian Mixture Model

    EM using maximum likelihood estimation: log(Pr(x,h|theta))
    EM using MAP: log(Pr(x,h|theta)*Pr(theta)) ?


    """

    def __init__(self, n_components, dim, epsilon=0.000001, max_iter=100, seed=0) -> None:
        """
        Parameters:
        h: the hidden variable
        dim: the dimension of the data
        max_iter: number of iterations to run
        """

        self.n_components = n_components
        self.dim = dim
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.seed = seed

        self.reset_params()

    def reset_params(self):
        """initiaize the parameters"""
        # lambda
        np.random.seed(self.seed)
        self.l = np.random.rand(self.n_components)
        self.l = self.l / np.sum(self.l)  # normalize
        # mu, the mean of the gaussian of the data, 3 is the dimension of the data
        self.means_ = np.random.rand(self.n_components, self.dim)
        # Sigma, the covariance matrix of the gaussian of the data
        self.covariances_ = np.random.rand(
            self.n_components, self.dim, self.dim)
        # make sure the matrix is symmetric
        self.covariances_ = 2*np.array(
            [(s + s.T)/2 for s in self.covariances_]) + self.dim*np.eye(self.dim)*10

    def fit(self, data):
        print("Fitting the model...")
        """
        Parameters:
        data: the data to be fitted
        """
        self.reset_params()

        error = np.inf
        iter = 0
        while error > self.epsilon and iter < self.max_iter:
            # keep a copy of the old parameters
            old_l = np.copy(self.l)
            old_mean = np.copy(self.means_)
            old_covar = np.copy(self.covariances_)

            # caculate the responsibility
            r = self.reponsibility(data)

            # update lambda
            self.l = np.sum(r, axis=1)/np.sum(r)

            # update the mean
            self.means_ = np.array([np.dot(r[i], data)/np.sum(r[i])
                                    for i in range(self.n_components)])

            # update the covariance
            for i in range(self.n_components):
                self.covariances_[i] = np.dot(r[i]*(data-self.means_[i]).T,
                                              data-self.means_[i])/np.sum(r[i])
                self.covariances_[
                    i] += np.eye(len(self.covariances_[i])) * 1e-9
            # self.covariances_ = np.array([np.sum(r[i].T[:, None, None]*((data-self.means_[i])[..., None]*(data-self.means_[i])
            #                                      [:, None, :]), axis=0)/np.sum(r[i]) for i in range(self.n_components)])
            # calculate the error
            print("Iteration: ", iter)
            error = np.average(np.array([np.max(np.abs(old_l-self.l)), np.max(
                np.abs(old_mean-self.means_)), np.max(np.abs(old_covar-self.covariances_))]))
            print("error:", error)
            iter += 1

    def reponsibility(self, data):
        """Calculate the responsibilites of each data point to each gaussian"""
        # one gaussian for each k
        norms = [multivariate_normal(
            mean=self.means_[i], cov=self.covariances_[i]) for i in range(self.n_components)]

        # calculate the responsibility
        # the numerator
        r = np.array([self.l[i]*norms[i].pdf(data)
                      for i in range(self.n_components)])
        # the denominator
        r /= np.sum(r, axis=0, initial=1e-10)

        return r

    def save_model(self, dir):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
            np.save(dir+"lambda.npy", self.l)
            np.save(dir+"means.npy", self.means_)
            np.save(dir+"covariances.npy", self.covariances_)
            print("Model saved to: ", dir)
        except FileNotFoundError:
            print("Error saving model")

    def load_model(self, dir):
        try:
            self.l = np.load(dir+"lambda.npy")
            self.means_ = np.load(dir+"means.npy")
            self.covariances_ = np.load(dir+"covariances.npy")
        except FileNotFoundError:
            print("No such file or directory")

    def pdf(self, data):
        return np.sum([self.l[i]*multivariate_normal(mean=self.means_[i], cov=self.covariances_[i]).pdf(data) for i in range(self.n_components)], axis=0)
