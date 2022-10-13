import numpy as np
from scipy.stats import multivariate_normal, norm


class GaussianMixtureModel():
    """
    Gaussian Mixture Model

    EM using maximum likelihood estimation: log(Pr(x,h|theta))
    EM using MAP: log(Pr(x,h|theta)*Pr(theta)) ?


    """

    def __init__(self, n_components, dim, max_iter=100, seed=0) -> None:
        """
        Parameters:
        h: the hidden variable
        dim: the dimension of the data
        max_iter: number of iterations to run
        """

        self.n_components = n_components
        self.dim = dim
        self.max_iter = max_iter
        self.seed = seed

        self.reset_params()

    def reset_params(self):
        # initiaize the parameters
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
        self.covariances_ = np.array(
            [(s + s.T)/2 for s in self.covariances_]) + self.dim*np.eye(self.dim)*10

    def fit(self, data):
        print("Fitting the model...")
        """
        Parameters:
        data: the data to be fitted
        """
        self.reset_params()

        for i in range(self.max_iter):
            old_l = np.copy(self.l)
            old_mean = np.copy(self.means_)
            old = np.copy(self.covariances_)

            # print("Iteration: ", i)
            # print("Before")
            # print("lambda:", self.l)
            # print("mean:", self.means_)
            # print("covariance:", self.covariances_)
            r = self.reponsibility(data)
            # r[0] = np.array([0.3, 0.2, 0.25, 0.6, 0.7])
            # r[1] = 1 - r[0]

            # update lambda
            self.l = np.sum(r, axis=1)/np.sum(r)
            # print(old-self.l)
            # update the mean
            self.means_ = np.array([np.dot(r[i], data)/np.sum(r[i])
                                    for i in range(self.n_components)])

            # update the covariance
            self.covariances_ = np.array([np.sum(r[i].T[:, None, None]*((data-self.means_[i])[..., None]*(data-self.means_[i])
                                                 [:, None, :]), axis=0)/np.sum(r[i]) for i in range(self.n_components)])
            # self.covariances_ = np.array(
            #     [np.cov(data.T, aweights=(r[i]), ddof=0) for i in range(self.n_components)])
            # self.covariances_ = np.array([np.sum(r[i].T[:, None, None]*(
            #     data-self.means_[i])**2, axis=0)/np.sum(r[i]) for i in range(self.n_components)])

            # print("After")
            # print("responsibility: ", r)
            # print("lambda:", self.l)
            # print("mean:", self.means_)
            # print("covariance:", self.covariances_)
            print("Iteration: ", i)
            print("l diff:", np.sum(np.abs(old_l-self.l)))
            print("mean diff:", np.sum(np.abs(self.means_-old_mean)))
            print("covariance diff:", np.sum(np.abs(self.covariances_-old)))

    def reponsibility(self, data):

        # one gaussian for each k
        norms = [multivariate_normal(
            mean=self.means_[i], cov=self.covariances_[i]) for i in range(self.n_components)]

        # calculate the responsibility
        # the numerator
        r = np.array([self.l[i]*norms[i].pdf(data)
                     for i in range(self.n_components)])
        # the denominator
        r /= np.sum(r, axis=0)

        return r
