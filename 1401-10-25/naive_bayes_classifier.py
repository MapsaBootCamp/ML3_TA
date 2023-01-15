import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.stds = None
    
    def fit(self, X, y):
        # Calculate the prior probability of each class
        self.classes, priors = np.unique(y, return_counts=True, )
        priors = priors / len(y)
        self.priors = priors[..., np.newaxis]

        # Calculate the mean and standard deviation of each feature for each class
        stds = []
        s_ap = stds.append
        means = []
        m_ap = means.append
        for c in self.classes:
            X_c = X[y == c]
            m_ap(np.mean(X_c, axis=0))
            s_ap(np.std(X_c, axis=0))
        self.means = np.array(means)[:, np.newaxis,:]
        self.stds = np.array(stds)[:, np.newaxis,:]
        return self
    
    def _likelihood(self, X):
        likelihood = np.prod(
            (1 / np.sqrt(2 * np.pi * self.stds ** 2))
            * np.exp(-(X - self.means) ** 2
                     / (2 * self.stds ** 2)), axis=-1)
        return likelihood

    def _posterior(self, X):
        likelihoods = self._likelihood(X)
        posteriors = likelihoods * self.priors
        posteriors /= np.sum(posteriors, axis=0)
        return posteriors
    
    def predict(self, X):
        posteriors = self._posterior(X)
        return np.argmax(posteriors, axis=0)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)