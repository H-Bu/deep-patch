"""use MLP to construct the arbiter classifier. (0-noise 1-clear)"""
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib


def C_train(noise_name, net_name):
    x = np.load(noise_name+net_name+'feature_for_C.npy')
    y = np.load(noise_name+net_name+'labels_for_C.npy')

    model = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=[32], random_state=1)
    model.fit(x, y)

    joblib.dump(model, 'C.model')
