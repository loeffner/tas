### Author: Andreas Loesel, ga63miy
##
### This file is used for testing and evaluating various machine learning techniques.
### Currently implemented are:
###     - K-Nearest Neighbors Classifier (KNN)
###     - PCA + KNN
###     - Multi-Layer Perceptron Classifier (MLP)

import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bssid_file = 'bssids.csv'

car_files = [('auto_gang_hinten', 'gang_hinten'), ('auto_gang_lab', 'gang_lab'), ('auto_gang_notaus', 'gang_notaus'), ('auto_kreuz', 'kreuz'), ('auto_lab_hinten', 'lab_hinten'), ('auto_lab_vorne', 'lab_vorne'), ('auto_treppen', 'treppen')]
car_files_2 = [('auto_gang_hinten_2', 'gang_hinten'), ('auto_gang_lab_2', 'gang_lab'), ('auto_gang_notaus_2', 'gang_notaus'), ('auto_kreuz_2', 'kreuz'), ('auto_lab_hinten_2', 'lab_hinten'), ('auto_lab_vorne_2', 'lab_vorne'), ('auto_treppen_2', 'treppen')]
car_files_comb = [(r'data\auto_gang_hinten_comb', 'gang_hinten'), (r'data\auto_gang_lab_comb', 'gang_lab'), (r'data\auto_gang_notaus_comb', 'gang_notaus'), (r'data\auto_kreuz_comb', 'kreuz'), (r'data\auto_lab_hinten_comb', 'lab_hinten'), (r'data\auto_lab_vorne_comb', 'lab_vorne'), (r'data\auto_treppen_comb', 'treppen')]
laptop_files = [('lab_vorne.csv', 'lab_vorne'), ('lab_hinten.csv', 'lab_hinten'), ('gang_lab.csv', 'gang_lab'), ('gang_notaus.csv', 'gang_notaus'), ('gang_hinten.csv', 'gang_hinten'), ('treppen.csv', 'treppen')]

# Load data from a data file to a numpy array
def load(data_file, label):
    return np.loadtxt(data_file, dtype=float), label
    
# Load BSSIDS
def load_bssids(bssid_file):
    return np.loadtxt(bssid_file, dtype=str)

# Bar Plot 
def bar_plot(data, x_tick_labels, title='', ax=plt, shift=0, width=0.4, color=None, label=None):
    x = np.arange(len(data))
    ax.bar(x + width * shift, data, width=width, color=color, label=label)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    # ax.set_ylim(0, 70)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

# Process the data s.t. a larger number corresponds to a better signal
def positive(data):
    data[np.where(data != 0)] = 100 + data[np.where(data != 0)]

# Split data int training set and testing set
def train_test(data, train_test_ratio=10, seed=0):
    train_spls = data[0][0]
    train_lbls = np.array([data[0][1]] * data[0][0].shape[0])
    for i in range(1, len(data)):
        train_spls = np.append(train_spls, data[i][0], axis=0)
        train_lbls = np.append(train_lbls, [data[i][1]] * data[i][0].shape[0])
    test_spls = train_spls[0].reshape(1, -1)
    np.delete(train_spls, 0)
    test_lbls = (train_lbls[0])
    np.delete(train_lbls, 0)
    random.seed(seed)
    for i in range(1, int(train_spls.shape[0] * train_test_ratio / 100)):
        r = random.randint(0, train_spls.shape[0]-1)
        test_spls = np.append(test_spls, train_spls[r].reshape(1, -1), axis=0)
        np.delete(train_spls, r)
        test_lbls = np.append(test_lbls, train_lbls[r])
        np.delete(train_lbls, r)
    return (train_spls, train_lbls), (test_spls, test_lbls)

# Get the occurrences of labels in a data set (to see the data distribution)
def occurence(labels, rel=True):
    unique, counts = np.unique(labels, return_counts=True)
    if rel:
        return (dict(zip(unique, counts/labels.shape[0])))
    else:
        return (dict(zip(unique, counts)))


if __name__=='__main__':
    # Load data files
    print('Loading data')
    data = []
    for sample_set in car_files_comb:
        data.append(load(sample_set[0], sample_set[1]))
    [positive(d[0]) for d in data]

    # Load bssid list
    bssids = load_bssids(bssid_file)
    bssids_short = [bssid[-3:] for bssid in bssids]

    # Visualize data
    fig, ax = plt.subplots(3, 3, sharey='row', constrained_layout=True)
    for i in range(7):
        bar_plot(np.mean(data[i][0], axis=0), width=0.8, x_tick_labels=bssids_short, title=data[i][1], ax=ax[i//3, i-i//3*3])
    for i in range(7, 9):
        ax[i//3, i-i//3*3].remove()
    plt.show()

    # Split data into train & test
    (train_spls, train_lbls), (test_spls, test_lbls) = train_test(data, train_test_ratio=10)
    print('Training Set Shape:', train_spls.shape)
    print('Testing Set Shape:', test_spls.shape)
    fig, ax = plt.subplots(1, 1)
    bar_plot(list(occurence(test_lbls).values()), shift=-0.5, x_tick_labels=list(occurence(test_lbls).keys()), color='r', label='Test Samples', ax=ax)
    bar_plot(list(occurence(train_lbls).values()), shift=0.5, x_tick_labels=list(occurence(train_lbls).keys()), color='b', label='Train Samples', ax=ax)
    ax.legend()
    plt.show()

    # Scale the data
    print('Scaling the data')
    scaler = StandardScaler()
    train_spls = scaler.fit_transform(train_spls)

    # KNN Classifier
    print('Creating KNN Classifier')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_spls, train_lbls)

    # PCA
    print('Applying PCA')
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(train_spls)

    # PCA + KNN Classifier
    pca_knn = KNeighborsClassifier(n_neighbors=3)
    pca_knn.fit(pca_data, train_lbls)

    # MLP Classifier
    layers = (156, 78)
    print('Training MLP Classifier with the following layer - design', layers)
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=200, activation='tanh', solver='lbfgs', alpha=0.1, verbose=False)
    mlp.fit(train_spls, train_lbls)

    err_knn= 0
    err_l_knn = []
    err_pca_knn = 0
    err_l_pca_knn = []
    err_mlp = 0
    err_l_mlp = []

    print('Evaluating')
    for i, sample in enumerate(test_spls):
        pred_knn = knn.predict(scaler.transform(sample.reshape(1, -1)))
        pred_pca_knn = pca_knn.predict(pca.transform(scaler.transform(sample.reshape(1, -1))))
        pred_mlp = mlp.predict(scaler.transform(sample.reshape(1, -1)))
        if test_lbls[i] != pred_knn:
            err_knn += 1
            err_l_knn.append(test_lbls[i])
        if test_lbls[i] != pred_pca_knn:
            err_pca_knn += 1
            err_l_pca_knn.append(test_lbls[i])
        if test_lbls[i] != pred_mlp:
            err_mlp += 1
            err_l_mlp.append(test_lbls[i])
    
    print('KNN:\n\tGuessed', test_lbls.shape[0]-err_knn, 'out of', test_lbls.shape[0], 'correct. Ratio:', err_knn/i)
    print('\tGuessed wrongly:', occurence(np.array(err_l_knn), rel=False))
    print('PCA + KNN:\n\tGuessed', test_lbls.shape[0]-err_pca_knn, 'out of', test_lbls.shape[0], 'correct. Ratio:', err_pca_knn/i)
    print('\tGuessed wrongly:', occurence(np.array(err_l_pca_knn), rel=False))
    print('KNN:\n\tGuessed', test_lbls.shape[0]-err_mlp, 'out of', test_lbls.shape[0], 'correct. Ratio:', err_mlp/i)
    print('\tGuessed wrongly:', occurence(np.array(err_l_mlp), rel=False))
