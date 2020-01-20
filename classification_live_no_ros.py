### Author: Andreas Loesel, ga63miy
##
### This program collects a wifi sample and performs a classification to locate the car
### The chosen algorithm is PCA + KNN, since it delivers the best results/effort ratio
### The performance of different methods i shown in classification.py


import subprocess
import random
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Never hardcode your password!
sudo_pass = 'YourPassword'
interface = 'YourInterface'
# iwlist cmd
cmd = 'sudo iwlist ' + interface + ' scan'

bssid_file = 'bssids.csv'

car_files_comb = [('data/auto_gang_hinten_comb', 'gang_hinten'), ('data/auto_gang_lab_comb', 'gang_lab'), ('data/auto_gang_notaus_comb', 'gang_notaus'), ('data/auto_kreuz_comb', 'kreuz'), ('data/auto_lab_hinten_comb', 'lab_hinten'), ('data/auto_lab_vorne_comb', 'lab_vorne'), ('data/auto_treppen_comb', 'treppen')]

# Load data from a data file to a numpy array
def load(data_file, label):
    return np.loadtxt(data_file, dtype=float), label
    
# Load BSSIDS
def load_bssids(bssid_file):
    return np.loadtxt(bssid_file, dtype=str)

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
    data = []
    for sample_set in car_files_comb:
        data.append(load(sample_set[0], sample_set[1]))
    [positive(d[0]) for d in data]

    # Load bssid list
    bssids = load_bssids(bssid_file)
    bssids_short = [bssid[-3:] for bssid in bssids]

    # Split data into train & test
    (train_spls, train_lbls), (test_spls, test_lbls) = train_test(data, train_test_ratio=0)

    # Scale the data
    scaler = StandardScaler()
    train_spls = scaler.fit_transform(train_spls)

    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(train_spls)

    # PCA + KNN Classifier
    pca_knn = KNeighborsClassifier(n_neighbors=3)
    pca_knn.fit(pca_data, train_lbls)

    input('Press enter to start wifi sample collection')
    for i in range(3):
        time.sleep(1)
        cmd1 = subprocess.Popen(['echo',sudo_pass], stdout=subprocess.PIPE)
        cmd2 = subprocess.Popen(['sudo','-S'] + cmd.split(), stdin=cmd1.stdout, stdout=subprocess.PIPE)
        wifi = cmd2.stdout.read().decode().lower()

    # Retreive MAC and signal strength
    wifi = wifi.split('cell')
    dct = {}
    for w in wifi:
        for line in w.split('\n'):
            words = line.split(' ')
            if 'address:' in words:
                mac = words[-1]
            if 'signal' in words:
                strength = words[-4].split('=')[-1]
                dct[mac] = strength

    signal = np.zeros_like(bssids, dtype=int)
    for j, bssid in enumerate(bssids):
        try:
            print(bssid, dct[bssid])
            signal[j] = dct[bssid]
        except KeyError:
            pass

    pred = pca_knn.predict(pca.transform(scaler.transform(signal.reshape(1, -1))))
    print(pred)

    x = 1.85
    y = 10.4
    pose = map(float, rospy.myargv()[1:4])
    cov = map(float, rospy.myargv()[4:6])
    t_stamp = rospy.Time()
    t_publish = rospy.Time()
    rospy.init_node('pose_setter', anonymous=True)
    rospy.loginfo("Going to publish pose {} with stamp {} at {}".format(pose, t_stamp.to_sec(), t_publish.to_sec()))
    pub = rospy.Publisher("initialpose", PoseWithCovarianceStamped, PoseSetter(x, y, cov, stamp=t_stamp, publish_time=t_publish), queue_size=1)

    # rospy.init_node('param_setter', anonymous=True)
    rospy.sleep(2)
    client = dynamic_reconfigure.client.Client("amcl")
    client.update_configuration({"update_min_a": 0})
    client.update_configuration({"update_min_d": 0})

    rospy.sleep(5)
    client = dynamic_reconfigure.client.Client("amcl")
    client.update_configuration({"update_min_a": 0.2})
    client.update_configuration({"update_min_d": 0.5})

    rospy.spin()