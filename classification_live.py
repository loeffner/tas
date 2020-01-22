import subprocess
import random
import time
import rospy
import math
import PyKDL
import dynamic_reconfigure.client
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Never hardcode your password!
sudo_pass = 'YourPassword'
interface = 'YourInterface'
# iwlist cmd
cmd = 'sudo iwlist ' + interface + ' scan'

# bssid file
bssid_file = 'bssids.csv'

# data files
car_files_comb = [(r'data\auto_gang_hinten_comb', 'gang_hinten', (r'data\auto_gang_lab_comb', 'gang_lab'), (r'data\auto_gang_notaus_comb', 'gang_notaus'), (r'data\auto_kreuz_comb', 'kreuz'), (r'data\auto_lab_hinten_comb', 'lab_hinten'), (r'data\auto_lab_vorne_comb', 'lab_vorne'), (r'data\auto_treppen_comb', ('treppen', -6.8, 8.6))]
coords = {  'gang_hinten':  (-9.6, 15.9),
            'gang_lab':     (-2.8, -0.6),
            'gang_notaus':  (4.3, 14.9),
            'kreuz':        (2, 10.5),
            'lab_hinten':   (-1.5, -8.4),
            'lab_vorne':    (1.5, -5),
            'treppen':      (-6.8, 8.6) }

class PoseSetter(rospy.SubscribeListener):
    def __init__(self, x, y, cov, stamp, publish_time):
        self.x = x
        self.y = y
        self.cov = cov
        self.stamp = stamp
        self.publish_time = publish_time

    def peer_subscribe(self, topic_name, topic_publish, peer_publish):
        p = PoseWithCovarianceStamped()
        p.header.frame_id = "map"
        p.header.stamp = self.stamp
        p.pose.pose.position.x = self.x
        p.pose.pose.position.y = self.y
        (p.pose.pose.orientation.x,
         p.pose.pose.orientation.y,
         p.pose.pose.orientation.z,
         p.pose.pose.orientation.w) = PyKDL.Rotation.RPY(0, 0, 1).GetQuaternion()
        p.pose.covariance[6*0+0] = 1
        p.pose.covariance[6*1+1] = 1
        p.pose.covariance[6*3+3] = math.pi/8.0

        # wait for the desired publish time
        while rospy.get_rostime() < self.publish_time:
            rospy.sleep(0.01)
        peer_publish(p)

def load(data_file, label):
    return np.loadtxt(data_file, dtype=float), label
    
def load_bssids(bssid_file):
    return np.loadtxt(bssid_file, dtype=str)

def bar_plot(data, x_tick_labels, title='', ax=plt, shift=0, width=0.4, color=None, label=None):
    x = np.arange(len(data))
    ax.bar(x + width * shift, data, width=width, color=color, label=label)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    # ax.set_ylim(0, 70)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

def positive(data):
    data[np.where(data != 0)] = 100 + data[np.where(data != 0)]

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

def occurence(labels, rel=True):
    unique, counts = np.unique(labels, return_counts=True)
    if rel:
        return (dict(zip(unique, counts/labels.shape[0])))
    else:
        return (dict(zip(unique, counts)))


# Load data files
data = []
for sample_set in car_files_comb:
    data.append(load(sample_set[0], sample_set[1]))
[positive(d[0]) for d in data]

# Load bssid list
bssids = load_bssids(bssid_file)
bssids_short = [bssid[-3:] for bssid in bssids]

# Split data into train & test
(train_spls, train_lbls), (test_spls, test_lbls) = train_test(data, train_test_ratio=10)

# Scale the data
scaler = StandardScaler()
train_spls = scaler.fit_transform(train_spls)

# PCA (Reduction to 5 dimensions, these might represent the 5 different routers in the environment)
pca = PCA(n_components=5)
pca_data = pca.fit_transform(train_spls)

# PCA + KNN Classifier
pca_knn = KNeighborsClassifier(n_neighbors=3)
pca_knn.fit(pca_data, train_lbls)

print('Starting wifi sample collection')
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

signal = np.zeros_like(bssids, dtype=float)
for j, bssid in enumerate(bssids):
    try:
        print(bssid, dct[bssid])
        signal[j] = float(dct[bssid])
    except KeyError:
        pass

positive(signal)
pred = pca_knn.predict(pca.transform(scaler.transform(signal.reshape(1, -1))))
print(pred, coords[pred])

x = coords[pred][0]
y = coords[pred][1]
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
