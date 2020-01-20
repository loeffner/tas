### Author: Andreas Loesel, ga63miy
##
### This program can be used to record and store samples for localization using wifi.
### I collected samples at 7 different locations and ran classifiers to locate the car.


import subprocess
import numpy as np

# Never hardcode your password!
sudo_pass = 'YourPassword'
# Your Network Interface Card
interface = 'YourInterface'

# iwlist cmd
cmd = 'sudo iwlist ' + interface + ' scan'

# max Number of samples to collect (you can CTRL-C at any moment, without data loss)
NR_SAMPLES = 500

# Sample files
fl = 'YourSampleFile'
# fl = 'auto_lab_hinten_2'
# fl = 'auto_lab_vorne_2'
# fl = 'auto_gang_lab_2'
# fl = 'auto_gang_notaus_2'
# fl = 'auto_gang_hinten_2'
# fl = 'auto_kreuz_2'
# fl = 'auto_treppen_2'

# In this file the BSSIDS we want to use are stored
bssids = np.loadtxt('bssids.csv', dtype=str)

# If a sample file is already available, the new data will be appended
samples = np.loadtxt(fl, dtype = float)

print(samples.shape)

for i in range(NR_SAMPLES):
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
            signal[j] = dct[bssid]
        except KeyError:
            pass
    
    samples = np.append(samples, signal.reshape(1, -1), axis=0)

    print('Current collection:', i+1, 'Total:', samples.shape[0])
    np.savetxt(fl, samples)
    
