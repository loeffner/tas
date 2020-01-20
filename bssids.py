### Author: Andreas Loesel, ga63miy
## 
### This Program was used to record and store all available BSSIDS.

import subprocess
import numpy as np

# Never hardcode your password!
sudo_pass = 'YourPassword'
interface = 'YourInterface'

# iwlist cmd
cmd = 'sudo iwlist ' + interface + ' scan'

def collect_bssids(bssids):
    cmd1 = subprocess.Popen(['echo',sudo_pass], stdout=subprocess.PIPE)
    cmd2 = subprocess.Popen(['sudo','-S'] + cmd.split(), stdin=cmd1.stdout, stdout=subprocess.PIPE)
    print('Scanning...')
    wifi = cmd2.stdout.read().decode().lower()

    # Retrieve MAC
    wifi = wifi.split('cell')
    for w in wifi:
        for line in w.split('\n'):
            words = line.split(' ')
            if 'address:' in words:
                if words[-1] not in bssids:
                    bssids.add(words[-1])
                    print('Found new bssid:', words[-1], 'Total number of bssids:', len(bssids))

try:
    bssids = set(np.loadtxt('bssids.csv', dtype=str))
except:
    bssids = set()
while True:
    collect_bssids(bssids)
    # np.savetxt('bssids.csv', np.array(sorted(bssids), dtype=str), delimiter=',', fmt='%s')



    
