import sys
import numpy as np
import shutil

BASE = "/scratch/jb6504/recycling_strong_lensing/data/samples/calibration/"
FILENAME_NEW = "x_calibrate_full_theta_{}_updated.npy"
FILENAME_OLD = "x_calibrate_full_theta_{}.npy"
FILENAME_OUT = "x_calibrate_full_theta_{}_out.npy"


def check(filename):
    try:
        x = np.load(filename)
        return x.shape[0]
    except:
        return -1


for i in range(625):
    # Check that files exist

    n_old = check(BASE + FILENAME_OLD.format(i))
    n_new = check(BASE + FILENAME_NEW.format(i))
    n = max(n_old, n_new)

    if n < 1:
        print("NO DATA for i = {}: {}, {}".format(i, n_old, n_new))
        continue

    if n < 10000:
        print("Little data for i = {}: {}, {}".format(i, n_old, n_new))

    if n_new > n_old:
        print("n {}".format(n_new))
        source = FILENAME_NEW.format(i)
    else:
        print("o {}".format(n_old))
        source = FILENAME_OLD.format(i)

    try:
        shutil.copy(BASE + source, BASE + FILENAME_OLD.format(i))
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())

print("Done!")
