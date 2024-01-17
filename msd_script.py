import os
import sys
import glob
import numpy as np

msd_subset_path = "/Users/de1ukc/git/music-research/msd_dataset/MillionSongSubset"
# msd_subset_data_path = os.path.join(msd_subset_path, "data")
# msd_subset_addf_path = os.path.join(msd_subset_path, "AdditionalFiles")
assert os.path.isdir(msd_subset_path), "wrong path"  # sanity check

# You need to have the MSD code downloaded from GITHUB. https://github.com/tbertinmahieux/MSongsDB
msd_code_path = "/Users/de1ukc/music-analysis/MSongsDB-master"
assert os.path.isdir(msd_code_path), "wrong path"  # sanity check

sys.path.append(os.path.join(msd_code_path, "PythonSrc"))

import hdf5_getters as GETTERS


# we define this very useful function to iterate the files
def apply_to_all_files(basedir, func=lambda x: x, ext=".h5"):
    """
    From a base directory , go through all subdirectories , find all files with the given extension, apply the
    given function 'func' to all of them.
    If no 'func' is passed, we do nothing except counting. INPUT
    basedir - base directory of the dataset
    func - function to apply to all filenames ext - extension , .h5 by default
    RETURN
    number of files
    """
    cnt = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, "*" + ext))  # count files
        cnt += len(files)

        # apply function to all files
        for f in files:
            func(f)

    return cnt


print('number of songs', apply_to_all_files(basedir=msd_subset_path))



