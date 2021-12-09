import os
import sys

def remove_unwanted_files():
    path = './'
    os.chdir(path)
    flist = open('/exports/csce/datastore/geos/users/s1440040/projects/lsdfailtools/automation/files_to_delete.txt')
    for f in flist:
        fname = f.rstrip() # or depending on situation: f.rstrip('\n')
        # or, if you get rid of os.chdir(path) above,
        # fname = os.path.join(path, f.rstrip())
        if os.path.isfile(fname): # this makes the code more robust
            os.remove(fname)

    # also, don't forget to close the text file:
    flist.close()


    files_in_directory = os.listdir(path)
    filtered_files = [file for file in files_in_directory if file.endswith(".npy")]
    for file in filtered_files:
    	path_to_file = os.path.join(path, file)
    	os.remove(path_to_file)
