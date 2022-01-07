import json

def read_paths_file(fname):
    """
    read_paths_file reads the .json file with the file paths to the data and the new files create.
    :returns:
        - FILE_PATHS: python dictionary of the file paths
    """
    with open(fname) as file_with_paths :
        FILE_PATHS = json.load(file_with_paths)
    return FILE_PATHS
