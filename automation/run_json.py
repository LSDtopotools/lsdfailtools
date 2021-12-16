import json

def read_paths_file():
    with open("/exports/csce/datastore/geos/users/s1440040/projects/lsdfailtools/automation/file_paths_landslide_automation.json") as file_with_paths :
        FILE_PATHS = json.load(file_with_paths)
    return FILE_PATHS
