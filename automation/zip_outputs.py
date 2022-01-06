from zipfile import ZipFile
import os

def zip_output_files(rundir):
    """
    zip_output_files creates a zip files from the output csv files
    :param rundir: path to the directory where the output files are
    :returns:
        - files_to_zip - list of the files that have been zipped
    """
    zipObj = ZipFile('landslide_failures_output.zip', 'w')

    # Add multiple files to the zip
    files_to_zip = []
    for file in os.listdir(rundir):
        if file.startswith('fos_timeseries'):
            print(file)
            files_to_zip.append(file)

    for i in range(len(files_to_zip)):
        zipObj.write(files_to_zip[i])

    # close the Zip File
    zipObj.close()
    return files_to_zip

def remove_zipped_files(files_to_remove):
    """
    remove_zipped files remove the files that have already been zipped
    :param files_to_remove: list of files that have been zipped and need deleted
    """
    for i in range(len(files_to_remove)):
        os.remove(files_to_remove[i])
