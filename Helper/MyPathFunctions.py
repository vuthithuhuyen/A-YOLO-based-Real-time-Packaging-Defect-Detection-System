import pathlib
import platform
import os
import string


def GetCWD():
    myWorkingdirectory = pathlib.Path(os.getcwd())
    if platform.system() == 'Windows':
        myWorkingdirectory = pathlib.Path(pathlib.PureWindowsPath(myWorkingdirectory))
    return (myWorkingdirectory)


def GetAllFilesInDirectory(directory: pathlib.Path):
    file_list = []
    for x in directory.iterdir():
        if x.is_file():
            file_list.append(x)
    return file_list


# Get list of Disk in Windows. E.g., [C:, D:, E:]
def GetListOfDisk():
    if platform.system() == 'Windows':
        available_drives = ['%s:' % d for d in string.ascii_uppercase if os.path.exists('%s:' % d)]
    else:
        available_drives = []
    return available_drives