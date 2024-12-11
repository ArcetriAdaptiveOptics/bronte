import os
from pathlib import Path
ROOT_DIR_KEY = 'BRONTE_ROOT_DIR'


def set_data_root_dir(folder):
    os.environ[ROOT_DIR_KEY] = folder


def data_root_dir():

    try:
        return Path(os.environ[ROOT_DIR_KEY])
    except KeyError:
        import pkg_resources
        dataroot = pkg_resources.resource_filename(
            'bronte',
            'data')
        return Path(dataroot)


def snapshot_folder():
    return data_root_dir() / "snapshots"

def elt_pupil_folder():
    return data_root_dir() / "elt_pupil"

def subaperture_set_folder():
    return data_root_dir() / "subaperture_set"

def reconstructor_folder():
    return data_root_dir() / "reconstructor"

def phase_screen_folder():
    return data_root_dir() / "phase_screens"

def modal_decomposer_folder():
    return data_root_dir() / "modal_decomposer"

def telemetry_folder():
    return data_root_dir() / "telemetry"

def shframes_folder(): 
    return data_root_dir() / "shwfs_frames"

def modal_offsets_folder():
    return data_root_dir() / "modal_offsets"

def temp_folder():
    return data_root_dir() / "temp_trash"