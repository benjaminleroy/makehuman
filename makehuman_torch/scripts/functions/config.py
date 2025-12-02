import os
import sys

MAKEHUMAN_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'makehuman'
        )
    )


def main_folder_preparation():
    """Prepare access to MakeHuman internal libraries.
    
    Details:
        This function sets up the system path and user directories
        required to access MakeHuman's internal libraries.
    """
    original_dir = os.getcwd()
    os.chdir(MAKEHUMAN_DIR)

    from makehuman import (
        set_sys_path, 
        make_user_dir,
        get_platform_paths
    )
    set_sys_path()
    make_user_dir()
    get_platform_paths()

    os.chdir(original_dir)

