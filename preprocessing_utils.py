import os
import shutil
import subprocess
from pathlib import Path

import yaml


def extract_frames_from_video(input_video: str, output_images_pattern: str):
    """
    Extracts frames from a video file and saves them as PNG images.

    Parameters:
    input_video (str): Path to the input video file.
    output_images_pattern (str): Pattern for the output image files.
                                 For example, 'output/images/%04d.png'.
    """
    # Construct the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_video,
        "-r", "24",  # Set the frame rate to 24 fps
        "-c:v", "png",
        output_images_pattern
    ]
    print(ffmpeg_command)
    # Run the command using subprocess
    try:
        subprocess.run(ffmpeg_command, check=True)
        print("Video frames extracted successfully.")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False, e


def copy_smpl_file(cache_dir_4dhumans: str):
    """
    Copies the SMPL_NEUTRAL.pkl file from the source directory to the destination directory.
    Ensures that the destination directory exists.

    Parameters:
    cache_dir_4dhumans (str): The cache directory containing the 4D Humans data.
    """
    destination_file = os.path.join(cache_dir_4dhumans, 'data', 'smpl', 'SMPL_NEUTRAL.pkl')
    source_file = 'data/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
    destination_dir = os.path.dirname(destination_file)

    # Ensure that the destination directory exists
    Path(destination_dir).mkdir(exist_ok=True, parents=True)

    try:
        shutil.copy(source_file, destination_file)
        print(f"File copied successfully from '{source_file}' to '{destination_file}'.")
        return True, None
    except IOError as e:
        print(f"Error occurred while copying file: {e}")
        return False, e


def create_symbolic_link():
    """
    Creates a symbolic link pointing to the source if it does not exist.

    Parameters:
    source (str): The target directory the symbolic link points to.
    link_name (str): The name of the symbolic link to be created.
    """
    source = "4D-Humans/data"
    link_name = "data"

    link_path = Path(link_name)
    print(os.listdir('4D-Humans/data'))
    if not link_path.exists():
        try:
            os.symlink(source, link_name)
            print(f"Symbolic link '{link_name}' created successfully.")
            return True, None
        except OSError as e:
            print(f"Error occurred while creating symbolic link: {e}")
            return False, e
    else:
        print(f"Symbolic link '{link_name}' already exists.")
        return False, Exception("Symbolic link '{link_name}' already exists.")


def rename_smpl_file():
    """
    Renames the SMPL file from basicModel_neutral_lbs_10_207_0_v1.1.0.pkl to basicModel_neutral_lbs_10_207_0_v1.0.0.pkl.
    """
    old_file = '4D-Humans/data/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
    new_file = '4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    print(os.listdir('.'))
    print(os.listdir('4D-Humans'))
    print(os.listdir('4D-Humans/data'))
    try:
        os.rename(src=old_file, dst=new_file)
        print(f"File renamed successfully from '{old_file}' to '{new_file}'.")
        return True, None
    except FileNotFoundError as e:
        print(f"Error occurred: {e}")
        return False, e
    except OSError as e:
        print(f"Error occurred: {e}")
        return False, e


def generate_smpls(reference_imgs_folder: str, driving_video_path: str, device: str):
    """
    Runs the SMPL generation script with the specified parameters.

    Parameters:
    reference_imgs_folder (str): Path to the folder containing reference images.
    driving_video_path (str): Path to the driving video.
    device (str): ID of the GPU device to use.
    """
    # Construct the command
    command = [
        "python",
        "-m", "scripts.data_processors.smpl.generate_smpls",
        "--reference_imgs_folder", reference_imgs_folder,
        "--driving_video_path", driving_video_path,
        "--device", device
    ]

    # Run the command using subprocess
    try:
        create_symbolic_link()
        copy_smpl_file("~/.cache/4DHumans")
        rename_smpl_file()
        subprocess.run(command, check=True)
        print("SMPL generation completed successfully.")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False, e


def smooth_smpl(smpls_group_path: str, smoothed_result_path: str):
    """
    Runs the Blender script to smooth SMPLs using the specified parameters.

    Parameters:
    smpls_group_path (str): Path to the input SMPLs group file.
    smoothed_result_path (str): Path to the output smoothed result file.
    """
    # Construct the Blender command
    blender_command = [
        "blender",
        "--background",
        "--python", "scripts/data_processors/smpl/smooth_smpls.py",
        "--smpls_group_path", smpls_group_path,
        "--smoothed_result_path", smoothed_result_path
    ]

    # Run the command using subprocess
    try:
        subprocess.run(blender_command, check=True)
        print("Blender script ran successfully.")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return True, e # Blender always throws error here


def transfer_smpl(reference_path: str, driving_path: str, output_folder: str):
    """
    Runs the SMPL transfer script with the specified parameters.

    Parameters:
    reference_path (str): Path to the reference image file.
    driving_path (str): Path to the driving video folder.
    output_folder (str): Path to the output folder.
    """
    # Construct the command
    command = [
        "python",
        "-m", "scripts.data_processors.smpl.smpl_transfer",
        "--reference_path", reference_path,
        "--driving_path", driving_path,
        "--output_folder", output_folder,
        "--figure_transfer",
        "--view_transfer"
    ]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print("SMPL transfer script ran successfully.")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False, e


def run_blender_rendering(driving_path: str, reference_path: str):
    """
    Runs the Blender rendering script with the specified parameters.

    Parameters:
    driving_path (str): Path to the driving SMPL results.
    reference_path (str): Path to the reference image file.
    """
    # Construct the Blender command
    blender_command = [
        "blender", "scripts/data_processors/smpl/blend/smpl_rendering.blend",
        "--background",
        "--python", "scripts/data_processors/smpl/render_condition_maps.py",
        "--",  # This separates Blender's own arguments from the script arguments
        "--driving_path", driving_path,
        "--reference_path", reference_path,
    ]

    # Run the command using subprocess
    try:
        subprocess.run(blender_command, check=True)
        print("Blender rendering script ran successfully.")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False, e


def run_generate_dwpose(input_path: str, output_path: str):
    """
    Runs the DW Pose generation script with the specified parameters.

    Parameters:
    input_path (str): Path to the input folder.
    output_path (str): Path to the output folder.
    """
    # Construct the command
    command = [
        "python",
        "-m", "scripts.data_processors.dwpose.generate_dwpose",
        "--input", input_path,
        "--output", output_path
    ]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print("DW Pose generation script ran successfully.")
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False, e


def update_yaml_with_dict(yaml_file_path: str, updated_file_path: str, update_dict: dict):
    """
    Updates the values in a YAML file with the values from a dictionary.

    Parameters:
    yaml_file_path (str): Path to the YAML file.
    update_dict (dict): Dictionary containing the values to update.
    updated_file_path (str):
    """
    try:
        # Read the existing data from the YAML file
        with open(yaml_file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        # Update the YAML data with the values from the dictionary
        yaml_data.update(update_dict)

        # Write the updated data back to the YAML file
        with open(updated_file_path, 'w') as file:
            yaml.safe_dump(yaml_data, file)

        print(f"Updated YAML file '{yaml_file_path}' saved successfully at {updated_file_path}.")
        return True, None
    except Exception as e:
        print(f"Error occurred: {e}")
        return False, e
