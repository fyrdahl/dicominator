#!/usr/bin/env python
"""DICOMinator - A tool to sort 4D Flow images.

This script takes the root folder containing DICOMs as input and recursively traverses
the folder to find all the DICOM files. It then sorts the DICOMs into a specified output
folder. It provides options to save the sorted data in various formats:
- NIfTI: The NIfTI files are split along the time dimension, allowing them to be imported as a MultiVolume in "3D Slicer".
- h5: The h5 format is compatible with Edward Ferdian's "4DFlowNet".
- mat: The mat format is specifically designed for Julio Sotelo's "4D-Flow-Matlab-Toolbox".
"""

import argparse
import contextlib
import copy
import glob
import os
import re
import unicodedata
import shutil
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pydicom
import scipy.io as sio
from tqdm import tqdm
from nibabel.spatialimages import HeaderDataError

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FlowDirectionNotFoundError(Exception):
    pass


def dicominator(
    input_root,
    output_root=None,
    description=None,
    save_as_h5=False,
    save_as_mat=False,
    save_as_nii=False,
    save_pcmra=False,
    list_descriptions=False,
    force=False,
):
    """
    DICOMinator main routine.

    This function takes an input root directory containing DICOM files and either lists all unique
    Series Descriptions in the dataset or processes the files and saves the results in the specified
    output root directory. If listing descriptions, no further processing is done. If sorting files,
    it can filter the files based on a given description and save the processed data in various formats
    (h5, mat, nii).

    Args:
        input_root (str): The root directory containing the DICOM files.
        output_root (str): The root directory where the processed files will be saved.
        description (str, optional): The series description to filter the DICOM files. Defaults to None.
        save_as_h5 (bool, optional): Whether to save the processed data in h5 format. Defaults to False.
        save_as_mat (bool, optional): Whether to save the processed data in mat format. Defaults to False.
        save_as_nii (bool, optional): Whether to save the processed data in nii format. Defaults to False.
        save_pcmra (bool, optional): Whether to save pcmra images in nii format. Defaults to False.
        list_descriptions (bool, optional): Whether to list all unique Series Descriptions in the dataset. Defaults to False.

    Returns:
        None
    """
    if not list_descriptions and not output_root:
        raise ValueError("Output root directory must be provided")

    if force:
        logging.info(
            "Force assumes that all datasets are flow datasets. Use with caution!"
        )

    dcm_extensions = {"dc", "dcm", "dic", "dicom", "ima", "img"}
    dcm_files = {
        f for ext in dcm_extensions for f in Path(input_root).rglob(f"*.{ext}")
    }

    if not dcm_files:
        logging.info("No files with known DICOM extensions found.")
        logging.info(
            "Its possible the exam was exported with the Create DICOM File System option turned on."
        )
        dcm_files = {
            f for f in Path(input_root).rglob("*") if f.is_file() and not f.suffix
        }
        logging.info(
            f"Found {len(dcm_files)} extensionless files in {input_root} that could be DICOM files"
        )
    else:
        logging.info(f"Found {len(dcm_files)} DICOM files in {input_root}")

    if list_descriptions:
        logging.info("Listing unique Series Descriptions")
    else:
        logging.info(
            f"Filtering by Series Description: {description}"
            if description
            else "No filter applied"
        )
        logging.info(f"Sorted data will be saved in {output_root}")

    descriptions = set()
    flow_status = {}
    subfolders_to_process = set()

    for file_path in tqdm(dcm_files, desc="Processing files"):
        with pydicom.dcmread(file_path, stop_before_pixels=True, force=True) as ds:
            protocol_name = getattr(ds, "ProtocolName", None)
            if protocol_name:
                base_desc = get_base_desc(ds.SeriesDescription)
                descriptions.add(ds.SeriesDescription)

                if list_descriptions or is_flow_dataset(ds, force=force):
                    if list_descriptions:
                        if protocol_name not in flow_status:
                            flow_status[protocol_name] = False
                        if is_flow_dataset(ds, force=force):
                            flow_status[protocol_name] = True
                            subfolders_to_process.add(base_desc)
                    else:
                        if not description or normalize_string(base_desc).startswith(
                            normalize_string(description)
                        ):
                            if is_flow_dataset(ds, force=force):
                                subfolders_to_process.add(base_desc)
                            subfolder = os.path.join(
                                output_root, sanitize_name(base_desc)
                            )
                            base_name = os.path.basename(file_path)

                            # Load the full file
                            with pydicom.dcmread(
                                file_path, stop_before_pixels=False
                            ) as ds:
                                if getattr(ds, "NumberOfFrames", 0) > 1:
                                    split_and_save_multiframe_dicom(
                                        ds, base_name, subfolder
                                    )
                                else:
                                    output_path = get_output_path(ds, subfolder)
                                    if output_path:
                                        save_dicom(ds, output_path, base_name)
    if list_descriptions:
        display_descriptions(descriptions, flow_status, force)
    else:
        # Perform post-processing steps for each processed subfolder
        for series_description in subfolders_to_process:
            subfolder = os.path.join(output_root, sanitize_name(series_description))
            try:
                validate_folder_structure(subfolder)
            except Exception as e:
                logging.warning(f"{series_description}: {e}")
                continue
            if save_as_h5 or save_as_mat or save_as_nii:
                process_and_save_data(
                    subfolder, save_as_h5, save_as_mat, save_as_nii, save_pcmra
                )

    logging.info("Done!")


def normalize_string(text):
    """
    Normalize a string by:
    1. Removing diacritics (accents)
    2. Removing all non-alphanumeric characters (including spaces)
    3. Converting to lowercase
    """
    # Remove diacritics (accents)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )

    # Remove all non-alphanumeric characters (including spaces)
    text = re.sub(r"[^a-zA-Z0-9]", "", text)

    # Convert to lowercase
    return text.lower()


def display_descriptions(descriptions, flow_status, force):
    unique_descriptions = {
        desc
        for desc in sorted(descriptions)
        if not desc.endswith("_P") or desc[:-2] not in descriptions
    }
    logging.info("Unique Series Descriptions:")
    for desc in sorted(unique_descriptions):
        if force:
            logging.info(f"- {desc} (Assumed to be a flow dataset by --force)")
        elif any(
            flow_status[protocol_name]
            for protocol_name in flow_status
            if protocol_name.startswith(desc)
        ):
            logging.info(f"- {desc} (Likely a flow dataset)")
        else:
            logging.info(f"- {desc}")


def get_base_desc(series_description):
    if series_description.endswith("_P"):
        return series_description[:-2]
    return series_description


def is_flow_dataset(ds, force=False):
    """
    Check if a DICOM dataset is likely to be a flow dataset.

    Args:
        ds (pydicom.Dataset): The DICOM dataset to examine.

    Returns:
        bool: True if the dataset is likely to be a flow dataset, False otherwise.
    """
    if force:
        return True
    try:
        if not hasattr(ds, "SequenceName"):
            return False

        with contextlib.suppress(KeyError):
            if ds[0x0021, 0x1049].value == "FLOW_ENCODED":
                return True

        try:
            _ = get_venc(ds.SequenceName)
        except ValueError:
            return False

        # Check the presence of required tags for processing
        required_tags = [
            "PixelSpacing",
            "SliceThickness",
            "TriggerTime",
            "Rows",
            "Columns",
        ]
        missing_tags = [tag for tag in required_tags if not hasattr(ds, tag)]
        if missing_tags:
            return False

        return True

    except AttributeError:
        return False


def split_and_save_multiframe_dicom(ds, base_name, output_root):
    """
    Split a multiframe DICOM dataset into single frames and save each frame as a separate DICOM file.

    Args:
        ds (pydicom.Dataset): The multiframe DICOM dataset to be split.
        base_name (str): The base name of the DICOM file.
        output_root (str): The root directory where the split DICOM files will be saved.

    Returns:
        None
    """
    for i, sub_array in enumerate(ds.pixel_array):
        new_ds = copy.deepcopy(ds)
        new_ds.PixelData = sub_array.tobytes()
        new_ds.NumberOfFrames = 1
        new_base_name = f"{base_name.split('.dcm')[0]}_frame_{i + 1}.dcm"
        output_path = get_output_path(new_ds, output_root)
        if output_path:
            save_dicom(new_ds, output_path, new_base_name)


def process_and_save_data(
    output_root, save_as_h5, save_as_mat, save_as_nii, save_pcmra
):
    """
    Process the sorted DICOM files and save the data in different formats (h5, mat, nii).

    Args:
        output_root (str): The root directory containing the sorted DICOM files.
        description (str): The series description used for naming the output files.
        save_as_h5 (bool): Whether to save the processed data in h5 format.
        save_as_mat (bool): Whether to save the processed data in mat format.
        save_as_nii (bool): Whether to save the processed data in nii format.
        save_pcmra (bool): Whether to save pcmra images.

    Returns:
        None
    """
    if not os.path.exists(output_root):
        return

    file_name = os.path.basename(output_root)
    processed_files = glob.glob(os.path.join(output_root, "**/*"), recursive=True)
    processed_files = [f for f in processed_files if os.path.isfile(f)]

    logging.info(
        f"Found {len(processed_files)} files with SeriesDescription {file_name}"
    )

    sample_ds = pydicom.dcmread(processed_files[0])
    if hasattr(sample_ds, "NumberOfTemporalPositions"):
        num_cardiac_phases = int(sample_ds.NumberOfTemporalPositions)
    elif hasattr(sample_ds, "CardiacNumberOfImages"):
        num_cardiac_phases = int(sample_ds.CardiacNumberOfImages)
    else:
        num_cardiac_phases = 1
    num_slices = int(len(processed_files) / num_cardiac_phases / 4)
    logging.info(
        f"Dataset has {num_slices} slices with {num_cardiac_phases} cardiac phases"
    )

    rows, cols = sample_ds.Rows, sample_ds.Columns
    images_tot = len(processed_files) // 4

    image_data, venc_data, pos_pat, tt_pat, count, ds_list = initialize_data_structures(
        rows, cols, images_tot
    )

    for file_path in tqdm(
        processed_files, desc=f"Processing {os.path.basename(output_root)}"
    ):
        process_file(file_path, image_data, venc_data, pos_pat, tt_pat, count, ds_list)

    for key in ["MAG", "AP", "RL", "FH"]:
        sort_data(
            key,
            image_data,
            venc_data,
            pos_pat,
            tt_pat,
            ds_list,
            num_slices,
            num_cardiac_phases,
        )

    if save_as_nii:
        save_nii_files(output_root, image_data, tt_pat, ds_list, save_pcmra)

    if save_as_h5 or save_as_mat:
        data = prepare_data_for_saving(
            image_data, venc_data, pos_pat, tt_pat, sample_ds
        )
    if save_as_h5:
        save_h5_file(output_root, file_name, data)
    if save_as_mat:
        save_mat_file(output_root, file_name, data)


def initialize_data_structures(rows, cols, images_tot):
    """
    Initialize the data structures for storing image data, VENC data, position, trigger time, and DICOM datasets.

    Args:
        rows (int): The number of rows in the image.
        cols (int): The number of columns in the image.
        images_tot (int): The total number of images.

    Returns:
        tuple: A tuple containing the initialized data structures.
    """
    image_data = {
        "MAG": np.zeros((rows, cols, images_tot)),
        "FH": np.zeros((rows, cols, images_tot)),
        "RL": np.zeros((rows, cols, images_tot)),
        "AP": np.zeros((rows, cols, images_tot)),
    }

    venc_data = {
        "AP": np.zeros((1, images_tot)),
        "RL": np.zeros((1, images_tot)),
        "FH": np.zeros((1, images_tot)),
    }

    pos_pat = {
        "MAG": np.zeros((3, images_tot)),
        "FH": np.zeros((3, images_tot)),
        "RL": np.zeros((3, images_tot)),
        "AP": np.zeros((3, images_tot)),
    }

    tt_pat = {
        "MAG": np.zeros((1, images_tot)),
        "FH": np.zeros((1, images_tot)),
        "RL": np.zeros((1, images_tot)),
        "AP": np.zeros((1, images_tot)),
    }

    count = {"MAG": 0, "FH": 0, "RL": 0, "AP": 0}
    ds_list = {"MAG": [], "FH": [], "RL": [], "AP": []}

    return image_data, venc_data, pos_pat, tt_pat, count, ds_list


def process_file(file_path, image_data, venc_data, pos_pat, tt_pat, count, ds_list):
    """
    Process a single DICOM file and update the data structures.

    Args:
        file_path (str): The path to the DICOM file.
        image_data (dict): The dictionary to store image data.
        venc_data (dict): The dictionary to store VENC data.
        pos_pat (dict): The dictionary to store position data.
        tt_pat (dict): The dictionary to store trigger time data.
        count (dict): The dictionary to store count data.
        ds_list (dict): The dictionary to store DICOM datasets.

    Returns:
        None
    """
    file_path = Path(file_path)
    ds = pydicom.dcmread(file_path)
    parent_name = file_path.parent.name

    if parent_name in ["MAG", "AP", "RL", "FH"]:
        ds_list[parent_name].append(ds)
        image_data[parent_name][:, :, count[parent_name]] = ds.pixel_array.astype(float)
        if parent_name != "MAG":
            image_data[parent_name][:, :, count[parent_name]] *= ds.RescaleSlope
            image_data[parent_name][:, :, count[parent_name]] += ds.RescaleIntercept
            venc_data[parent_name][:, count[parent_name]] = get_venc(ds.SequenceName)

        pos_pat[parent_name][:, count[parent_name]] = ds.ImagePositionPatient
        tt_pat[parent_name][:, count[parent_name]] = ds.TriggerTime

        count[parent_name] += 1


def sort_data(
    key, image_data, venc_data, pos_pat, tt_pat, ds_list, num_slices, num_cardiac_phases
):
    """
    Sort the data based on slice direction and trigger time.

    Args:
        key (str): The key representing the data type (MAG, AP, RL, FH).
        image_data (dict): The dictionary containing image data.
        venc_data (dict): The dictionary containing VENC data.
        pos_pat (dict): The dictionary containing position data.
        tt_pat (dict): The dictionary containing trigger time data.
        ds_list (dict): The dictionary containing DICOM datasets.
        num_slices (int): The number of slices.
        num_cardiac_phases (int): The number of cardiac phases.

    Returns:
        None
    """
    slice_dir = np.argmax(np.mean(np.abs(np.diff(pos_pat[key], axis=1)), axis=1))
    slice_indices = np.argsort(pos_pat[key][slice_dir, :])
    tt_sorted_indices = np.argsort(
        tt_pat[key][:, slice_indices].reshape(num_slices, num_cardiac_phases),
        axis=1,
    )
    idx_sort = slice_indices.reshape(num_slices, num_cardiac_phases)[
        np.arange(num_slices)[:, None], tt_sorted_indices
    ]

    image_data[key] = image_data[key][:, :, idx_sort]
    pos_pat[key] = pos_pat[key][:, idx_sort]
    tt_pat[key] = tt_pat[key][:, idx_sort]
    ds_list[key] = [ds_list[key][int(i)] for i in idx_sort.ravel()]
    if key != "MAG":
        venc_data[key] = venc_data[key][:, idx_sort]


def save_nii_files(output_root, image_data, tt_pat, ds_list, save_pcmra):
    """
    Save the image data as NII files.

    Args:
        output_root (str): The root directory for saving the NII files.
        image_data (dict): The dictionary containing image data.
        tt_pat (dict): The dictionary containing trigger time data.
        ds_list (dict): The dictionary containing DICOM datasets.
        save_pcmra (bool): Whether to save pcmra images.

    Returns:
        None
    """
    if not os.path.exists(os.path.join(output_root, "nii")):
        os.makedirs(os.path.join(output_root, "nii"))

    keys = ["MAG", "AP", "RL", "FH"]

    if save_pcmra:
        velocity_data = np.stack(
            [image_data[key] for key in ["AP", "RL", "FH"]],
            axis=-1,
        )
        speed = np.sqrt(np.sum(velocity_data**2, axis=-1))

        mag_data = image_data["MAG"]
        min_mag = np.min(0.7 * mag_data)
        max_mag = np.max(0.7 * mag_data)
        mag_data = np.clip(mag_data, min_mag, max_mag)
        mag_data = (mag_data - min_mag) / (max_mag - min_mag)

        pcmra = np.mean((speed * mag_data) ** 2, axis=-1)
        p2 = np.percentile(pcmra, 99.8)
        pcmra[pcmra > p2] = p2

        image_data["PCMRA"] = pcmra
        keys.append("PCMRA")
        ds_list["PCMRA"] = ds_list["MAG"]
        tt_pat["PCMRA"] = tt_pat["MAG"]

    for key in keys:
        if not os.path.exists(os.path.join(output_root, "nii", key)):
            os.makedirs(os.path.join(output_root, "nii", key))

        image_data_sorted = image_data[key]
        ds = ds_list[key][0]

        affine = affine3d(ds_list[key])
        pixel_spacing = ds.PixelSpacing
        slice_thickness = ds.SliceThickness
        voxel_dims = np.array(
            [
                pixel_spacing[0],
                pixel_spacing[1],
                slice_thickness,
                np.mean(np.diff(tt_pat[key])) * 10**-3,
            ]
        )

        header = nib.Nifti1Header()
        header.set_data_shape(image_data_sorted.shape)
        header.set_data_dtype(np.float32)
        try:
            header.set_zooms(voxel_dims)
        except HeaderDataError:
            header.set_zooms(voxel_dims[:3])
        header.set_xyzt_units(xyz="mm", t="msec")

        if len(image_data_sorted.shape) == 3:
            image_data_sorted = np.expand_dims(image_data_sorted, axis=3)

        for idx, sub_volume in enumerate(image_data_sorted.transpose(3, 0, 1, 2)):
            nifti_image = nib.Nifti1Image(sub_volume, affine, header=None)
            output_filename = os.path.join(output_root, "nii", key, f"{key}_{idx}.nii")
            nib.save(nifti_image, output_filename)


def affine3d(ds_list):
    """See: https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-affine-formula"""

    N = len(ds_list)
    if N < 2:
        raise ValueError(
            "ds_list must contain at least two datasets to compute affine matrix"
        )

    # Extract image positions
    positions = np.array([ds.ImagePositionPatient for ds in ds_list])
    T1, T2, T3 = positions[:, 0], positions[:, 1], positions[:, 2]

    # Extract orientation
    orientation = np.array(ds_list[0].ImageOrientationPatient)
    row_x, row_y, row_z = orientation[:3]
    col_x, col_y, col_z = orientation[3:]

    # Calculate slice direction
    slice_x, slice_y, slice_z = np.cross([row_x, row_y, row_z], [col_x, col_y, col_z])

    # Extract pixel spacing
    dr, dc = ds_list[0].PixelSpacing

    # Try to get slice thickness
    try:
        dslice = float(ds_list[0].SliceThickness)
    except AttributeError:
        try:
            dslice = float(ds_list[0][0x0018, 0x0050].value)
        except KeyError:
            dslice = np.linalg.norm(positions[-1] - positions[0]) / (N - 1)

    # Calculate affine matrix
    affine = np.array(
        [
            [row_x * dr, col_x * dc, slice_x * dslice, T1[0]],
            [row_y * dr, col_y * dc, slice_y * dslice, T2[0]],
            [row_z * dr, col_z * dc, slice_z * dslice, T3[0]],
            [0, 0, 0, 1],
        ]
    )

    return affine


def prepare_data_for_saving(image_data, venc_data, pos_pat, tt_pat, sample_ds):
    """
    Prepare the data dictionary for saving.

    Args:
        image_data (dict): The dictionary containing image data.
        venc_data (dict): The dictionary containing VENC data.
        pos_pat (dict): The dictionary containing position data.
        tt_pat (dict): The dictionary containing trigger time data.
        sample_ds (pydicom.Dataset): A sample DICOM dataset.

    Returns:
        dict: The prepared data dictionary.
    """
    # TODO: Check if this is correct
    phaseRange = 4095
    for key in ["FH", "AP", "RL"]:
        image_data[key] = (image_data[key] / phaseRange) * venc_data[key]
    tt = np.unique(tt_pat["MAG"]) * 10**-3

    return {
        "MR_FFE_FH": image_data["MAG"],
        "MR_FFE_AP": image_data["MAG"],
        "MR_FFE_RL": image_data["MAG"],
        "MR_PCA_FH": image_data["FH"],
        "MR_PCA_AP": image_data["AP"],
        "MR_PCA_RL": image_data["RL"],
        "POS_PAT": pos_pat["MAG"],
        "TT_PAT": tt_pat["MAG"],
        "VENC": venc_data["FH"][0],
        "voxel_MR": [
            sample_ds.PixelSpacing[0],
            sample_ds.PixelSpacing[1],
            sample_ds.SliceThickness,
        ],
        "heart_rate": round(60 / (sample_ds.NominalInterval / 1000)),
        "type": "DCM",
        "dt": np.mean(np.diff(tt)),
    }


def save_h5_file(output_root, file_name, data):
    """
    Save the data as an H5 file.

    Args:
        output_root (str): The root directory for saving the H5 file.
        file_name (str): The name of the H5 file.
        data (dict): The data dictionary to be saved.

    Returns:
        None
    """
    rename_h5 = {
        "MR_FFE_FH": "mag_w",
        "MR_FFE_AP": "mag_v",
        "MR_FFE_RL": "mag_u",
        "MR_PCA_FH": "w",
        "MR_PCA_AP": "v",
        "MR_PCA_RL": "u",
        "voxel_MR": "dx",
    }

    with h5py.File(os.path.join(output_root, f"{file_name}.h5"), "w") as f:
        for key, value in data.items():
            if key in rename_h5:
                f.create_dataset(rename_h5[key], data=value)

        f.create_dataset("u_max", data=data["VENC"][0])
        f.create_dataset("v_max", data=data["VENC"][0])
        f.create_dataset("w_max", data=data["VENC"][0])

    logging.info(f"Saved h5-file to {os.path.join(output_root, f'{file_name}.h5')}")


def save_mat_file(output_root, file_name, data):
    """
    Save the data as a MAT file.

    Args:
        output_root (str): The root directory for saving the MAT file.
        file_name (str): The name of the MAT file.
        data (dict): The data dictionary to be saved.

    Returns:
        None
    """
    if not os.path.exists(os.path.join(output_root, file_name)):
        os.makedirs(os.path.join(output_root, file_name))
    sio.savemat(os.path.join(output_root, file_name, "data.mat"), {"data": data})
    logging.info(f"Saved mat-file to {os.path.join(output_root, 'data/data.mat')}")


def get_venc(sequence_name):
    """
    Extract the numeric value (VENC) from the sequence name.

    Args:
        sequence_name (str): The sequence name containing the VENC value.

    Returns:
        float: The VENC value extracted from the sequence name.

    Raises:
        ValueError: If no numeric value is found in the sequence name.
    """
    match = re.search(r"v(\d+)", sequence_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"No numeric value found in sequence name: {sequence_name}")


def sanitize_name(input_string, max_length=255):
    """
    Sanitize the input string to create a valid file- or pathname.

    - Replaces spaces with underscores.
    - Removes invalid characters: < > : " / \ | ? *
    - Truncates the name to the specified maximum length.

    Args:
        input_string (str): The input string to be sanitized.
        max_length (int, optional): The maximum length of the resulting name. Defaults to 255.

    Returns:
        str: The sanitized filename.
    """
    invalid_chars = r'<>:"/\|?*'
    sanitized_string = input_string.replace(" ", "_")
    sanitized_string = re.sub(f"[{re.escape(invalid_chars)}]", "", sanitized_string)
    return sanitized_string[:max_length]


def save_dicom(ds, output_path, base_name):
    """
    Save the DICOM dataset to the specified output path.

    Args:
        ds (pydicom.Dataset): The DICOM dataset to be saved.
        output_path (str): The output directory where the DICOM file will be saved.
        base_name (str): The base name of the DICOM file.

    Returns:
        None

    Raises:
        OSError: If an error occurs while creating the output directory or saving the DICOM file.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, base_name)
        ds.save_as(output_file)
    except OSError as e:
        logging.info(f"Error occurred while saving DICOM file: {str(e)}")
        raise


def validate_folder_structure(subfolder_root):
    """
    Ensure the folder structure is compliant and resolve ambiguity between "IN" or "THROUGH" folders.

    The function checks if the required folder MAG exists in the output root directory.
    If MAG is missing, an exception is raised.

    The function then checks for the presence of "IN" or "THROUGH" folders in the output root directory.
    If either of these folders is found and there is exactly one missing folder among AP, RL, or FH,
    the function renames the "IN" or "THROUGH" folder to the missing folder.

    Args:
        subfolder_root (str): The root directory containing the sub-folders.

    Returns:
        None

    Raises:
        FileExistsError: If the destination folder already exists during the renaming process.
        OSError: If an error occurs during the renaming process.
        Exception: If the folder structure is non-compliant and cannot be resolved.
    """

    existing_folders = set(os.listdir(subfolder_root))

    # Check if the required folder MAG exists
    if "MAG" not in existing_folders:
        raise Exception("Non-compliant folder structure. Missing required folder: MAG")

    rename_candidates = {"IN", "THROUGH"}
    required_folders = {"AP", "RL", "FH"}

    # Check if there are any rename candidates and missing required folders
    candidates_found = rename_candidates.intersection(existing_folders)
    missing_required_folders = required_folders - existing_folders
    if candidates_found and missing_required_folders:
        if len(candidates_found) == 1 and len(missing_required_folders) == 1:
            candidate = candidates_found.pop()
            missing_folder = missing_required_folders.pop()

            source_folder = os.path.join(subfolder_root, candidate)
            destination_folder = os.path.join(subfolder_root, missing_folder)

            try:
                os.rename(source_folder, destination_folder)
                logging.info(
                    f"{os.path.basename(subfolder_root)}: Renamed {candidate} to {missing_folder}"
                )
            except FileExistsError:
                raise Exception(
                    f"Destination folder {missing_folder} already exists. Skipping renaming."
                )
            except OSError as e:
                raise Exception(
                    f"Error occurred while renaming {candidate} to {missing_folder}: {str(e)}"
                )
        else:
            raise Exception(
                "Ambiguity detected: multiple candidates OR multiple missing required folders."
            )

    # Final check for compliant folder structure
    if missing_required_folders:
        missing_folders_str = ", ".join(missing_required_folders)
        raise Exception(
            f"Non-compliant folder structure. Missing folders: {missing_folders_str}"
        )


def get_output_path(ds, output_root):
    """
    Determine the output path for the DICOM dataset based on its image type.

    Magnitude images are typically stored as:
    - ORIGINAL/PRIMARY/M
    - ORIGINAL/PRIMARY/MAG
    - DERIVED/PRIMARY/M
    - DERIVED/PRIMARY/MAG_SUM

    Phase images are typically stored as:
    - ORIGINAL/PRIMARY/P
    - DERIVED/PRIMARY/P

    Args:
        ds (pydicom.Dataset): The DICOM dataset.
        output_root (str): The root directory for the output path.

    Returns:
        str: The output path for the DICOM dataset if a valid sub-folder is determined, or None otherwise.

    Raises:
        FlowDirectionNotFoundError: If a valid flow direction could not be determined for phase images.
    """
    sub_folder = None

    # TODO: This will currently catch speed images as MAG, need to find a better way to differentiate
    try:
        if ds.ImageType[1] == "PRIMARY" and ds.ImageType[2].startswith("M"):
            sub_folder = "MAG"
        elif (
            ds.ImageType[0] == "DERIVED"
            and ds.ImageType[1] == "PRIMARY"
            and ds.ImageType[2].startswith("M")
        ):
            sub_folder = "MAG"
        elif ds.ImageType[2] == "P":
            sub_folder = get_flow_direction(ds)

        if sub_folder:
            return os.path.join(output_root, sub_folder)
    except FlowDirectionNotFoundError:
        return None


def get_flow_direction(ds):
    """
    Determine the flow direction from the DICOM dataset.

    Args:
        ds (pydicom.Dataset): The DICOM dataset.

    Returns:
        str: The standardized flow direction (e.g., "AP", "RL", "FH") if found.

    Raises:
        FlowDirectionNotFoundError: If a valid flow direction could not be determined.
    """
    direction_mapping = {
        "AP": "AP",
        "PA": "AP",
        "RL": "RL",
        "LR": "RL",
        "FH": "FH",
        "HF": "FH",
    }

    nondescript_orientations = ["IN", "THROUGH"]

    tags_to_check = [
        (0x0018, 0x0024),  # Siemens AdvFlow
        (0x0021, 0x1129),  # Also Siemens AdvFlow(?)
        (0x0051, 0x1014),  # Northwestern/Greifswald(?)
    ]

    for tag in tags_to_check:
        try:
            value = ds[tag].value.upper()
            for suffix, direction in direction_mapping.items():
                if value.endswith(suffix):
                    return direction
            for suffix in nondescript_orientations:
                if value.endswith(suffix):
                    return suffix
        except KeyError:
            continue

    raise FlowDirectionNotFoundError(
        f"Could not determine flow direction for {ds.ProtocolName}"
    )


def is_valid_input_path(parser, arg, check_existence=True):
    """
    Check if the given path is valid.

    Args:
        parser (argparse.ArgumentParser): The ArgumentParser object.
        arg (str): The path to be validated.
        check_existence (bool, optional): Whether to check if the path exists. Defaults to True.

    Returns:
        str: The validated path if it is a valid format and exists (if check_existence is True).

    Raises:
        argparse.ArgumentTypeError: If the path format is invalid or if the path does not exist and check_existence is True.
    """
    if not arg:
        return arg

    if not isinstance(arg, str) or arg.strip() == "":
        parser.error(f"The input path {arg} is not a valid path string.")

    if check_existence and not os.path.exists(arg):
        parser.error(f"The input path {arg} does not exist.")

    return arg


if __name__ == "__main__":
    logging.info(r"""   ___   ____ _____ ____   __  ___ _             __            """)
    logging.info(r"""  / _ \ /  _// ___// __ \ /  |/  /(_)___  ___ _ / /_ ___   ____""")
    logging.info(r""" / // /_/ / / /__ / /_/ // /|_/ // // _ \/ _ `// __// _ \ / __/""")
    logging.info(r"""/____//___/ \___/ \____//_/  /_//_//_//_/\_,_/ \__/ \___//_/   """)
    logging.info(r"""                                                               """)

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=r"""Example: python dicominator.py /path/to/input /path/to/output --name "4D Flow" --nii --h5""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_root",
        type=lambda x: is_valid_input_path(parser, x),
        help="the root of the folder where DICOMs are stored",
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="the root of the folder where DICOMs will be stored [will be created if it does not exist]",
    )
    parser.add_argument(
        "-d",
        "--desc",
        type=str,
        help="optional 'SeriesDescription' to search for",
    )
    parser.add_argument(
        "-p",
        "--purge",
        action="store_true",
        help="purge the output folder before processing",
    )
    parser.add_argument("--nii", action="store_true", help="save as NIfTI")
    parser.add_argument("--h5", action="store_true", help="save as h5")
    parser.add_argument("--mat", action="store_true", help="save as mat")
    parser.add_argument(
        "--pcmra", action="store_true", help="save pcmra images in NIfTI format"
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list all unique Series Descriptions in the dataset without further processing",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force process files thay may or may not be flow datasets",
    )

    args = parser.parse_args()

    ignored_flags = [
        flag
        for flag, value in [
            ("--purge/-p", args.purge),
            ("--desc/-d", args.desc),
            ("--nii", args.nii),
            ("--h5", args.h5),
            ("--mat", args.mat),
            ("--pcmra", args.pcmra),
        ]
        if args.list and value
    ]

    if ignored_flags:
        ignored_flags_str = ", ".join(ignored_flags)
        logging.warning(
            f"The {ignored_flags_str} flag(s) will be ignored when only listing SeriesDescriptions."
        )

    if args.pcmra and not args.nii and not args.list:
        logging.warning(
            "The --pcmra flag currently requires the --nii flag to be set. The --nii flag will be set automatically."
        )
        args.nii = True

    if args.purge and args.output_root and not args.list:
        logging.info(f"Purging output folder {args.output_root}...")
        shutil.rmtree(args.output_root, ignore_errors=True)

    dicominator(
        args.input_root,
        output_root=args.output_root,
        description=args.desc,
        save_as_h5=args.h5,
        save_as_mat=args.mat,
        save_as_nii=args.nii,
        save_pcmra=args.pcmra,
        list_descriptions=args.list,
        force=args.force,
    )
