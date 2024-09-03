# DICOMinator

DICOMinator is an open-source Python script for sorting images from common 4D Flow pulse sequences.

## Features :sparkles:

- :open_file_folder: Recursively traverses the input directory to find all DICOM files.
- :open_file_folder: Sorts the DICOM files into a specified output directory based on encoding direction.
- :floppy_disk: Provides options to save the sorted data in various formats:
    - :package:	NIfTI: The NIfTI files are split along the time dimension, allowing them to be imported as a Sequence (or MultiVolume) in [3D Slicer](https://www.slicer.org/).
    - :package:	h5: The h5 format should be compatible with Edward Ferdian's [4DFlowNet](https://github.com/edwardferdian/4DFlowNet).
    - :package: mat: The mat format should be compatible with Julio Sotelo's [4D-Flow-Matlab-Toolbox](https://github.com/JulioSoteloParraguez/4D-Flow-Matlab-Toolbox).


## Installation :hammer_and_wrench:

1. Clone the repository:
    ```sh
    git clone https://github.com/fyrdahl/dicominator.git
    cd dicominator
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Prerequisites :clipboard:

- Python 3.6 or higher
- Required Python libraries (listed in `requirements.txt`)

## Usage :rocket:

### Arguments

- `input_root`: The root folder where DICOMs are stored.
- `output_root`: The root folder where DICOMs will be stored (will be created if it does not exist).
- `-d`, `--desc`: Optional 'SeriesDescription' to search for.
- `-p`, `--purge`: Purge the output folder before processing.
- `--nii`: Save as NIfTI.
- `--h5`: Save as h5.
- `--mat`: Save as mat.
- `--pcmra`: Save PCMRA image in NIfTI format
- `-l`, `--list`: List all unique Series Descriptions in the dataset without further processing.
- `-f`, `--force`: Force process files thay may or may not be flow datasets.

### Example

```sh
python dicominator.py /path/to/input /path/to/output --desc "4D Flow" --nii --h5 --mat
```

## Contact :mailbox:

For support, please contact alexander.fyrdahl@ki.se or open an issue.

## Acknowledgments :pray:

- Edward Ferdian for [4DFlowNet](https://github.com/edwardferdian/4DFlowNet)
- Julio Sotelo for [4D-Flow-Matlab-Toolbox](https://github.com/JulioSoteloParraguez/4D-Flow-Matlab-Toolbox)
- Pia Callmer for contributing code

## Contributing :handshake:

Contributions are welcome! Is your 4D Flow sequence not supported? Please feel free to open an issue or submit a pull request.

## Code Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for code formatting.

## License :balance_scale:

This project is licensed under the [MIT License](LICENSE).
