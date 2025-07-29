import os
import gdown
import tarfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

def extract_drive_file_id(url):
    """Extracts the Google Drive file ID from various URL formats."""
    parsed_url = urlparse(url)
    if "drive.google.com" in url and "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    if "id=" in parsed_url.query:
        return parse_qs(parsed_url.query).get("id", [None])[0]
    return None

def dataset_already_downloaded(output_file, save_dir):
    """Checks if the dataset archive or extracted files already exist."""
    if not Path(output_file).exists():  
        return False
    if tarfile.is_tarfile(output_file):  
        with tarfile.open(output_file, "r:gz") as tar:
            return all(Path(save_dir, member.name).exists() for member in tar.getmembers())
    return Path(output_file).exists()

def download_dataset(url: str, save_dir: str, filename: str = None):
    """
    Downloads a file from Google Drive to the specified directory, extracts it if it's a .tar.gz file,
    and deletes the archive after extraction. Skips download and extraction if the output already exists.

    Args:
        url (str): The Google Drive shareable URL.
        save_dir (str): Directory where the file will be saved.
        filename (str): Optional filename for the downloaded file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("[ERROR] Invalid Google Drive URL.")

    output_file = save_dir / (filename if filename else f"file_{file_id}.tar.gz")

    if dataset_already_downloaded(output_file, save_dir):
        print("[INFO] Dataset already exists. Skipping download and extraction.")
        return

    # Download the dataset
    print(f"[INFO] Downloading dataset from: {url}")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output_file), quiet=False)

    # Extract if it's a .tar.gz file
    if tarfile.is_tarfile(output_file):
        try:
            with tarfile.open(output_file, "r:gz") as tar:
                tar.extractall(save_dir)
                print("[INFO] Dataset successfully extracted.")
        except tarfile.TarError as e:
            print(f"[ERROR] Failed to extract dataset: {e}")
            return  

        # Cleanup: Delete the archive file
        try:
            output_file.unlink()
            print("[INFO] Deleted archive file after extraction.")
        except Exception as e:
            print(f"[WARNING] Failed to delete archive: {e}")
    else:
        print("[WARNING] Downloaded file is not a .tar.gz archive. No extraction performed.")
