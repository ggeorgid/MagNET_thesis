import os
import gdown
import tarfile
from urllib.parse import urlparse, parse_qs

def download_dataset(url: str, save_dir: str, filename: str = None):
    """
    Downloads a file from Google Drive to the specified directory, extracts it if it's a .tar.gz file,
    and deletes the archive after extraction. Skips download and extraction if the output already exists.

    Args:
        url (str): The Google Drive shareable URL.
        save_dir (str): Directory where the file will be saved.
        filename (str): Optional filename for the downloaded file. Defaults to None for automatic naming.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract the file ID
    file_id = None
    if "drive.google.com" in url and "/d/" in url:
        # Standard Google Drive URL
        try:
            file_id = url.split("/d/")[1].split("/")[0]
        except IndexError:
            raise ValueError("Invalid Google Drive URL format.")
    elif "drive.usercontent.google.com" in url or "id=" in url:
        # Alternate URL format with 'id=' parameter
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        file_id = query_params.get("id", [None])[0]

    if not file_id:
        raise ValueError("Could not extract file ID from the provided URL.")

    # Construct the download URL
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Determine the output file path
    output_file = os.path.join(save_dir, filename) if filename else os.path.join(save_dir, f"file_{file_id}.tar.gz")

    # Pre-check: Determine if the contents of the archive already exist
    if tarfile.is_tarfile(output_file):
        with tarfile.open(output_file, "r:gz") as tar:
            all_exist = all(
                os.path.exists(os.path.join(save_dir, member.name)) for member in tar.getmembers()
            )
            if all_exist:
                print("[MagNet] All dataset files already exist. Skipping download and extraction.")
                return

    # Check if the file already exists (only for tar.gz downloads)
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}. Skipping download.")
    else:
        # Download the file
        print(f"Downloading from: {download_url}")
        gdown.download(download_url, output_file, quiet=False)
        print(f"File downloaded to: {output_file}")

    # Extract the file if it's a .tar.gz archive
    if tarfile.is_tarfile(output_file):
        try:
            with tarfile.open(output_file, "r:gz") as tar:
                all_exist = all(
                    os.path.exists(os.path.join(save_dir, member.name)) for member in tar.getmembers()
                )
                if all_exist:
                    print("[MagNet] Extracted files already exist. Skipping extraction.")
                else:
                    # Extract the files
                    tar.extractall(save_dir)
                    print("[MagNet] Dataset unzipped successfully.")
        except Exception as e:
            print(f"[MagNet] Error during extraction: {e}")
        finally:
            # Ensure the .tar.gz file is deleted
            try:
                os.remove(output_file)
                print("[MagNet] Cleanup successful. Deleted archive file.")
            except Exception as e:
                print(f"[MagNet] Error during cleanup: {e}")
    else:
        print("[MagNet] The downloaded file is not a .tar.gz archive. No extraction performed.")
