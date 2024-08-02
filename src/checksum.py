"""
This module contains functions to calculate the checksum of a file and to check if the checksums
in a list match the checksum of the corresponding files.
"""
import hashlib
from pathlib import Path
import tqdm

def checksum(file_path):
    """
    Calculate the checksum of a file.
    :param file_path: The Path object of the file.
    :return: The checksum of the file.
    """
    hash_object = hashlib.sha256()
    # Open the file in binary mode
    with open(file_path, "rb") as file:
        # Read the file in chunks
        for chunk in iter(lambda: file.read(4096), b""):
            # Update the hash object with each chunk
            hash_object.update(chunk)
    # Return the hexadecimal digest of the hash
    return hash_object.hexdigest()

def test_line_from_file(line: str, ignore_missing_files=False):
    """
    Check if the checksum in the line matches the checksum of the corresponding file.
    :param line: The line containing the checksum and the file name.
    :param ignore_missing_files: If True, ignore missing files. If False, print a message if a file is missing.
    :return: True if the checksum matches, else False.
    """
    # Split the line into file name and checksum
    checksum_value, check_path = line.strip().split("  ")
    # Create a Path object for the file
    check_path = Path(check_path)
    # Calculate the checksum of the file
    if check_path.exists():
        file_not_found = False
        file_checksum = checksum(check_path)
        result = file_checksum == checksum_value
    elif ignore_missing_files:
        file_not_found = False
        result = True
    else:
        file_not_found = True
        result = False

    if not result and not file_not_found:
        print(f"\nChecksum mismatch for file: {check_path}")
    elif file_not_found:
        print(f"\nFile not found: {check_path}")
    return result

def test_file_list_from_file(file_path: Path):
    """
    Check if the checksums in the list match the checksum of the corresponding files.
    :param file_path: The Path object of the file containing the checksums.
    :return: True if all checksums match, else False.
    :return: A list of tuples containing the file name and the result of the checksum comparison.
    """
    if not file_path.exists():
        return []
    results = []
    # Open the file in read mode
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()

    failed_files = 0

    for line in tqdm.tqdm(lines):
        result = test_line_from_file(line)
        if not result:
            failed_files += 1

    print("All files checked.")
    print(f"Failed: {failed_files}/{len(lines)}")
    return failed_files==0, results
