import os
import sys

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) >= 1 else os.getcwd()
    total_files = count_files(path)
    print(f"Total files (excluding folders): {total_files}")
