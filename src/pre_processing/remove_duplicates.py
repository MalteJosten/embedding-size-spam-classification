import argparse
import os
import time
import shutil
from util.progress_bar import progressBarTime

def remove_duplicates(src_dir, dst_dir):
    hashes = {}

    existing_files = os.listdir(dst_dir)
    files = os.listdir(src_dir)
    files = list(set(files) - set(existing_files))
    files.sort()

    dupe_count = 0

    start_time = time.time()

    for mail_name in progressBarTime(files, start_time, prefix="Progress", length=50):
        with open(os.path.join(src_dir, mail_name), "r") as mail:
            mail_content = mail.read()
            mail_body = "\n\n".join(mail_content.split("\n\n")[1:])
            mail_hash = hash(mail_body)

            if hashes.get(mail_hash, None):
                dupe_count += 1

            hashes[mail_hash] = 1

            try:
                shutil.copyfile(os.path.join(src_dir, mail_name), os.path.join(dst_dir, mail_name))
            except OSError:
                del hashes[mail_hash]
                print(mail_name)

    return dupe_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.dst):
        os.mkdir(args.dst)

    dupe_count = remove_duplicates(args.src, args.dst)

    print(f"Found and removed {dupe_count} duplicates")


if __name__ == "__main__":
    main()