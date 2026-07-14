import argparse
import os
import random
import shutil


def prepare_destination_directory(dest_dir="mails/spam/test"):
    if os.path.exists(dest_dir):
        for file in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, file))
    else:
        os.makedirs(dest_dir)


def pick_files(directory, x):
    all_files = os.listdir(directory)
    random_files = random.sample(all_files, x)

    print(f"Picked {x} random files from a total of {len(all_files)} files.")

    return random_files


def copy_mails(source_dir, picks):
    for e_id in picks:
       shutil.copyfile(os.path.join(source_dir, e_id), os.path.join("mails/spam/test", e_id)) 


def pick_spam(src_dir, count, name_file):
    prepare_destination_directory()
    
    picks = pick_files(src_dir, count)

    with open(name_file, "w") as mail_name_file:
        for name in picks:
            mail_name_file.write(f"{name}\n")

    copy_mails(src_dir, picks)

    print("Copied files.")


def main():
    parser = argparse.ArgumentParser(prog='Spam Mail Picker')
    parser.add_argument("--src", required=True, help="Source directory")
    parser.add_argument("--count", required=True, type=int, help="Mail count")
    parser.add_argument("--file", required=True, help="Mail name file")
    args = parser.parse_args()

    pick_spam(args.src, args.count, args.file)

if __name__ == "__main__":
    main()