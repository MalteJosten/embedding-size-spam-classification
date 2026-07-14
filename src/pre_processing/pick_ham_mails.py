import os
import random
import sys
import shutil


def prepare_destination_directory(dest_dir):
    if os.path.exists(dest_dir):
        for file in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, file))
    else:
        os.makedirs(dest_dir)


def pick_files(directory, x, ignore=None):
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if ignore:
       all_files = [f for f in all_files if f not in ignore] 
        
    random_files = random.sample(all_files, x)

    print(f"Picked {x} random files from a total of {len(all_files)} files.")

    return random_files


def validate(test, train):
    same = [x for x in test if x in train]

    if len(same) != 0:
        print(f"Found {len(same)} duplicates in test and train selection. Aborting...")
        exit()
    else:
        print("No duplicates found. Continuing...")


def copy_mails(src_dir, dest_dir, picks):
    for e_id in picks:
       shutil.copyfile(os.path.join(src_dir, e_id), os.path.join(dest_dir, e_id)) 


def pick_ham():
    src_dir = sys.argv[1]
    number_train = int(sys.argv[2])
    number_test  = int(sys.argv[3])

    train_dir = "mails/ham/train"
    prepare_destination_directory(train_dir)
    train_picks = pick_files(src_dir, number_train)

    test_dir = "mails/ham/test"
    prepare_destination_directory(test_dir)
    test_picks = pick_files(src_dir, number_test, train_picks)

    validate(train_picks, test_picks)

    with open("mails/ham_mail_train.txt", "w") as mail_name_file:
        for name in train_picks:
            mail_name_file.write(f"{name}\n")

    with open("mails/ham_mail_test.txt", "w") as mail_name_file:
        for name in test_picks:
            mail_name_file.write(f"{name}\n")

    copy_mails(src_dir, train_dir, train_picks)
    copy_mails(src_dir, test_dir, test_picks)


def main():
    pick_ham()

if __name__ == "__main__":
    main()
