import time
import os
import json
import numpy as np
import argparse
from util.progress_bar import progressBarTime


def determine_mail_lengths(mail_dir):
    mail_lengths = {}
    start_time = time.time()

    for mail_name in progressBarTime(os.listdir(mail_dir), start_time=start_time, suffix="", length=50):
        with open(os.path.join(mail_dir, mail_name), "r") as mail_file:
            split_mail = mail_file.read().split("\n\n")
            mail_body = "\n\n".join(split_mail[1:])

            char_count = len(mail_body)
            mail_lengths[char_count] = mail_lengths.get(char_count, 0) + 1

    result = {}
    result["lengths"] = mail_lengths
    lengths = list(mail_lengths.keys())
    result["min"] = int(np.min(lengths))
    result["max"] = int(np.max(lengths))
    result["avg"] = int(np.average(lengths))
    result["med"] = int(np.median(lengths))
    result["quantiles"] = [round(float(x), 2) for x in np.percentile(lengths, [20, 40, 60, 80])]

    return result


def write_result(result, file_name):
    if not os.path.exists("output/mail_length"):
        os.makedirs("output/mail_length")
    
    with open(f"{file_name}.json", "w") as res_file:
        res_file.write(json.dumps(result, indent=2))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", type=str, required=True)
    argparser.add_argument("--label", type=str, required=True)

    args = argparser.parse_args()

    file_name = f"output/mail_length/{args.label}"
    result = determine_mail_lengths(args.dir)

    write_result(result, file_name)


if __name__ == "__main__":
    main()