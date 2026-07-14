import json
import matplotlib.pyplot as plt
import argparse


def load_result_from_file(file_name):
    with open(f"{file_name}.json", "r") as res_file:
        results = json.loads(res_file.read())

    return results


def plot_histogram(result, scale, file_name):
    x = list(result["lengths"].keys())
    y = list(result["lengths"].values())

    plt.bar(x, y, edgecolor="black")
    if scale == "log":
        plt.xscale("log")
    elif scale == "lin":
        plt.xscale("linear")
    plt.xlabel("Frequency")
    plt.ylabel("Character Count")
    plt.title(f"Mail Body Lengths ({"_".join(file_name.split("_")[2:])})")

    plt.savefig(f"{file_name}-{scale}.png")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--label", type=str, required=True)
    argparser.add_argument("--scale", type=str)

    args = argparser.parse_args()

    file_name = f"output/mail_length/{args.label}"
    result = load_result_from_file(file_name)

    if not args.scale:
        print("ERROR: Provide --scale: 'log' or 'lin'")
        exit()

    plot_histogram(result, args.scale, file_name)


if __name__ == "__main__":
    main()