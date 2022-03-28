import argparse
import os
import re
from glob import glob


def argparser():
    def dir_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory path")

    parser = argparse.ArgumentParser(
        description="Preprocess Text Files - Step #1",
    )
    parser.add_argument("input_dir", type=dir_path, help="Input Files Directory")
    parser.add_argument("output_dir", type=dir_path, help="Output Files Directory")
    args = parser.parse_args()

    if args.input_dir == args.output_dir:
        print(f"<input_dir> and <output_dir> must be different!")
        exit(1)

    return args


def main():
    args = argparser()

    files = glob(os.path.join(args.input_dir, "*.txt"))

    pat1 = re.compile(r"for The New York Times")  # filter image captions
    pat2 = re.compile(r"\s{2,}")  # filter consecutive white space
    pat3 = re.compile(r"[”“″]|(’’)")  # normalize starting/ending double-quotes
    pat4 = re.compile(r"[‘’]")  # normalize starting/ending single-quotes
    pat5 = re.compile(r"[—–]")  # normalize em/en-dashes

    for file in files:
        with open(file) as fin:
            fname = os.path.basename(file)
            with open(os.path.join(args.output_dir, fname), "w") as fout:
                print(".", end="", flush=True)
                for i, line in enumerate(fin):
                    assert pat1.search(line) is None, f"{fname}:{i+1}: {line}"
                    line = line.strip()
                    if not line:
                        continue
                    line = pat2.sub(" ", line)  # adjacent whitespaces
                    line = pat3.sub('"', line)  # double-quotes normalization
                    line = pat4.sub("'", line)  # single-quotes normalization
                    line = pat5.sub("-", line)  # dash normalization
                    fout.write(line + "\n")
    print()
    return 0


if __name__ == "__main__":
    exit(main())
