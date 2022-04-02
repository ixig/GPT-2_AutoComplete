import argparse
import os
from glob import glob


def argparser() -> argparse.Namespace:
    def dir_path(path: str) -> str:
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory path")

    parser = argparse.ArgumentParser(
        description="Combine Individual Text Files into Single (one-line) File ",
    )
    parser.add_argument("input_dir", type=dir_path, help="Input Files Directory")
    parser.add_argument("fout", type=argparse.FileType("w"), help="Output Combined Texts File")
    return parser.parse_args()


def main() -> int:
    args = argparser()

    files = glob(os.path.join(args.input_dir, "*.txt"))

    for file in files:
        with open(file) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    args.fout.write(line + " ")
    return 0


if __name__ == "__main__":
    exit(main())
