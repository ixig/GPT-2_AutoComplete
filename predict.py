import argparse
import os
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore


class Colors:
    Endc = "\033[0m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    LightGray = "\033[37m"
    White = "\033[97m"


MODEL_CKPT = "distilgpt2-mlm"
# fmt: off
BANNED_TOKENS = [
    [12], [438], [532], [784], [851], [960], [1377], [11420],       # dashes
    [0], [1], [4], [6], [11], [13], [14], [25], [26],               # ! " % ' , / . : ;
    [338], [357], [366], [526], [553], [705], [720], [737], [828],  # 's ( " ." ," ' $ ). ),
    [1539], [1600], [1911], [2474], [2637], [7874], [14004]]        # ., ", ". !" .' .- ,''
# fmt: on
CTX_LEN = 128

LOOKBACK = 28
NEW_TOKENS = 1 + 2
NUM_PREDS = 10
NUM_BEAMS = NUM_PREDS
EXTRA_PREDS = 0


def argparser() -> argparse.Namespace:
    def dir_path(path: str) -> str:
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory path")

    parser = argparse.ArgumentParser(
        description="Evaluate performance of Text Auto-Completer on file",
    )
    parser.add_argument(
        "fin",
        type=argparse.FileType("r"),
        help="Input File for Evaluation",
    )
    parser.add_argument(
        "fout", type=argparse.FileType("w"), help="Output File with Predictions Annotated"
    )
    parser.add_argument(
        "-m", action="store_true", help="Annotate with Prediction Misses", dest="misses"
    )
    parser.add_argument(
        "-d",
        type=dir_path,
        default=MODEL_CKPT,
        help="Model Checkpoint Directory",
        dest="ckpt",
    )
    return parser.parse_args()


def main() -> int:
    args = argparser()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    config = AutoConfig.from_pretrained(
        args.ckpt, n_ctx=CTX_LEN, pad_token_id=tokenizer.eos_token_id
    )
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, config=config).to(device)

    fin = args.fin  # open(argv[1], "r")
    text_in = fin.read()
    fin.close()

    pat_mkr = re.compile(r"(\^+)")
    pat_mkr_end = re.compile(r"\^+$")
    pat_sep = re.compile(r"[\s-]+")
    pat_punct = re.compile(r"[()\".,;:!$%@–—-]")  # DON'T include "'"

    fout = args.fout  # open(argv[2], "w")
    # fout = sys.stdout

    text_in = text_in.replace("\n", "^")  # insert markers for NL ...
    text_in = pat_mkr.sub(r"\1 ", text_in)  # plus space for splitting
    words = pat_sep.split(text_in)
    LB_BUF = [""] * (LOOKBACK - 1) + [words[0]]
    print(LB_BUF[-1], end=" ", file=fout)
    print(LB_BUF[-1], end=" ", flush=True)

    top1_hits, top5_hits, top10_hits = 0, 0, 0

    for word in words[1:]:
        mkrs = pat_mkr_end.search(word)  # check for NL markers
        if mkrs:
            num_mkrs = len(mkrs.group())
            word = word[:-num_mkrs]
            sep = "\n" * num_mkrs
        else:
            sep = " "

        text_sep = " ".join(LB_BUF)
        text_sep = text_sep.replace("^", "")
        input_ids = tokenizer(text_sep, return_tensors="pt")["input_ids"].to(device)

        # Generate first/starter token
        output = model.generate(
            input_ids,
            max_new_tokens=1,
            min_length=len(input_ids),
            num_return_sequences=NUM_PREDS + EXTRA_PREDS,
            num_beams=NUM_BEAMS + EXTRA_PREDS,
            do_sample=False,
            bad_words_ids=BANNED_TOKENS,
        )

        # Filter out unsuitable starter tokens
        next_first = []
        for i in range(output.shape[0]):
            next_first.append((i, tokenizer.decode(output[i, len(input_ids[0])]).strip()))
        # tmp = [(idx, word) for idx, word in next_first if pat_punct.match(word)]
        # if tmp: print(tmp[0][1], output[tmp[0][0], len(input_ids[0])])
        next_filt = [idx for idx, word in next_first if not pat_punct.match(word)]
        output = output[next_filt]  # remove filtered tokens

        # Generate subsequent tokens (for multi-token words)
        output = model.generate(
            output,
            max_new_tokens=NEW_TOKENS - 1,
            num_return_sequences=1,
            num_beams=1,
            do_sample=False,
        )

        # Split off hypenated words and truncate punctuations
        next_words = []
        for i in range(output.shape[0]):
            next_words.append(tokenizer.decode(output[i, len(input_ids[0]) :]).strip())
        next_words = [pat_sep.split(word)[0] for word in next_words if word]
        next_words = [word.rstrip("\"'.,;:!$%@–—-") for word in next_words]
        next_words = [word for word in next_words if word]
        next_words = next_words[:NUM_PREDS]  # limit to NUM_PREDS if EXTRA_PREDS > 0

        # Check for prediction hits
        word_nopunct = pat_punct.sub("", word)  # remove beginning/ending punctuations
        if word_nopunct in next_words:
            idx = next_words.index(word_nopunct)
            if idx == 0:
                attr = Colors.LightGreen  # Top-1
                top1_hits += 1
            elif idx < 5:
                attr = Colors.LightCyan  # Top-5
                top5_hits += 1
            else:
                attr = Colors.LightYellow  # Top-10
                top10_hits += 1
            s = f"{attr}{word}{Colors.Endc}{sep}"
            print(s, end="", file=fout)
            print(s, end="", flush=True)
        else:
            if args.misses:
                s = f"{Colors.LightRed}{'|'.join(next_words)}═{Colors.Endc}"
                print(s, end="", file=fout)
                print(s, end="", flush=True)
            print(word + sep, end="", file=fout)
            print(word + sep, end="", flush=True)
        LB_BUF = LB_BUF[1:] + [word]

    total_hits = top1_hits + top5_hits + top10_hits
    s = (
        f"\nLKBACK: {LOOKBACK}, PREDS: {NUM_PREDS}, EXTRA_PREDS: {EXTRA_PREDS}, NEW_TOKS: {NEW_TOKENS}"
        f" => #HITS: {top1_hits},{top5_hits},{top10_hits} / {len(words)} ({100*total_hits/len(words):.1f})%"
    )
    print(s, file=fout)
    print(s)

    fout.close()
    return 0


if __name__ == "__main__":
    exit(main())
