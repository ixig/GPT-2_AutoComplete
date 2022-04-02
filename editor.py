import argparse
import curses
import os
import re
from typing import Any, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type:ignore

COLOR_BRIGHT = 8
EDITOR_ROWS = 28
DEBUG_ROWS = 4
NUM_PREDS = 10
DEBUGGING = False

MODEL_CKPT = "distilgpt2-mlm"

TEXT = """After a Battering, Ukraine Seeks Momentum

ODESSA, Ukraine - Ukrainian forces carried out counter-offensives against Russian positions on \
Wednesday, seeking to inflict what one official called "maximum losses," even as the invading \
Russian military stepped up its lethal attacks on cities.

In Mariupol, an airstrike destroyed a theater where about 1,000 people had taken shelter, \
according to city and regional administrators, and photos and videos posted online showed \
the burning wreckage of the building.
"""


class WinDebug:
    def __init__(self, h: int, w: int, y: int, x: int) -> None:
        if not DEBUGGING:
            return
        self.win = curses.newwin(h, w, y, x)

    def __call__(self, s: Any) -> None:
        if not DEBUGGING:
            return
        if not isinstance(s, str):
            s = str(s)
        self.win.clear()
        self.win.addstr(s[: 80 * DEBUG_ROWS - 1])
        self.win.refresh()


class WinPred:
    NUM_PREDS = 10
    COL_OFFS = (0, 16, 32, 48, 64)
    CHOICES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

    def __init__(
        self, h: int, w: int, y: int, x: int, a_choice: int, a_pred: int, a_disabled: int
    ) -> None:
        self.win = curses.newwin(h, w, y, x)
        self.disabled = False
        self.preds = [""] * self.NUM_PREDS
        self.a_choice = a_choice
        self.a_pred = a_pred
        self.a_disabled = a_disabled

    def show_preds(self, preds: Optional[List[str]]) -> None:
        self.win.clear()
        if preds is not None:
            assert len(preds) == self.NUM_PREDS
            self.preds = preds
        for i, pred in enumerate(self.preds):
            y = i // (self.NUM_PREDS // 2)
            x = self.COL_OFFS[i % (self.NUM_PREDS // 2)]
            if self.disabled:
                self.win.addstr(y, x, f"{self.CHOICES[i]}:", self.a_disabled)
                self.win.addstr(y, x + 2, f"---", self.a_pred)
            else:
                self.win.addstr(y, x, f"{self.CHOICES[i]}:", self.a_choice)
                self.win.addstr(y, x + 2, f"{pred[:12]}", self.a_pred)
        self.win.refresh()

    def disable(self, disable: bool) -> None:
        self.disabled = disable
        self.show_preds(None)


class Editor:
    PAD_MAX_LINES = 200
    LINES_LOOKBACK = 5

    def __init__(self, h: int, w: int, win_y: int, win_x: int) -> None:
        self.pad = curses.newpad(self.PAD_MAX_LINES, 80)
        self.h = h
        self.w = w
        self.win_y = win_y
        self.win_x = win_x
        self.cy, self.cx = 0, 0
        self.h_off = 0
        self.pat_spc = re.compile(r"\s+")
        self.pat_blank = re.compile(r"^\s*$")
        self.pat_parag = re.compile(r"\s{2,}$")

    def refresh(self) -> None:
        self.pad.refresh(
            0 if self.cy < self.h else (self.cy - self.h + 1),
            0,
            self.win_y,
            self.win_x,
            self.win_y + self.h,
            self.win_x + self.w,
        )

    def update_cursor(self, n: int) -> None:
        if self.cx + n < 0:
            self.cy -= 1 + max(0, (n // self.w - 1))
        if self.cx + n >= self.w:
            self.cy += 1 + max(0, (n // self.w - 1))
        self.cx = (self.cx + n) % self.w

    def refresh_cursor(self) -> None:
        self.pad.move(self.cy, self.cx)
        self.refresh()

    def get_text(self) -> str:
        start_y = max(0, self.cy - self.LINES_LOOKBACK)
        buf = [
            self.pad.instr(row, 0, self.w).decode("utf-8") for row in range(start_y, self.cy + 1)
        ]
        buf = ["" if self.pat_blank.match(row) else row for row in buf]
        text = self.pat_spc.sub(" ", "".join(buf))
        return text

    def get_buf(self) -> str:
        buf = [self.pad.instr(row, 0, self.w).decode("utf-8") for row in range(self.cy + 1)]
        buf = ["\n" if self.pat_blank.match(row) else row for row in buf]
        buf = [(row.rstrip() + "\n") if self.pat_parag.search(row) else row for row in buf]
        text = "".join(buf)
        return text

    def addstr(self, s: str, attr: int = curses.A_NORMAL) -> None:
        self.pad.addstr(self.cy, self.cx, s, attr)
        self.cy, self.cx = self.pad.getyx()
        self.refresh()

    def addch(self, ch: str) -> None:
        self.pad.addch(self.cy, self.cx, ch)
        self.update_cursor(1)
        self.refresh()

    def backspace(self) -> None:
        if self.cy == self.cx == 0:
            return
        self.pad.delch(self.cy, self.cx)
        self.update_cursor(-1)
        self.pad.move(self.cy, self.cx)
        self.pad.clrtoeol()
        self.refresh()

    def newline(self) -> None:
        self.pad.clrtoeol()
        self.cx = 0
        self.cy += 1
        self.pad.move(self.cy, self.cx)
        self.pad.clrtoeol()
        self.refresh()

    def cmdkey(self, key: Union[int, str]) -> None:
        if key == curses.KEY_BACKSPACE or key == "\x7f" or key == "\b":
            # https://stackoverflow.com/a/54303430
            self.backspace()
        elif key == "\n":
            self.newline()


class Predictor:
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

    def __init__(self, checkpoint: str) -> None:
        self.pat_mkr = re.compile(r"(\^+)")
        self.pat_mkr_end = re.compile(r"\^+$")
        self.pat_sep = re.compile(r"[\s-]+")
        self.pat_punct = re.compile(r"[()\".,;:!$%@–—-]")  # DON'T include "'"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(
            checkpoint,
            n_ctx=self.CTX_LEN,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, config=self.config).to(
            self.device
        )

    def predict(self, text: str) -> List[str]:
        text_sep = " ".join(self.pat_sep.split(text)[-self.LOOKBACK :])
        self.text_sep = text_sep
        input_ids = self.tokenizer(text_sep, return_tensors="pt")["input_ids"].to(self.device)

        # Generate first/starter token
        output = self.model.generate(
            input_ids,
            max_new_tokens=1,
            min_length=len(input_ids),
            num_return_sequences=self.NUM_PREDS + self.EXTRA_PREDS,
            num_beams=self.NUM_BEAMS + self.EXTRA_PREDS,
            do_sample=False,
            bad_words_ids=self.BANNED_TOKENS,
        )

        # Filter out unsuitable starter tokens
        next_first = []
        for i in range(output.shape[0]):
            next_first.append((i, self.tokenizer.decode(output[i, len(input_ids[0])]).strip()))
        # tmp = [(idx, word) for idx, word in next_first if pat_punct.match(word)]
        # if tmp: print(tmp[0][1], output[tmp[0][0], len(input_ids[0])])
        next_filt = [idx for idx, word in next_first if not self.pat_punct.match(word)]
        output = output[next_filt]  # remove filtered tokens

        # Generate subsequent tokens (for multi-token words)
        output = self.model.generate(
            output,
            max_new_tokens=self.NEW_TOKENS - 1,
            num_return_sequences=1,
            num_beams=1,
            do_sample=False,
        )

        # Split off hypenated words and truncate punctuations
        next_words = []
        for i in range(output.shape[0]):
            next_words.append(self.tokenizer.decode(output[i, len(input_ids[0]) :]).strip())
        next_words = [self.pat_sep.split(word)[0] for word in next_words if word]
        next_words = [word.rstrip("\"'.,;:!$%@–—-") for word in next_words]
        next_words = [word for word in next_words if word]
        next_words = next_words[: self.NUM_PREDS]  # limit to NUM_PREDS if EXTRA_PREDS > 0
        if len(next_words) < self.NUM_PREDS:  # pad to NUM_PREDS if too few
            next_words += [""] * (self.NUM_PREDS - len(next_words))

        return next_words[: self.NUM_PREDS]


class KeyParser:
    def __init__(
        self,
        predictor: Predictor,
        win_pred: WinPred,
        win_editor: Editor,
        win_debug: WinDebug,
        a_top1: int,
        a_top5: int,
        a_top10: int,
    ) -> None:
        self.predictor = predictor
        self.win_pred = win_pred
        self.editor = win_editor
        self.debug = win_debug
        self.a_top1 = a_top1
        self.a_top5 = a_top5
        self.a_top10 = a_top10
        self.preds = [""] * NUM_PREDS
        self.prev_key = ""
        self.escaped = False
        self.pat_punct = re.compile(r"[.,!]")

        self.win_pred.show_preds(self.preds)
        self.editor.refresh_cursor()

    def parse(self, key: str) -> bool:
        if len(key) == 1 and (key.isprintable() or key == "\n"):
            if key in [" ", "-", "\n"] or (key.isdigit() and not self.escaped):
                if key in [" ", "-"]:
                    self.editor.addch(key)
                elif key == "\n":
                    self.editor.cmdkey(key)
                else:
                    idx = (int(key) - 1) % 10
                    if idx == 0:
                        attr = self.a_top1
                    elif idx < 5:
                        attr = self.a_top5
                    else:
                        attr = self.a_top10
                    self.editor.addstr(self.preds[idx] + " ", attr)
                text = self.editor.get_text()[:-1]  # don't include trailing space
                self.preds = self.predictor.predict(text)
                self.win_pred.show_preds(self.preds)
                self.debug(">" + self.predictor.text_sep + "<")
                self.editor.refresh_cursor()
            elif self.pat_punct.match(key) and self.prev_key.isdigit() and not self.escaped:
                self.editor.cmdkey("\x7f")  # BACKSPACE
                self.editor.addch(key)
            else:
                self.editor.addch(key)
        else:  # non-printable key
            if key == "\x1b":  # ESC
                self.escaped = not self.escaped
                self.win_pred.disable(self.escaped)
                self.editor.refresh_cursor()
            if key == "\x04":  # <CTRL>+D
                return False
            else:
                self.editor.cmdkey(key)

        self.prev_key = key
        return True


def main(scr: curses.window, args: argparse.Namespace) -> int:
    scr.clear()
    scr.refresh()

    scr_max_y, scr_max_x = scr.getmaxyx()
    assert scr_max_x >= 80 and scr_max_y >= 22
    assert curses.has_colors()

    curses.set_escdelay(200)

    curses.init_color(0, 0, 0, 0)
    curses.init_pair(1, COLOR_BRIGHT + curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, COLOR_BRIGHT + curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, COLOR_BRIGHT + curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, COLOR_BRIGHT + curses.COLOR_CYAN, curses.COLOR_BLACK)
    FG_GRN = curses.color_pair(1)
    FG_RED = curses.color_pair(2)
    FG_YEL = curses.color_pair(3)
    FG_CYN = curses.color_pair(4)

    win_editor = Editor(EDITOR_ROWS, 80, 0, 0)
    win_editor.addstr(TEXT)

    win_pred = WinPred(2, 80, EDITOR_ROWS + 1, 0, a_choice=FG_YEL, a_pred=FG_CYN, a_disabled=FG_RED)

    win_debug = WinDebug(DEBUG_ROWS, 80, EDITOR_ROWS + 4, 0)

    predictor = Predictor(args.ckpt)

    key_parser = KeyParser(
        predictor, win_pred, win_editor, win_debug, a_top1=FG_GRN, a_top5=FG_CYN, a_top10=FG_YEL
    )

    key = scr.getkey()
    while key_parser.parse(key):
        key = scr.getkey()

    args.fout.write(win_editor.get_buf())
    return 0


def argparser() -> argparse.Namespace:
    def dir_path(path: str) -> str:
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory path")

    parser = argparse.ArgumentParser(
        description="Text Auto-Completer Editor",
    )
    parser.add_argument("fout", type=argparse.FileType("w"), help="Output File")
    parser.add_argument(
        "-d",
        type=dir_path,
        default=MODEL_CKPT,
        help="Model Checkpoint Directory",
        dest="ckpt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    curses.wrapper(main, args)
