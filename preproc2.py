import argparse
import os
import re
from glob import glob

import spacy

# fmt: off
# Notable NER:Persons that should not be converted to "XXX"
PERSONS_WHITELIST = [ # fmt: off
    'Vladimir V. Putin', 'Vladimir Putin', 'Putin',
    'Volodymyr Zelensky', 'Volodymyr Zelenskyy', 'Zelensky',
    'Biden',
    'Xi Jinping', 'Xi',
    'Olaf Scholz', 'Scholz',
    'Boris Johnson', 'Johnson',
    'Emmanuel Macron', 'Macron',
    'Sergey V. Lavrov', 'Lavrov',
    'Aleksei A. Navalny', 'Navalny',
    'Jens Stoltenberg', 'Stoltenberg',
    'Antony J. Blinken', 'Blinken',
    'Mark A. Milley', 'Milley',
    'Kamala Harris', 'Harris',
    'Barack Obama', 'Obama',
    'Donald J. Trump', 'Trump',
    ### Misclassified as PERSON (by Spacy) ###
    'Stinger', 'Javelin', 'Brexit', 'C.I.A.', 'Twitter', 'Mykolaiv'
]

# Notable NER:Places that should not be converted to "YYY"
PLACES_WHITELIST = [ # fmt: off
    'U.S.', 'United States', 'Washington',
    'Russia', 'Soviet Union', 'Moscow', 'Crimea', 'Belarus', 'Chechnya',
    'Ukraine', 'Kyiv', 'Kharkiv', 'Lviv', 'Kherson', 'Odessa', 'Mariupol', 'Donetsk', 'Irpin', 'Mykolaiv',
    'China', 'Beijing',
    'Germany', 'Berlin',
    'U.K.', 'Britain', 'London',
    'France', 'Paris',
    'Poland', 'Warsaw',
    'Brussels', 'Netherlands',
    'Lithuania', 'Romania', 'Latvia', 'Estonia', 'Moldova', 'Slovakia',
    'Canada', 'China', 'Israel', 'Syria', 'Afghanistan', 'Iran', 'Iraq', 'North Korea',
    ### Misclassified as PLACE (my Spacy) ###
    'Ukrainian'
]
# fmt: on


def argparser() -> argparse.Namespace:
    def dir_path(path: str) -> str:
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory path")

    parser = argparse.ArgumentParser(
        description="Preprocess Text Files - Step #2",
    )
    parser.add_argument("input_dir", type=dir_path, help="Input Files Directory")
    parser.add_argument("output_dir", type=dir_path, help="Output Files Directory")
    args = parser.parse_args()

    if args.input_dir == args.output_dir:
        print(f"<input_dir> and <output_dir> must be different!")
        exit(1)

    return args


def main() -> int:
    def process_file(text: str) -> str:
        doc = nlp(text)
        for ent in reversed(doc.ents):
            if ent.label == 380:  # persons
                ent_text = ent.text
                if m := pa_apostr.match(ent_text):  # e.g. "Biden's"
                    ent_text = m.group(1)
                if ent_text not in PERSONS_WHITELIST:
                    # Normalize non-significant Persons to "XXX"
                    text = text[: ent.start_char] + "XXX" + text[ent.end_char :]

            if ent.label == 384:  # places
                ent_text = ent.text.lower()
                if m := pa_the.match(ent_text):  # e.g. "the Soviet Union"
                    ent_text = m.group(1)
                if ent_text not in places_whitelist:
                    # Normalize non-significant Places to "YYY"
                    text = text[: ent.start_char] + "YYY" + text[ent.end_char :]

        text = text.replace(
            "XXX XXX", "XXX"
        )  # incomplete NER parsing (e.g. "Ursula von der Leyen")
        return text

    args = argparser()

    files = glob(os.path.join(args.input_dir, "*.txt"))

    nlp = spacy.load("en_core_web_md")
    places_whitelist = [place.lower() for place in PLACES_WHITELIST]

    pa_apostr = re.compile(r"(.+)'s")  # Spacy doesn't strip off "'s" when returning entities
    pa_the = re.compile(r"the (.+)")  # Spacy includes "the" (e.g. "the U.S.") with entities

    for file in files:
        with open(file) as fin:
            fname = os.path.basename(file)
            with open(os.path.join(args.output_dir, fname), "w") as fout:
                print(".", end="", flush=True)
                text_in = fin.read()
                text_out = process_file(text_in)
                fout.write(text_out)
    print()
    return 0


if __name__ == "__main__":
    exit(main())
