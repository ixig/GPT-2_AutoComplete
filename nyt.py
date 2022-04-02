import argparse
import os
import sys
from glob import glob
from time import sleep
from typing import Any

import requests
from bs4 import BeautifulSoup

SAVE_FILENAME_MAXLEN = 60
API_DELAY = 5

CLASS_TITLE = "css-fwqvlz"
CLASS_PARAGH = "css-g5piaz evys1bk0"
FACET = "Russian Invasion of Ukraine (2022)"
API_KEY = ""
BASE_URL_V2 = "https://api.nytimes.com/svc/{}/v2/"
BASE_URL_V3 = "https://api.nytimes.com/svc/{}/v3/"

key: str


def argparser() -> argparse.Namespace:
    def dir_path(path: str) -> str:
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory path")

    parser = argparse.ArgumentParser(
        description="Scrape NYT News Articles using NYT Developer APIs",
        epilog=(
            "The API Key can be provided by setting the environmental variable NYT_API_KEY, "
            "or by overriding using the -k command argument"
        ),
    )
    parser.add_argument("save_dir", type=dir_path, help="Saved Articles Directory")
    try:
        api_key = os.environ["NYT_API_KEY"]
    except:
        api_key = API_KEY
    parser.add_argument(
        "-k",
        type=str,
        default=api_key,
        help="NYT Developer API Key",
        dest="api_key",
    )
    return parser.parse_args()


def get(api: str, json: str, q: str = "") -> Any:
    query = q and ("?q=" + q)
    api_key = "&api-key=" if q else "?api-key="
    base_url = BASE_URL_V3 if api == "news" else BASE_URL_V2
    url = base_url.format(api) + json + ".json" + query
    response = requests.get(url + api_key + key)
    assert response.status_code == 200
    # print(url, response.status_code)
    # print(response.json().keys())
    return response.json()


def main() -> int:
    args = argparser()
    global key
    key = args.api_key

    existing_files = glob(os.path.join(args.save_dir, "*.txt"))
    existing_files = [os.path.basename(path) for path in existing_files]

    resp = get("topstories", "world")
    assert resp["status"] == "OK"

    hits = [x for x in resp["results"] if FACET in x["des_facet"]]
    hits.sort(key=lambda x: x["title"])  # type: ignore
    titles_urls = [(hit["title"], hit["url"]) for hit in hits]

    for title, url in titles_urls:
        if title[:SAVE_FILENAME_MAXLEN] + ".txt" in existing_files:
            print(f"SKIPPING: {title}")
            continue
        else:
            print(f"DOWNLOADING: {title}")

        resp = requests.get(url)
        assert resp.status_code == 200

        soup = BeautifulSoup(resp.content, "html.parser")
        body = soup.find("body")
        title = body.find(class_=CLASS_TITLE).text
        paragraphs = body.find_all(class_=CLASS_PARAGH)
        texts = [x.text for x in paragraphs]
        text = "\n\n".join(texts)

        with open(os.path.join(args.save_dir, title[:SAVE_FILENAME_MAXLEN] + ".txt"), "w") as f:
            f.write(title + "\n\n")
            f.write(text)

        sleep(API_DELAY)

    return 0


if __name__ == "__main__":
    exit(main())
