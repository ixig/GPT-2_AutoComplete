![](assets/Banner.jpg)

# Context Fine-Tuned, GPT-2 Auto-Completion Text Editor
## Building a Better Text-completion Prediction Engine with Transformers

<p>&nbsp;</p>

Five journalists have lost their lives in the course of performing their work in just the first month of the 2022 Russia-Ukraine War.

<p align="center">
<img src="assets/UkraineJournalistsKilled.jpg" alt="5 journalists killed in 1 month" width=500>
</p>

The Internet is awash with examples of auto-completion fails. In particular, the next-word auto-completion suggestions offered by the generic virtual keyboard of a smartphone or tablet remains woefully bad even for casual conversational text, and even more so for writing articles in more specialized domains, such as international journalism or technical blogging.

<p align="center">
<img src="assets/HowDoIConvertTo.png" alt="How do I convert to ...?" width=500>
</p>

---

I fine-tuned a Hugging Face [DistilGPT2](https://huggingface.co/distilgpt2) Language Model on a corpus consisting of just *one week* of news articles from the NYT and AP News at the start of the war. The news articles were [scraped](https://github.com/ixig/GPT-2_AutoComplete/blob/main/nyt.py) using the NYT Developer APIs in conjunction with Beautiful Soup for HTML parsing.

Preprocessing was done using Python RegEx in the [first step](https://github.com/ixig/GPT-2_AutoComplete/blob/main/preproc1.py). Next, entities of interest were [identified](https://github.com/ixig/GPT-2_AutoComplete/blob/main/UkraineNER.ipynb) and white-listed. In the [second step](https://github.com/ixig/GPT-2_AutoComplete/blob/main/preproc2.py), NER using Spacy was used to normalize names of people and places to improve model performance. The [final step](https://github.com/ixig/GPT-2_AutoComplete/blob/main/combine.py) was to combine all sentences from all articles into one line of text which will then be chunked into a context length of 128 tokens for [fine-tuning](https://github.com/ixig/GPT-2_AutoComplete/blob/main/UkraineTrainTextGen.ipynb) a Causal Language Model.

The goal is to show a proof-of-concept of a context fine-tuned, text-prediction auto-completion editor for journalists who have to report from the field under trying conditions and severe time pressures.

---

In the video below, the curses-based [auto-completion editor](https://github.com/ixig/GPT-2_AutoComplete/blob/main/editor.py) is being evaluated on a *new* article written by the NYT *subsequent* to corpus collection. It achieved a Top-10 match of 59%, meaning the writer only needed to type 2 out of every 5 words in the article when writing on a touchscreen device (e.g. phone or tablet). The Top-1 prediction rate was 28%.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/FHjRNtiVmrc/0.jpg)](https://www.youtube.com/watch?v=FHjRNtiVmrc)

---

The following table shows the results of [evaluating](https://github.com/ixig/GPT-2_AutoComplete/blob/main/predict.py) the auto-completion engine on three *new* articles written by the NYT *subsequent* to corpus collection.

| Article | Top-1 %Hit | Top-5 %Hit | Top-10 %Hit |
| --- | --- | --- | --- |
| [new1.txt](https://github.com/ixig/GPT-2_AutoComplete/blob/main/dataset/new/After%20a%20Battering%2C%20Ukraine%20Seeks%20Momentum.txt) | 28% | 51% | 59% |
| [new2.txt](https://github.com/ixig/GPT-2_AutoComplete/blob/main/dataset/new/Bolstering%20Ukraine%20With%20Arms%20That%20Are%20Easy%20to%20Carry%20and%20Simple%20to%20Use.txt) | 28% | 49% | 57% |
| [new3.txt](https://github.com/ixig/GPT-2_AutoComplete/blob/main/dataset/new/Survivors%20Found%20in%20Theater%20Rubble%2C%20but%20Suffering%20Widens.txt) | 27% | 48% | 57% |

This shows that, if necessary (e.g. due to limited screen space), limiting the predictions shown to just the Top-5 Hits still results in 85% of the editor's word prediction capabilities being retained.

Prediction results for the evaluation text can be found in `/results`. These files are ANSI-colored text files. To view, simply `clear && cat xxx.ansi` on a color-enabled terminal window, or Preview in VS Code using the [ANSI Colors](https://marketplace.visualstudio.com/items?itemName=iliazeus.vscode-ansi) extension.

---

The video below shows the auto-completion engine being evaluated on new3.txt -- you can see (based on word coloring) the distribution of the Top-1/5/10 word hits and what types of words were successfully predicted. You can pass the `-m` option to `predict.py` to have it annotate the ranked predictions in the output text for all missed words.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/dkYQAd-K-9U/0.jpg)](https://www.youtube.com/watch?v=dkYQAd-K-9U)

---

## Installation
```bash
git clone https://github.com/ixig/GPT-2_AutoComplete.git
cd GPT-2_AutoComplete
conda create --name gpt2ac --file requirements.txt
conda activate gpt2ac
python -m spacy download en_core_web_md
```

---

## Usage
```
export NYT_API_KEY="..."

# Run the following command daily to collect more articles
python nyt.py dataset/orig

# Once the corpus has been assembled, pre-process the data
python preproc1.py dataset/orig/ dataset/preproc1/
python preproc2.py dataset/preproc1/ dataset/preproc2/
python combine.py dataset/preproc2/ dataset/combine.txt

# Fine-Tuning takes only ~8mins on a P100 GPU
jupyter-notebook UkraineTrainTextGen.ipynb

# Copy the trained model to your local machine
cp <hf_model_ckpt> ./distilgpt2-mlm

# Start typing a fresh article on the (rudimentary!) editor
python editor.py _new1.txt

# Evaluate the Performance of the model or out-of-sample article
# Make sure to run preproc1.py on it first (no need for preproc2)
python predict.py dataset/test/new1.txt _new1.txt -m
```

#### nyt.py :
```
usage: nyt.py [-h] [-k API_KEY] save_dir

Scrape NYT News Articles using NYT Developer APIs

positional arguments:
  save_dir    Saved Articles Directory

optional arguments:
  -k API_KEY  NYT Developer API Key

The API Key can be provided by setting the environmental variable NYT_API_KEY,
or by overriding using the -k command argument
```

#### preproc1.py :
```
usage: preproc1.py [-h] input_dir output_dir

Preprocess Text Files - Step #1

positional arguments:
  input_dir   Input Files Directory
  output_dir  Output Files Directory
```

#### preproc2.py :
```
usage: preproc2.py [-h] input_dir output_dir

Preprocess Text Files - Step #2

positional arguments:
  input_dir   Input Files Directory
  output_dir  Output Files Directory
```

#### combine.py :
```
usage: combine.py [-h] input_dir fout

Combine Individual Text Files into Single (one-line) File

positional arguments:
  input_dir   Input Files Directory
  fout        Output Combined Texts File
```

#### editor.py :
```
usage: editor.py [-h] [-d CKPT] fout

Text Auto-Completer Editor

positional arguments:
  fout        Output File

optional arguments:
  -h, --help  show this help message and exit
  -d CKPT     Model Checkpoint Directory
```

#### predict.py :
```
usage: predict.py [-h] [-m] [-d CKPT] fin fout

Evaluate performance of Text Auto-Completer on file

positional arguments:
  fin         Input File for Evaluation
  fout        Output File with Predictions Annotated

optional arguments:
  -h, --help  show this help message and exit
  -m          Annotate with Prediction Misses
  -d CKPT     Model Checkpoint Directory
```