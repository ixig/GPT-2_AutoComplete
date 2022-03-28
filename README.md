# GPT-2_AutoComplete
## A Context Fine-Tuned, GPT-2 Auto-Completion Text Editor

Five journalists lost their lives in the course of performingtheir work in just the first month of the 2022 Russia-Ukraine War.

<p align="center">
<img src="UkraineJournalistsKilled.jpg" alt="5 journalists killed in 1 month" width=500>
</p>

I fine-tuned a Hugging Face DistilGPT2 Language Model on a corpus consisting of one week of news articles from the NYT and AP News at the start of the war. The news articles were [scraped](https://github.com/ixig/GPT-2_AutoComplete/blob/main/nyt.py) using the NYT Developer APIs in conjunction with Beautiful Soup for HTML parsing.

Preprocessing was done using Python RegEx in the [first step](https://github.com/ixig/GPT-2_AutoComplete/blob/main/preproc1.py). In the [second step](https://github.com/ixig/GPT-2_AutoComplete/blob/main/preproc2.py), NER using Spacy was used to normalize names of people and places. The [final step](https://github.com/ixig/GPT-2_AutoComplete/blob/main/combine.py) was to combine all sentences from all articles into one line of text which will then be chunked into a context length of 128 tokens for [fine-tuninng](https://github.com/ixig/GPT-2_AutoComplete/blob/main/UkraineTrainTextGen.ipynb).

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

---

The video below shows the auto-completion engine being evaluated on new3.txt -- you can see (based on word coloring) the distribution of the Top-1/5/10 word hits and what types of words were successfully predicted. You can pass the `-m` option to `predict.py` to have it annotate the ranked predictions in the output text for all missed words.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/dkYQAd-K-9U/0.jpg)](https://www.youtube.com/watch?v=dkYQAd-K-9U)
