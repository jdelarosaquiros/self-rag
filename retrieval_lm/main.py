from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import requests
import string
from lxml import html
from googlesearch import search
from bs4 import BeautifulSoup
import re
import json
import time
from run_short_form import (
    call_model_rerank_w_scores_batch,
    load_special_tokens,
)


def get_top_search_result(query):
    try:
        fallback = "No source found."
        search_results = list(search(query, tld="co.in", num=5, stop=3, pause=1))
        results: list[tuple[str, str]] = []
        print(search_results)
        # search_results = ["https://en.wikipedia.org/wiki/Computer_science"]
        if not search_results:
            return "No search results found."
        for result in search_results:
            # print(result)
            page = requests.get(result, timeout=2)
            if page.status_code != 200:
                continue
            soup = BeautifulSoup(page.content, features="lxml")
            article_text: str = ""
            article = soup.findAll("p")
            # print(article)
            for element in article:
                article_text += "\n" + "".join(element.findAll(string=True))
            article_text = article_text.replace("\n", "")
            first_sentence = article_text.split(".")
            first_sentence = first_sentence[0].split("?")[0]

            chars_without_whitespace = first_sentence.translate(
                {ord(c): None for c in string.whitespace}
            )

            if len(chars_without_whitespace) <= 0:
                article_text = fallback

            results.append((result, article_text))
        return results
    except:
        print("An error occurred while searching the web.")
        return results


model_name = "selfrag/selfrag_llama2_7b"
download_dir = "/gscratch/h2lab/akari/model_cache"
dtype = "half"
threshold = 0.05
mode = "adaptive_retrieval"
use_groundness = True
use_utility = True
use_seqscore = True
task = None
w_rel = 0.2
w_sup = 1.0
w_use = 1.0
max_new_tokens = 100
world_size = 1


class Prediction:
    def __init__(
        self,
        text: str,
        do_retrieve: bool,
        passage: str | None = None,
        score: float | None = None,
        source: str | None = None,
    ):

        self.text = text
        self.passage = passage
        self.score = score
        self.source = source
        self.do_retrieve = do_retrieve


gpt = model_name
tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
    tokenizer, use_grounding=use_groundness, use_utility=use_utility
)

model = LLM(
    model=gpt,
    download_dir=download_dir,
    dtype=dtype,
    tensor_parallel_size=world_size,
)


def generate(prompt, evidences):
    return call_model_rerank_w_scores_batch(
        prompt,
        evidences=evidences,
        model=model,
        max_new_tokens=max_new_tokens,
        rel_tokens=rel_tokens,
        ret_tokens=ret_tokens,
        grd_tokens=grd_tokens,
        ut_tokens=ut_tokens,
        threshold=threshold,
        use_seqscore=use_seqscore,
        w_rel=w_rel,
        w_sup=w_sup,
        w_use=w_use,
        mode=mode,
        closed=task in ["fever", "arc_c"],
    )


def get_best_prediction(predictions: list[Prediction]):
    if len(predictions) == 0:
        return None

    if len(predictions) == 1:
        return predictions[0]

    best_prediction = predictions[0]

    for prediction in predictions:
        if best_prediction.score < prediction.score:
            best_prediction = prediction

    return best_prediction


# Start Script
# model = LLM(
#     "selfrag/selfrag_llama2_7b",
#     download_dir="/gscratch/h2lab/akari/model_cache",
#     dtype="half",
# )
# sampling_params = SamplingParams(
#     temperature=0.4,
#     top_p=1.0,
#     max_tokens=100,
#     skip_special_tokens=False,
#     logprobs=2,
#     prompt_logprobs=2,
# )


start_time = time.time()
prompt = "what did Ada Lovelace wrote during the translation of a French article?"
search_results = get_top_search_result(prompt)
best_predictions = []

if len(search_results) < 1:
    print("No web page found")
    exit()

for source, result in search_results:
    passages = [result[i : i + 1000] for i in range(0, len(result), 1000)]
    print("Source:", source)
    print("Number of Passages: ", len(passages))

    # prompts = [format_prompt(prompt, paragraph=passage) for passage in result_parts]

    #   preds = model.generate([format_prompt(prompt, paragraph=passage) for passage in passages], sampling_params)
    #   best_predictions.append(get_best_prediction_compose(preds, passages))

    pred, pred_key, results, do_retrieve = generate(prompt, passages)
    # print(results)

    if type(pred) is str and pred[0] == "#" or pred[0] == ":":
        pred = pred[1:]

    if do_retrieve:
        best_prediction = Prediction(
            pred,
            do_retrieve,
            results[pred_key]["ctx"],
            results[pred_key]["score"],
            source,
        )
    else:
        best_prediction = Prediction(pred, do_retrieve)
        best_predictions.append(best_prediction)
        break

    best_predictions.append(best_prediction)


if len(best_predictions) <= 0:
    print("No predictions found")
    exit()

best_prediction = get_best_prediction(best_predictions)

end_time = time.time()
execution_time = end_time - start_time

print()
print("Do Retrieval:", best_prediction.do_retrieve)
print("Prediction:", best_prediction.text)
print("Score:", best_prediction.score)
print("Passage Cited:", best_prediction.passage)
print("Source:", best_prediction.source)
print("Execution time:", execution_time, "seconds")
