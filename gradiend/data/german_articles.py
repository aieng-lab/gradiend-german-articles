import logging
import os
import random
import re
from operator import index
from pathlib import Path


import pandas as pd
from datasets import load_dataset
from statsmodels.tsa.statespace.tools import set_mode
from tqdm.auto import tqdm
import json
import spacy
import re


logger = logging.getLogger(__name__)
nlp = spacy.load("de_core_news_sm")
nlp.add_pipe("sentencizer")

def load_german_wiki_data(dataset_name="wikipedia", local=False, split='train', chunk_size=200, skip_size=7000, num_repeat=70, total_lines=2665357, dataset_part="20220301.de", output_path= "/wiki_samples.jsonl"):
    logger.warning("Loading the wiki data")
    german_texts = []
    if local:
        german_texts = load_dataset("json", data_files=dataset_name)
    else:

        wiki_dataset = load_dataset(dataset_name, dataset_part, streaming=True, split=split, trust_remote_code=True)
        assert num_repeat * (skip_size + chunk_size) < total_lines

        iterator = iter(wiki_dataset)

        for _ in tqdm(range(num_repeat)):
            german_texts.extend([next(iterator)["text"] for _ in range(chunk_size)])
            # skip some lines
            [next(iterator) for _ in range(skip_size)]

        # save the wiki texts
        save_file_w_json(german_texts, output_path)

    logger.info("Texts loaded")
    return german_texts

def filter_wiki_texts(inputs, article, case, pos,  output_dir, mask, num_samples = 20000):
    """
    :param inputs: the datasets containing the sentences.
    :param article: one of the german articles e.g. der.
    :param case: the case of the article, e.g. NOM, should correspond to the spaCy case notation.
    :param pos:  the part of speech of the article, e.g. DET, should correspond tho the spaCy pos notation.
    :param output_dir: path to where the filtered sentences should be saved.
    :param num_samples the number of samples to keep from the filtered sentences.
    """

    article = article.lower() # ensure that the article is in lowercase
    article_regex = rf"\b{article}\b"
    regex_pattern = re.compile(article_regex, re.IGNORECASE)

    logger.warning('creating wiki sentences')
    wiki_sentences = create_wiki_sentences(inputs)
    filtered_sentences = []

    progress = tqdm(total=len(wiki_sentences), desc="Filtering sentences")
    step = 0

    for sent in wiki_sentences:
        step += 1
        if step % 1000 == 0:
            progress.update(1000)

        if regex_pattern.match(sent.lower()):
            doc = nlp(sent)
            article_tokens = [token for token in doc if token.text.lower() == article and token.pos_ == pos]
            if len(article_tokens) < 5 and all(token.morph.get("Case") == [case] for token in article_tokens):
                filtered_sentences.append(sent)

    logger.warning(f"Filtered {len(filtered_sentences)} sentences for article: {article}.")
    if (len(filtered_sentences) > num_samples):
        filtered_sentences = random.sample(filtered_sentences, num_samples)

    create_filtered_set(filtered_sentences, article.upper() ,mask, output_dir)


def filter_article(sentences, article, cases, pos, mask, output_dir, gender, dataset_label,number,num_samples = None, isDas= False):
    """
    Filters sentences based on the specified article, cases, part of speech.
    sentences: the list of sentences to filter.
    article: the article to filter by.
    cases: the grammatical cases to filter by.
    """
    article = article.lower()  
    article_regex = rf"\b{article}\b"
    regex_pattern = re.compile(article_regex, re.IGNORECASE)

    filtered_sentences = []

    progress = tqdm(total=len(sentences), desc="Filtering sentences")
    step = 0

    for sent in sentences:
        step += 1
        if step % 1000 == 0:
            progress.update(1000)

        if len(sent) < 50 or len(sent) > 500:
            continue

        if regex_pattern.search(sent.lower()):
            doc = nlp(sent)
            if len(doc.ents) > 3 :continue
            else:
                article_tokens = [token for token in doc if token.text.lower() == article and token.pos_ == pos]
                if 0 < len(article_tokens) < 5 and all(token.morph.get("Case") in [[case] for case in cases] and token.morph.get("Gender") == [gender] and token.morph.get("Number") == [number] for token in article_tokens):
                    if isDas:     
                        modified_sent = modify_das_pron(sent)
                        filtered_sentences.append(modified_sent)
                    else:
                        filtered_sentences.append(sent)

    logger.warning(f"Filtered {len(filtered_sentences)} sentences for article: {article}.")
    if num_samples and len(filtered_sentences) > num_samples:
        random.seed(42)
        filtered_sentences = random.sample(filtered_sentences, num_samples)

    if isDas: 
        create_filtered_set(filtered_sentences, 'das_det', mask, output_dir, dataset_label)
    else:
        create_filtered_set(filtered_sentences, article.upper() ,mask, output_dir, dataset_label)


def modify_das_pron(sent):
    sent_doc = nlp(sent)
    modified = ""
    for token in sent_doc: 
        if token.text.lower() == 'das' and token.pos_ == 'DET':
            modified += "das_det"
            modified += token.whitespace_
        else:
            modified += token.text_with_ws
    return modified        


def create_wiki_sentences(wiki_texts):
    wiki_sentences = []

    for entry in tqdm(wiki_texts):
        sentences = [sent.text.strip() for sent in nlp(entry).sents]
        wiki_sentences.extend(sentences)

    #TODO: change to info -> config has to be changed so it actually prints 
    logger.info(f"{len(wiki_sentences)} sentences were created.")
    return wiki_sentences


def save_file_w_json(input, output_dir):
    with open(output_dir, "a", encoding="utf-8" ) as f:
        for entry in input:
            json.dump({"text": entry}, f, ensure_ascii=False)
            f.write("\n")


def create_filtered_set(inputs,label, mask, output_dir, dataset_label):
    article = label.lower()  # ensure that the article is in lowercase
    article_regex = rf"\b{re.escape(article)}\b"
    regex_pattern = re.compile(article_regex, re.IGNORECASE)

    columns = ['id', 'text', 'masked', 'label', 'token_count', 'dataset_label']
    row_id = 0
    rows = []
    for sent in inputs:
        sent_dict = {}
        row_id += 1

        token_count = len(regex_pattern.findall(sent))

        masked_sentence = re.sub(article_regex, mask, sent, flags=re.IGNORECASE)
        sent  = substitute_das_det(sent)

        print(masked_sentence)
        sent_dict.update({
            'id': row_id,
            'text': sent.replace("\n"," "),
            'masked': masked_sentence.replace("\n"," "),
            'label': label,
            'token_count': token_count,
            'dataset_label': dataset_label
        })
        rows.append(sent_dict)

    
    output_dir = Path(output_dir) / dataset_label
    print(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_dir = output_dir / f'filtered_{dataset_label}.csv'

    filtered_df = pd.DataFrame(rows, columns=columns)
    filtered_df.to_csv(save_dir, index=False)



def substitute_das_det(text):
    # Replace sentence-start 'das_det' with 'Das'
    text = re.sub(r'(^\s*)(das_det)\b', r'\1Das', text)

    # Replace '. das_det' with '. Das'
    text = re.sub(r'(\.\s+)(das_det)\b', r'\1Das', text)

    # Replace all other 'das_det' with 'das'
    text = re.sub(r'\bdas_det\b', 'das', text)

    return text


def filter_ner(sentence):
    sentence_nlp = nlp(sentence)
    if len(sentence_nlp.ents) > 3:
        return False
    else:
        return True


def generate_data(use_wiki_data=True):
    if use_wiki_data:
        wiki_data = load_german_wiki_data()
        sentences = create_wiki_sentences(wiki_data)
    else:
        # this dataset should work better for direct MFN training
        # load https://downloads.wortschatz-leipzig.de/corpora/deu_news_2024_300K.tar.gz
        leipzig_dataset = pd.read_csv("gradiend/data/der_die_das/deu_news_2024_300K/deu_news_2024_300K.txt")
        sentences = leipzig_dataset['text'].tolist()

    case_gender_mapping = {
        'Nom': {'Fem': 'die', 'Masc': 'der', 'Neut': 'das'},  # Nominativ
        'Gen': {'Fem': 'der', 'Masc': 'des', 'Neut': 'des'},  # Genitiv
        'Acc': {'Fem': 'die', 'Masc': 'den', 'Neut': 'das'},  # Akkusativ
        'Dat': {'Fem': 'der', 'Masc': 'dem', 'Neut': 'dem'},  # Dativ
    }

    leipzig_str = '' if use_wiki_data else 'leipzig/'

    for case, gender_data in case_gender_mapping.items():
        for gender, article in gender_data.items():
            dataset_label = f"{case[0].upper()}{gender[0].upper()}"
            output_dir = f'data_test/der_die_das/splits_no_limit/{leipzig_str}'
            filter_article(
                sentences,
                article=article,
                cases=[case],
                pos='DET',
                mask=f"[{article.upper()}_MASK]",
                output_dir=output_dir,
                gender=gender,
                dataset_label=dataset_label,
                number='Sing',
                isDas=(article == 'das'),
            )


def some_leipzig_filtering():
    suffix = ['train', 'test', 'val']

    gram_categories = ['AM', 'AN', 'AF', 'NM', 'NF', 'NN']

    for gram_cat in gram_categories:
        for s in suffix:
            source_csv_path = f'gradiend/data/der_die_das/splits/leipzig/{gram_cat}/filtered_{gram_cat}_{s}.csv'
            df = pd.read_csv(source_csv_path)
            half_indices = random.sample(list(df.index), k=len(df) // 2)
            df.loc[half_indices, 'dataset_label'] = df.loc[half_indices, 'dataset_label'].apply(lambda x: f"_{x}")

            directory = f'gradiend/data/der_die_das/splits/MFN/leipzig/{gram_cat}_mfn'

            if not os.path.exists(directory):
                os.makedirs(directory)

            target_csv_path = f'{directory}/filtered_{gram_cat}_{s}.csv'

            df.to_csv(target_csv_path, index=False)





#import threading
if __name__ == '__main__':
    generate_data(use_wiki_data=True)