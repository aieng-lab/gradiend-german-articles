import os
import re
import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

nlp = spacy.load("de_core_news_sm")

def filter_gendered_words(sentences):

    progress = tqdm(total=len(sentences), desc="Filtering sentences")
    step = 0

    neutral_sentences = []

    for sentence in sentences: 
        sent_dict = {}
        ent_search = ", ".join(line.strip() for line in sentence.splitlines())


        step += 1
        if step % 1000 == 0:
            progress.update(1000)


        if len(sentence) < 50: 
            continue
        # helps avoid sentences which contain mostly proper nouns.
        doc = nlp(sentence)
        if len(nlp(ent_search).ents) > 2 :continue

        sentence_tokens = [token for token in doc]

        if any(token.text.lower() == 'das' for token in sentence_tokens): continue

        if any(token.pos_ == 'DET' for token in sentence_tokens):
            continue

        skip_sentence = False
        for token in sentence_tokens:
            if token.pos_ == 'PRON':
                person = token.morph.get('Person')
                if person and person[0] not in ('1', '2'):
                    skip_sentence = True
                    break
        if skip_sentence: 
            continue        
        else: 
            if contains_article(sentence): 
                continue
            else:
                sent_dict.update({
                'text': sentence,
                'dataset_label': "NEUTRAL",
                "label": "NEUTRAL"
            })
                neutral_sentences.append(sent_dict)  

    neutral_sent_df = pd.DataFrame(neutral_sentences)

    # remove duplicates
    neutral_sent_df = neutral_sent_df.drop_duplicates(subset=['text']).reset_index(drop=True)

    return neutral_sent_df


def contains_article(sentence):
    pattern = r'\b(der|die|das|den|dem|des|ein|eine|einer|einem|einen|eines|kein|keine|keiner|keinem|keinen|keines)\b'
    return bool(re.search(pattern, sentence, re.IGNORECASE))


if __name__ == '__main__':
    sentences_leipzig = pd.read_csv("data/der_die_das/deu_news_2024_300K/deu_news_2024_300K-sentences.txt", sep='\t', header=None, names=['id', 'text', 'meta'], quoting=3, engine="python")["text"]

    #wiki_sents_path = "gradiend/data/der_die_das/raw_data/wiki_sentences_extended_1m.jsonl"
    #sentences = load_dataset("json", data_files=wiki_sents_path)["train"]["text"]

    filtered_sents = filter_gendered_words(sentences_leipzig)
    output = 'data_test/der_die_das/neutral/neutral_dwk.csv'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    filtered_sents.to_csv(output)
