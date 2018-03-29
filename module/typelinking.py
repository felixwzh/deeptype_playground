import sys
import os
dataroot = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(dataroot, "learning/"))

import time
import marisa_trie
import pickle
from dataset import *
from wikidata_linker_utils.offset_array import OffsetArray
import train_type as tp
from collections import defaultdict
from tqdm import tqdm
import re

def create_indices2title(infile, outfile):
    out = defaultdict(list)
    with open(infile) as f:
        for line in tqdm(f):
            it = line.replace('\n', '').split('\t')
            target = 'wiki/'
            start = it[0].index(target)
            end = start + len(target)
            k, v = it[0][:start], it[0][end:]
            out[int(it[1])][k] = v
    with open(outfile, 'wb') as f:
        pickle.dump(out, f)

def get_prob(tagger_ins,sentence_splits):
    ps = tagger_ins.predict_proba_sentences([sentence_splits])
    output = [i for i in ps]
    probs = output[0]['type']
    return probs[0]


def load_trie(language_path):
    trie_index2indices_values = OffsetArray.load(
        join(language_path, "trie_index2indices")
    )
    trie_index2indices_counts = OffsetArray(
        np.load(join(language_path, "trie_index2indices_counts.npy")),
        trie_index2indices_values.offsets
    )
    trie = marisa_trie.Trie().load(
        join(language_path, "trie.marisa")
    )
    return trie_index2indices_values, trie_index2indices_counts, trie

def solve_indices_and_linkprob(mention, trie, trie_index2indices_values, trie_index2indices_counts, min_prob=0.01):
    anchor = trie.get(mention)
    if anchor is not None:
        indices = trie_index2indices_values[anchor]
        link_probs = trie_index2indices_counts[anchor]
        link_probs = link_probs / link_probs.sum()
        mask = link_probs > min_prob
        indices = indices[mask]
        link_probs = link_probs[mask]
    else:
        indices = None
        link_probs = 0.0
    return indices, link_probs

def simple_tokenize(sentence):
    sentence = re.sub(r'[^\w ]', '', sentence).lower()
    return sentence.split()

def solve_model_probs(sentence, tagger, tokenize=simple_tokenize):
    sent_splits = tokenize(sentence)
    model_probs = get_prob(tagger,sent_splits)
    return sent_splits, model_probs

def solve_type_probs(mention, sent_splits, model_probs, type_oracle, indices, alpha_type_belief=0.5):
    token_location = sent_splits.index(mention)
    type_belief = model_probs[token_location]
    assignments = type_oracle.classify(indices)
    type_probs = type_belief[assignments]
    type_probs = alpha_type_belief * type_probs + (1.0 - alpha_type_belief)
    return type_probs

def solve_full_score(link_probs, type_probs, beta=0.99):
    full_score = link_probs * (1.0 - beta + beta * type_probs)
    return full_score

def pick_top_entity(full_score, indices, indices2title):
    index = full_score.argmax()
    top_pick = indices[index]
    return indices2title[top_pick]

def run(mentions, sent_splits, model_probs,  indices2title, type_oracle, trie, trie_index2indices_values, trie_index2indices_counts, only_link=False):
    entities = []
    for mention in mentions:
        indices, link_probs = solve_indices_and_linkprob(mention, trie, trie_index2indices_values, trie_index2indices_counts)
        if indices is not None:
            if only_link:
                full_score = link_probs
            else:
                type_probs = solve_type_probs(mention, sent_splits, model_probs, type_oracle, indices)
                full_score = solve_full_score(link_probs, type_probs)
            entity = pick_top_entity(full_score, indices, indices2title)
        else:
            entity = None
        entities.append(entity)
    return entities

if __name__ == '__main__':
    global dataroot
    tagger = tp.SequenceTagger(os.path.join(dataroot,'my_great_model/'))
    with open(os.path.join(dataroot, 'data/wikidata/indices2title.pkl'), 'rb') as hdl:
        indices2title = pickle.load(hdl)
    sentence = "The prey saw the jaguar cross the jungle"
    sent_splits, model_probs = solve_model_probs(sentence, tagger)
    mentions = ["jaguar", "jungle"]
    type_oracle = load_oracle_classification(os.path.join(dataroot, "data/classifications/type_classification"))
    trie_index2indices_values, trie_index2indices_counts, trie = load_trie(os.path.join(dataroot, 'data/en_trie'))
    entities = run(mentions, sent_splits, model_probs, indices2title, type_oracle, trie, trie_index2indices_values, trie_index2indices_counts)
    print(entities)

