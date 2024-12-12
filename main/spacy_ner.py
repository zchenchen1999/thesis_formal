# %%
import spacy
nlp = spacy.load("en_core_web_trf")
import en_core_web_trf
spacy_nlp = en_core_web_trf.load()
from spacy import displacy

import scispacy
scispacy_nlp = spacy.load("en_ner_bionlp13cg_md")

from pprint import pprint
from collections import Counter


import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="Processing rows")
# %%

def get_entities(text):
    spacy_doc = spacy_nlp(text)
    spacy_entity = [(X.text, X.label_) for X in spacy_doc.ents]
    
    scispacy_doc = scispacy_nlp(text)
    scipacy_entity = [(X.text, X.label_) for X in scispacy_doc.ents]

    # 顯示
    # pprint(spacy_entity)
    # spacy_labels = [x.label_ for x in spacy_doc.ents]
    # Counter(spacy_labels)
    # displacy.render(spacy_nlp(text), jupyter=True, style='ent')

    # pprint(scipacy_entity)
    # scispacy_labels = [x.label_ for x in scispacy_doc.ents]
    # Counter(scispacy_labels)
    # displacy.render(scispacy_nlp(text), jupyter=True, style='ent')

    return spacy_entity, scipacy_entity