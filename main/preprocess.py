# %%
import pandas as pd

from replace_pronoun import get_replace_pronoun
from spacy_ner import get_entities
from create_sentences import get_sentences

from tqdm import tqdm
tqdm.pandas(desc="Processing rows")

import utils
from evidence_path_umls import get_umls_evidence
# %%

def main():
    df = pd.read_csv('/home/zchenchen1999/thesis/dataset/BC5CDR/BC5CDR_all.csv')

    # df = df.head(1)
    # 轉換成 list
    df = utils.str_to_list(df, ['drugs', 'symptoms', 'relations'])

    # 切句子
    df['text'] = df['text'].apply(lambda x: x.replace("Read More Read Less", ""))
    df['sents'] = df['text'].apply(lambda x: get_sentences(x))

    # 產生替換代名詞的句子
    df['sents_replace_pronoun'] = df.progress_apply(lambda x: get_replace_pronoun(x['sents']), axis=1)
    df['text_replace_pronoun'] = df.apply(lambda x: " ".join(x['sents_replace_pronoun']), axis=1)


    # 轉換小寫
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text_replace_pronoun'] = df['text_replace_pronoun'].apply(lambda x: x.lower())
    df['sents'] = df['sents'].apply(lambda x: [y.lower() for y in x])
    df['sents_replace_pronoun'] = df['sents_replace_pronoun'].apply(lambda x: [y.lower() for y in x])
    df['drugs'] = df['drugs'].apply(lambda x: [y.lower() for y in x])
    df['symptoms'] = df['symptoms'].apply(lambda x: [y.lower() for y in x])
    df['relations'] = df['relations'].apply(
        lambda relation_list: [(x.lower(), y.lower(), z.lower()) for x, y, z in relation_list]
    )

    # 產生 spacy entity
    df[['spacy_entity', 'scispacy_entity']] = df.apply(lambda x: get_entities(x['text']), axis=1, result_type='expand')
    df[['spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun']] = df.apply(lambda x: get_entities(x['text_replace_pronoun']), axis=1, result_type='expand')

    # 使用的 context_type 種類（原始 or 替換代名詞）
    context_type = "_replace_pronoun" # "", "_replace_pronoun"

    df['drugs'] = df['drugs'].progress_apply(
        lambda drug_list: [(item, value1, value2) 
            for item, (value1, value2) in [(item, utils.get_mention_concept(item, mention_type='drug'))
            for item in drug_list]]
    )
    df['symptoms'] = df['symptoms'].progress_apply(
        lambda symptom_list: [(item, value1, value2) 
            for item, (value1, value2) in [(item, utils.get_mention_concept(item, mention_type='symptom'))
            for item in symptom_list]]
    )

    # 儲存整篇文章的 drug, symptom（也儲存對應到 umls 的結果），並且 explode
    df['all_drug'] = df['drugs'].apply(lambda x: [item[0] for item in x])
    df['all_drug_concept'] = df['drugs'].apply(lambda x: [item[1] for item in x])
    df['all_symptom'] = df['symptoms'].apply(lambda x: [item[0] for item in x])
    df['all_symptom_concept'] = df['symptoms'].apply(lambda x: [item[1] for item in x])
    df['all_scispacy_concept'] = df[f'scispacy_entity{context_type}'].apply(lambda x: utils.add_all_entities(x))
    df['all_scispacy_concept'] = df['all_scispacy_concept'].progress_apply(lambda x: [utils.get_mention_concept(item, mention_type='None') for item in x])

    df = utils.explode_relation(df)

    # 重新排序
    df = df[['id', 'title', 'drugs', 'symptoms', 'relations', 'text', 'sents',
            'spacy_entity', 'scispacy_entity', 'text_replace_pronoun',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun',
            'scispacy_entity_replace_pronoun', 'all_drug', 'all_drug_concept',
            'all_symptom', 'all_symptom_concept', 'all_scispacy_concept',
            'ground_truth']]
    
    df = df.reset_index(drop=True)

    # # 每一種 path 的 evidence
    print("正在處理 consecutive evidence...")
    df['consecutive_evidence'] = df.apply(lambda x: utils.get_consecutive_string(x['drugs'][0], x['symptoms'][0], x[f'sents{context_type}'], x['all_drug'], x['all_symptom'], x[f'spacy_entity{context_type}'], x[f'scispacy_entity{context_type}']), axis=1)

    print("正在處理 multi hop evidence...")
    df['multi_hop_evidence'] = df.apply(lambda x: utils.get_multi_hop_string(x['drugs'][0], x['symptoms'][0], x[f'sents{context_type}'], x['all_drug'], x['all_symptom'], x[f'spacy_entity{context_type}'], x[f'scispacy_entity{context_type}']), axis=1)

    print("正在處理 default evidence...")
    df['default_evidence'] = df.apply(lambda x: utils.get_default_string(x['drugs'][0], x['symptoms'][0], x[f'sents{context_type}'], x['all_drug'], x['all_symptom'], x[f'spacy_entity{context_type}'], x[f'scispacy_entity{context_type}']), axis=1)

    # 重新排序
    df = df[['id', 'title', 'drugs', 'symptoms', 'relations', 'text', 'sents',
            'spacy_entity', 'scispacy_entity', 'text_replace_pronoun',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun',
            'scispacy_entity_replace_pronoun', 'all_drug', 'all_drug_concept',
            'all_symptom', 'all_symptom_concept', 'all_scispacy_concept',
            'consecutive_evidence', 'multi_hop_evidence', 'default_evidence',
            'ground_truth']]

    # 分成兩個資料集
    train_set = df.groupby('ground_truth', group_keys=False).apply(lambda x: x.sample(frac=0.7))
    test_set = df.drop(train_set.index) # 剩下的資料拿去抽樣 demonstration

    train_set.to_csv("/home/zchenchen1999/thesis/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_train.csv", index=False)
    test_set.to_csv("/home/zchenchen1999/thesis/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_test.csv", index=False)
    return df
main()
# %%
