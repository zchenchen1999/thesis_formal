# %%
from neo4j import GraphDatabase, basic_auth
import pandas as pd
import utils
from tqdm import tqdm
tqdm.pandas(desc="Processing rows")

import warnings
warnings.filterwarnings("ignore")

# %%
# ==============================連線知識圖譜============================== #
# 1. build neo4j knowledge graph datasets
uri = "bolt://localhost:7687"
username = "neo4j"
password = "test"

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()

# %%
def get_all_side_effects(drug_CID):
    exit_SID = set()
    this_drug_name = ""
    side_effect_with_frequency = []
    with driver.session() as session:
        result = session.run(
            """MATCH (d:Symptom)<-[r]-(s:Drug {CUI:$start_entity_ID}) 
            RETURN s, d, r.frequency""",
            start_entity_ID=drug_CID
        )
        for record in result:
            this_drug_name = record['s']['umls_name']
            if (record['d']['CUI'] not in exit_SID):
                tmp = {
                    'name': record['d']['umls_name'],
                    'CUI': record['d']['CUI'],
                    'frequency': record['r.frequency']
                }
                side_effect_with_frequency.append(tmp)
                exit_SID.add(record['d']['CUI'])
    return this_drug_name, side_effect_with_frequency

def check_has_side_effeck(drug, symptom, drug_CID, symptom_CID):
    match_templete = """[v] "{}"(concept: {}) is "{}"(concept: {}) known adverse drug reaction, frequency: {}"""
    mismatch_templete = """[x] "{}" is "{}" unknown adverse drug reaction."""
    other_similar_templete = """Other possible adverse drug reactions of {}: """

    adr_string = ""
    this_drug_name, all_side_effect = get_all_side_effects(drug_CID)
    for side_effect in all_side_effect:
        if (symptom_CID == side_effect.get('CUI', None)):
            adr_string = match_templete.format(symptom, side_effect['name'], drug, this_drug_name, side_effect['frequency'])
            return adr_string

    # 沒有找到，比對最相似前三名回傳
    if (this_drug_name != ""):
        mis_string = mismatch_templete.format(symptom, drug)
        other_string = ""
        # 將單詞列表轉換為嵌入
        word_embedding = utils.get_embedding(symptom)
        for side_effect in all_side_effect:
            side_effect['embedding'] = utils.get_embedding(side_effect['name'])
            side_effect['similarity'] = utils.cosine_similarity_vectors(word_embedding, side_effect['embedding'])

        # 根據相似度進行排序
        word_list_sorted = sorted(all_side_effect, key=lambda x: x['similarity'], reverse=True)

        # 只保留最相似的前三個 symptoms
        top_3_similar_words = word_list_sorted[:3]
        for index, item in enumerate(top_3_similar_words):
            if (index == 0):
                other_string += "\nOther possible similar adverse drug reactions:"
            other_string += f"\n{index+1}: {this_drug_name}'s adverse drug reaction: {item['name']}, frequency: {item['frequency']}"
            # other_string += f"\n{index+1}: {item['name']}, , Frequency: {item['frequency']}"
        return mis_string + other_string
    else:
        return "No this drug information!"
    return "", {}, all_side_effect
# %%
df = pd.read_csv('/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/multi_path_result/reasoning/asign_answer/asign_concept/10_12_train_r_dsm_GPT4omini.csv')
df = utils.str_to_list(df, ['drugs','symptoms','drug_concept', 'symptom_concept'])
row = df.loc[971]
print(row)
check_has_side_effeck(row['drugs'][0], row['symptoms'][0], row['drug_concept'][1], row['symptom_concept'][1])
# %%
df = pd.read_csv('/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/multi_path_result/reasoning/asign_answer/asign_concept/10_12_train_r_dsm_GPT4omini.csv')
df = utils.str_to_list(df, ['drugs','symptoms','drug_concept', 'symptom_concept'])
# df = df.head(5)
df['side_effect_check'] = df.progress_apply(lambda x: check_has_side_effeck(x['drugs'][0], x['symptoms'][0], x['drug_concept'][1], x['symptom_concept'][1]), axis=1)
df
df.to_csv('/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/multi_path_result/reasoning/asign_answer/asign_concept/asign_adr_check/10_12_train_r_dsm_GPT4omini.csv', index=False)
# %%
