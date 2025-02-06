# %%
from neo4j import GraphDatabase, basic_auth
import itertools
import pandas as pd
import utils
# from RE_prompt import get_keyword_form_mention_prompt_system, get_keyword_form_mention_prompt_user
from RE_prompt import rewrite_KG_path_prompt_system, rewrite_KG_path_prompt_user

# %%
# ==============================連線知識圖譜============================== #
# 1. build neo4j knowledge graph datasets
uri = "bolt://localhost:7688"
username = "neo4j"
password = "test"

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()

exclude_relation = ['inverse_isa','concept_in_subset', 'subset_includes_concept', 'is_seronet_authorized_value_for_variable', 'has_seronet_authorized_value', 'has_acc-aha_sars2_authorized_value', 'is_acc-aha_sars2_authorized_value_for_variable', 'is_dipg_dmg_authorized_value_for_variable', 'has_dipg_dmg_authorized_value', 'has_pcdc_all_authorized_value', 'has_pcdc_aml_authorized_value', 'has_pcdc_ews_authorized_value', 'has_pcdc_gct_authorized_value', 'has_pcdc_hl_authorized_value', 'has_pcdc_os_authorized_value']



###################################################
# 給予 head/tail entity，查詢 KG 中他們之間的最短路徑
###################################################
def find_shortest_path(start_entity_name, end_entity_name, all_drug_entity, all_symptom_entity, all_other_entity):
    try_count = 0
    
    related_entity = utils.add_all_entities(all_drug_entity, all_symptom_entity, all_other_entity)

    # print(f"before: {start_entity_name}")
    start_entity_name = utils.get_mention_concept(start_entity_name, mention_type="drug")
    # print(f"after: {start_entity_name}")
    # print(f"before: {end_entity_name}")
    end_entity_name = utils.get_mention_concept(end_entity_name, mention_type="symptom")
    # print(f"after: {end_entity_name}")

    # print(related_entity)

    with driver.session() as session:
        result = session.run(
            """WITH $others AS other_entities
MATCH (start_entity:Concept{name:$start_entity_name}), (end_entity:Concept{name:$end_entity_name})
MATCH p = allShortestPaths((start_entity)-[r*..5]->(end_entity))
WHERE NONE(rel IN r WHERE rel.RELA IN $exclude_relation)
WITH p, [x IN NODES(p) WHERE x.name IN other_entities] AS common_entities, LENGTH(p) AS path_length, [rel IN RELATIONSHIPS(p) WHERE rel.SOURCE IN ['MED-RT', 'SNOMEDCT_US', 'DRUGBANK']] AS relevant_sources_relation
RETURN p, path_length, SIZE(common_entities) AS num_relate_entities, [c IN common_entities | c.name] AS relate_entities, relevant_sources_relation, SIZE(relevant_sources_relation) as num_relevant_sources_relation
ORDER BY num_relevant_sources_relation DESC, path_length ASC
LIMIT 3
""",
            # "WITH $others AS other_entities "
            # "MATCH (start_entity:Concept{name:$start_entity_name}), (end_entity:Concept{name:$end_entity_name}) "
            # "MATCH p = allShortestPaths((start_entity)-[r*..5]->(end_entity)) "
            # "WHERE NONE(rel IN r WHERE rel.RELA IN ['inverse_isa','concept_in_subset', 'subset_includes_concept', 'is_seronet_authorized_value_for_variable', 'has_seronet_authorized_value', 'has_acc-aha_sars2_authorized_value', 'is_acc-aha_sars2_authorized_value_for_variable', 'is_dipg_dmg_authorized_value_for_variable', 'has_dipg_dmg_authorized_value', 'has_pcdc_all_authorized_value', 'has_pcdc_aml_authorized_value', 'has_pcdc_ews_authorized_value', 'has_pcdc_gct_authorized_value', 'has_pcdc_hl_authorized_value', 'has_pcdc_os_authorized_value']) "
            # "WITH p, [x IN NODES(p) WHERE x.name IN other_entities] AS common_entities, LENGTH(p) AS path_length "
            # "RETURN p, path_length, SIZE(common_entities) AS num_relate_entities, [c in common_entities | c.name] AS relate_entities "
            # "ORDER BY path_length ASC, num_relate_entities DESC "
            # "LIMIT 5",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name,
            others=related_entity, 
            exclude_relation=exclude_relation
        )
        paths = []
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            # print(path.relationships)
            # 儲存這個 path 的 entities 與 relations
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship['RELA']
                    relations.append(relation_type)

            # 1. 將這個 path 寫成 string (用 "->" 連結)
            # 2. 判斷 path 中的每一個 entity 是否在 candidate_list
                # 如果有，把這個 entity 加入 exist_entity，代表路經中有經過 candidate_list 其中一個，並且只回傳此 path string
                # 如果沒有，將此 path string 加入到 paths，繼續看下一條 path
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            paths.append(path_str)
            
        # if len(paths) > 5:        
        #     paths = sorted(paths, key=len)[:5]

        return paths
    

###################################################
# 呼叫 LLM
###################################################
def prompt_path_finding(path_input):
    """
    input 為多個類似 'vancomycin->may treat->Diarrhea->is seronet authorized value for variable->Post-COVID-19 Symptom->has seronet authorized value->Dizziness', 'vancomycin->contraindicated with disease->Drug Allergy->has contraindicated drug->cyclizine hydrochloride->may treat->Dizziness', 'vancomycin->contraindicated with disease->Drug Allergy->has contraindicated drug->cyclizine lactate->may treat->Dizziness', 'vancomycin->contraindicated with disease->Drug Allergy->has contraindicated drug->cyclizine->may treat->Dizziness', 'vancomycin->contraindicated with disease->Drug Allergy->has contraindicated drug->cyclizine tartrate->may treat->Dizziness'
    句子組成的 list
    """

    llm_var = {
        'Path': path_input,
        'text': {}
    }

    response_of_KG_path, _, _ = utils.get_llm_result(
        rewrite_KG_path_prompt_system, 
        rewrite_KG_path_prompt_user,
        llm_var,
        model="gpt-4o"
    )

    return response_of_KG_path

###################################################
# （ＬＬＭ）將路徑以 natural language 重寫
###################################################
def get_prompt_path(result_path):
    # 6. knowledge gragh path based prompt generation
    response_of_KG_list_path = ''
    if result_path == []:
        response_of_KG_list_path = ''
    else:
        result_new_path = []
        for total_path_i in result_path:
            # path_input = "->".join(total_path_i)
            result_new_path.append(total_path_i)
        
        path = "\n".join(result_new_path)
        response_of_KG_list_path = prompt_path_finding(path)

    return response_of_KG_list_path

def get_umls_evidence(start_entity_name, end_entity_name, all_drug_entity, all_symptom_entity, all_scispacy_entity):
    paths = find_shortest_path(start_entity_name, end_entity_name, all_drug_entity, all_symptom_entity, all_scispacy_entity)
    if (len(paths) < 1):
        return "", ""
    path_string = get_prompt_path(paths)
    return paths, path_string