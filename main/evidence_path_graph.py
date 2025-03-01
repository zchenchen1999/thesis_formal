# %%
import networkx as nx
from collections import defaultdict
import heapq
from pyvis.network import Network
# %%
# Step 1: 建立實體共現圖, document 為斷句的 list
def build_entity_graph(document, entities):
    graph = nx.Graph()
    for index, sentence in enumerate(document):
        present_entities = [entity for entity in entities if entity in sentence]
        for i in range(len(present_entities)):
            for j in range(i + 1, len(present_entities)):
                graph.add_edge(present_entities[i], present_entities[j], sentence=sentence, sent_index=index)
    return graph

# Step 2: 計算實體的特徵向量中心性
def calculate_eigenvector_centrality(graph):
    centrality = nx.eigenvector_centrality(graph)
    return centrality

# 給予 2 entity，使用 beam search 找到最高分的路徑
def beam_search(graph, start_entity, end_entity, width=10):
    centrality = calculate_eigenvector_centrality(graph)
    print(centrality)
    beam = [(centrality[start_entity], -len([start_entity]), start_entity, [start_entity])]
    best_path = []
    
    while beam:
        print("==========new while==========")
        for path in beam:
            print(path)
        score, neg_len, current_entity, path = heapq.heappop(beam)
        
        print(f"current: {current_entity}")
        
        if current_entity == end_entity:
            print(f"end: {end_entity}")
            best_path = path
            break
        
        neighbors = graph.neighbors(current_entity)

        for neighbor in neighbors:
            if neighbor not in path:
                print(f"====={neighbor}")
                new_path = path + [neighbor]
                print(f"new path: {new_path}")
                new_score = min(score, centrality[neighbor])
                heapq.heappush(beam, (new_score, -len(new_path), neighbor, new_path))
                print(beam)
        
        beam = sorted(beam, reverse=True)[:width]
        # beam = sorted(beam, reverse=True)
    
    return best_path

# 取得路徑中的所有句子
def get_sentences_from_path(graph, paths):
    all_sentences = []
    all_sentences_index = []
    all_link_path_string = ""
    for path in paths:
        sentences = []
        sentences_index = []
        link_path_string = ""
        for index in range(len(path) -1):
            if index == len(path):
                break
            edge_data = graph.get_edge_data(path[index], path[index+1])
            sentences.append(edge_data['sentence'])
            sentences_index.append(edge_data['sent_index'])
            link_path_string += f"{edge_data['sentence']}\n({(path[index])}, {(path[index+1])})\n\n"
        all_sentences.append(sentences)
        all_sentences_index.append(sentences_index)
        all_link_path_string += f"\n=================\n{link_path_string}"

    all_link_path_string+="================="
    all_sentences_index = sorted(all_sentences_index, key=lambda x: x[0])

    # 去除重複的 sentence 組合
    sorted_inner = [tuple(sorted(inner)) for inner in all_sentences]
    # 使用 set 去重
    unique_sorted = set(sorted_inner)
    all_sentences = [list(item) for item in unique_sorted]

    # 去除重複的 sentence 組合
    sorted_inner = [tuple(sorted(inner)) for inner in all_sentences_index]
    # 使用 set 去重
    unique_sorted = set(sorted_inner)
    all_sentences_index = [list(item) for item in unique_sorted]

    return all_sentences, all_sentences_index, all_link_path_string

def order_sentences(original_sentences, evidence_sentences_index):
    # 取 unique
    unique_index = list(set(item for sublist in evidence_sentences_index for item in sublist))
    sorted_evidence_indexs = sorted(unique_index)
    envidence_string = ""
    # 創建 evidence_sentences 的索引順序
    for index, sentence_index in enumerate(sorted_evidence_indexs):
        envidence_string += f"{index+1}. {original_sentences[sentence_index]}\n"
    return sorted_evidence_indexs, envidence_string

def get_evidence_sents(sentences_for_search, all_entities, head_entity, tail_entity):
    # 1. 建立文件 graph (entity 為 node, 共現句子為 edge)
    graph = build_entity_graph(sentences_for_search, all_entities)
    #2. 找所有最短路徑
    try:
        shortest_paths = list(nx.all_shortest_paths(graph, source=head_entity, target=tail_entity, weight=None))
        evidence_sentences, evidence_sentences_index, link_path_string = get_sentences_from_path(graph, shortest_paths)
        ordered_evidence_indexs, ordered_evidence_sentences = order_sentences(sentences_for_search, evidence_sentences_index)
        return ordered_evidence_indexs, link_path_string, ordered_evidence_sentences
    except:
        return [], "", ""
    # shortest_paths = p.get(head_entity, {}).get(tail_entity, None)
    # if (shortest_paths == None):
    #     return [], "", ""
    # else:
    #     evidence_sentences, sentences_index, link_path_string = get_sentences_from_path(graph, shortest_paths)
    #     ordered_evidence_sentences = order_sentences(all_sentences, evidence_sentences)
    #     return sentences_index, link_path_string, ordered_evidence_sentences

# # %%
# ############################################
# # Test
# ############################################
# import pandas as pd
# import ast

# def str_to_list(df, columns):
#     for c in columns:
#         df[c] = df[c].apply(lambda x: ast.literal_eval(x))
#     return df
# def add_all_entities(*args):
#     unique_elements = set()

#     for arg in args:
#         for item in arg:
#             if isinstance(item, tuple):
#                 unique_elements.add(item[0])  # 如果是 tuple，取第 0 項
#             else:
#                 unique_elements.add(item)  # 如果不是 tuple，直接取值
    
#     return list(unique_elements)

# store_path = "/home/zchenchen1999/thesis_formal/main/result/webmd"
# df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/WebMD/WebMD_annotated_v2_exploded_reasoning_train.csv')

# # 轉換成 list
# df_train = str_to_list(df_train, 
#         ['drugs', 'symptoms', 'relations', 
#         'sents', 'spacy_entity', 'scispacy_entity',
#         'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun',
#         'all_drug', 'all_symptom']
#     )

# row = df_train.loc[0]

# search_ecidence_type = "_replace_pronoun"
# context = row[f'text{search_ecidence_type}']
# sents = row[f'sents{search_ecidence_type}']
# head_mention = row['drugs'][0]
# tail_mention = row['symptoms'][0]
# all_drug_metions = row['all_drug']
# all_symptom_mentions = row['all_symptom']
# # spacy_entity = row[f'spacy_entity{search_ecidence_type}']
# scispacy_entity = row[f'scispacy_entity{search_ecidence_type}']

# entity_list = add_all_entities(all_drug_metions, all_symptom_mentions, scispacy_entity)
# entity_list
# # %%
# sentences_index, link_path_string, multi_hop_string = get_evidence_sents(sents, entity_list, head_mention, tail_mention)
# multi_hop_string
# # %%
# sentences_index
# # %%
