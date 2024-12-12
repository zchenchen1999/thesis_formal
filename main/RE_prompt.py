# =====================================
# Relation Definitions
# =====================================
# The following are the Relation Definitions and Extraction Rules:
# ****
# Relation Definitions:
#     1. has_positive_reaction: This relation captures the connection between a drug and the improvement of an existing symptom that was already present in the patient before taking the drug.
#     2. has_negative_reaction: This relation captures the connection between a drug and the worsening or lack of improvement of an existing symptom that was already present in the patient before taking the drug. Specifically, the "lack of improvement" aspect applies only if the patient explicitly states that the drug was ineffective for addressing a particular symptom.
#     3. has_adverse_drug_reaction: This relation captures the connection between a drug and a new adverse reaction experienced by the patient after taking the drug.
#     4. none_of_above: This relation is used when the reaction or symptom change does not fit into any of the above categories.

# Extraction Rules:
#     - Explicit Mentions: Only extract relations where the reaction or symptom change is explicitly mentioned as a result of taking the drug. Do not infer relationships that are not directly stated by the patient.
#     - Context Consideration: Consider the context of the sentence to ensure the reaction or symptom change is indeed linked to the drug and not another potential cause.
#         - Example: "I took Aspirin and later felt better, but I also rested a lot." In this case, the relationship may be ambiguous and should be noted as such.
#     - Multiple Drugs and Reactions: Classify reactions resulting from the combination of multiple drugs as none_of_above, as it is unclear which drug caused the reaction or if the combination caused the reaction.
#         - Example: "After taking Aspirin and Ibuprofen, I felt nauseous." In this case, classify as none_of_above due to the combination of drugs.
#     - Non-specific Terms: Classify statements that do not clearly indicate a drug or a reaction as none_of_above.
#         - Example: "I felt better after my medication." This statement is too vague to extract a specific relation, so classify as none_of_above.
# ****


# ==================================================================================================== #
#                                              BC5CDR                                                  #
#                                              BC5CDR                                                  #
#                                              BC5CDR                                                  #
#                                              BC5CDR                                                  #
# ==================================================================================================== #

##########################################
# BC5CDR relation extraction system prompt
##########################################
bc5cdr_prompt_system = """Task: Relation Extraction

Given a context, identify the most accurate relation between two specified entities. There are two possible relation categories that the relation must belong to: A, B.

Please ensure that the identified relation follows the definitions provided below.

**Relation Definitions**:
```
Relation Definitions:
    1. A: Refer to two types of relationships between a chemical and a disease in CTD: 
            - Putative mechanistic relationships between a chemical and disease indicates that the chemical may play a role in the etiology of the disease (e.g., exposure to chemical X causes lung cancer).
            - Biomarker relations between a chemical and disease indicates that the chemical correlates with the disease (e.g., increased abundance in the brain of chemical X correlates with Alzheimer disease).
    2. B: This relation is used when the reaction does not fit into A category.
```

Here are some examples:
{demonstrations}
"""
#########################################
# 給予 context 判斷 entity pair 的 relation
#########################################
bc5cdr_vanilla_ICL_prompt_user = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.
MAKE SURE your output is one of the four categories stated (A,B).
Response the determined relation type (one of A or B) only, nothing else.

Context: {context}

Given the context, the relation between "{head_entity}" and "{tail_entity}" is """

#############################################################
# 給予 context + evidence sentence 判斷 entity pair 的 relation
#############################################################
bc5cdr_S2PR_evidence_prompt_user = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the evidence provided in the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.
MAKE SURE your output is one of the four categories stated (A,B).
Response the determined relation type (one of A or B) only, nothing else.

Context: {context}

Evidence sentences in the context:
```
{evidence}
```

Given the context and **focusing on the evidence sentences**, the relation between "{head_entity}" and "{tail_entity}" is """


#######################
# (CoT) system prompt
#######################
bc5cdr_prompt_system_reasoning = """Task: Relation Extraction

Given a context, identify the most accurate relation between two specified entities. There are two possible relation categories that the relation must belong to: A, B.

Please ensure that the identified relation follows the definitions provided below.

**Relation Definitions**:
```
Relation Definitions:
    1. A: Refer to two types of relationships between a chemical and a disease in CTD: 
            - Putative mechanistic relationships between a chemical and disease indicates that the chemical may play a role in the etiology of the disease (e.g., exposure to chemical X causes lung cancer).
            - Biomarker relations between a chemical and disease indicates that the chemical correlates with the disease (e.g., increased abundance in the brain of chemical X correlates with Alzheimer disease).
    2. B: This relation is used when the reaction does not fit into A category.
```
"""

########################################################
# (CoT- reasoning extraction) 給予 context，推導 reasoning
########################################################
bc5cdr_vanilla_ICL_prompt_user_reasoning = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.

Context: {context}

Q: Given the context, which relation is between "{head_entity}" and "{tail_entity}"?

A: Let's think step by step.
{reasoning}
{anwser_prefix}
"""

##################################################################
# (CoT - reasoning extraction) 給予 context+evidence，推導 reasoning
##################################################################
bc5cdr_S2PR_evidence_prompt_user_reasoning = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the evidence provided in the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.

Context: {context}

Evidence sentences in the context:
```
{evidence}
```

Q: Given the context and **focusing on the evidence sentences**, which relation is between "{head_entity}" and "{tail_entity}"?

A: Let's think step by step.
{reasoning}
{anwser_prefix}
"""



# ==================================================================================================== #
#                                              WebMD                                                  #
#                                              WebMD                                                  #
#                                              WebMD                                                  #
#                                              WebMD                                                  #
# ==================================================================================================== #

#########################################
# WebMD relation extraction system prompt
#########################################
webMD_prompt_system = """Task: Relation Extraction

Given a context, identify the most accurate relation between two specified entities. There are four possible relation categories that the relation must belong to: A, B, C, or D.

Please ensure that the identified relation follows the definitions and extraction rules provided below.

**Relation Definitions and Extraction Rules**:
```
Relation Definitions:
    1. A: This relation captures the connection between a drug and the improvement of an existing symptom that was already present in the patient before taking the drug.
    2. B: This relation captures the connection between a drug and the worsening or lack of improvement of an existing symptom that was already present in the patient before taking the drug. Specifically, the "lack of improvement" aspect applies only if the patient explicitly states that the drug was ineffective for addressing a particular symptom.
    3. C: This relation captures the connection between a drug and a new adverse reaction experienced by the patient after taking the drug.
    4. D: This relation is used when the reaction or symptom change does not fit into any of the above categories.

Extraction Rules:
    - Explicit Mentions: Only extract relations where the reaction or symptom change is explicitly mentioned as a result of taking the drug. Do not infer relationships that are not directly stated by the patient.
    - Context Consideration: Consider the context of the sentence to ensure the reaction or symptom change is indeed linked to the drug and not another potential cause.
        - Example: "I took Aspirin and later felt better, but I also rested a lot." In this case, the relationship may be ambiguous and should be noted as such.
    - Multiple Drugs and Reactions: Classify reactions resulting from the combination of multiple drugs as D, as it is unclear which drug caused the reaction or if the combination caused the reaction.
        - Example: "After taking Aspirin and Ibuprofen, I felt nauseous." In this case, classify as D due to the combination of drugs.
    - Non-specific Terms: Classify statements that do not clearly indicate a drug or a reaction as D.
        - Example: "I felt better after my medication." This statement is too vague to extract a specific relation, so classify as D.
    - No Assumptions About Improvement: Don’t automatically think a drug has helped a symptom just because it's meant to treat it. The patient needs to clearly say the drug made things better or that the symptom got better after taking it.
        - Example: "I was prescribed Fluoxetine for my anxiety." In this case, the statement does not explicitly say that Fluoxetine improved the patient's anxiety. It only mentions that the drug was prescribed for it, so no assumption should be made about the relationship between Fluoxetine and the improvement of anxiety, so classify as D.
    - Stick to the Context: Don’t rely on what you know or any outside info. Make your judgment only based on what’s actually said in the text.
```

Here are some examples:
{demonstrations}
"""

##########################################
# 給予 context 判斷 entity pair 的 relation
##########################################
webMD_vanilla_ICL_prompt_user = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.
MAKE SURE your output is one of the four categories stated (A,B,C,D).
Response the determined relation type (one of A, B, C, or D) only, nothing else.

Context: {context}

Given the context, the relation between "{head_entity}" and "{tail_entity}" is 
"""

#############################################################
# 給予 context + evidence sentence 判斷 entity pair 的 relation
#############################################################
webMD_S2PR_evidence_prompt_user = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the evidence provided in the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.
MAKE SURE your output is one of the four categories stated (A,B,C,D).
Response the determined relation type (one of A, B, C, or D) only, nothing else.

Context: {context}

Evidence sentences in the context:
```
{evidence}
```

Given the context and **focusing on the evidence sentences**, the relation between "{head_entity}" and "{tail_entity}" is """

########################################################
# 給予 context+ KG path 判斷 entity pair 的 relation
########################################################
webMD_S2PR_KG_path_prompt_user = """Now please solve this Relation Extraction task.
Note: Refer to the full text and focus on the provided context for determining the relation, and use the Knowledge Base Evidences as supplementary information to support your reasoning. Make sure to base your decision primarily on the context, using the Knowledge Base Evidences only as a reference.
MAKE SURE your output is one of the four categories stated (A,B,C,D).
Response the determined relation type (one of A, B, C, or D) only, nothing else.

Context: {context}

Knowledge Base Evidences:
```
{evidence}
```

Given the context and using the knowledge base evidences as supplementary information, the relation between "{head_entity}" and "{tail_entity}" is """


#######################
# (CoT) system prompt
#######################
webMD_prompt_system_reasoning = """Task: Relation Extraction

Given a context, identify the most accurate relation between two specified entities. There are four possible relation categories that the relation must belong to: A, B, C, or D.

Please ensure that the identified relation follows the definitions and extraction rules provided below.

**Relation Definitions and Extraction Rules**:
```
Relation Definitions:
    1. A: This relation captures the connection between a drug and the improvement of an existing symptom that was already present in the patient before taking the drug.
    2. B: This relation captures the connection between a drug and the worsening or lack of improvement of an existing symptom that was already present in the patient before taking the drug. Specifically, the "lack of improvement" aspect applies only if the patient explicitly states that the drug was ineffective for addressing a particular symptom.
    3. C: This relation captures the connection between a drug and a new adverse reaction experienced by the patient after taking the drug.
    4. D: This relation is used when the reaction or symptom change does not fit into any of the above categories.

Extraction Rules:
    - Explicit Mentions: Only extract relations where the reaction or symptom change is explicitly mentioned as a result of taking the drug. Do not infer relationships that are not directly stated by the patient.
    - Context Consideration: Consider the context of the sentence to ensure the reaction or symptom change is indeed linked to the drug and not another potential cause.
        - Example: "I took Aspirin and later felt better, but I also rested a lot." In this case, the relationship may be ambiguous and should be noted as such.
    - Multiple Drugs and Reactions: Classify reactions resulting from the combination of multiple drugs as D, as it is unclear which drug caused the reaction or if the combination caused the reaction.
        - Example: "After taking Aspirin and Ibuprofen, I felt nauseous." In this case, classify as D due to the combination of drugs.
    - Non-specific Terms: Classify statements that do not clearly indicate a drug or a reaction as D.
        - Example: "I felt better after my medication." This statement is too vague to extract a specific relation, so classify as D.
    - No Assumptions About Improvement: Don’t automatically think a drug has helped a symptom just because it's meant to treat it. The patient needs to clearly say the drug made things better or that the symptom got better after taking it.
        - Example: "I was prescribed Fluoxetine for my anxiety." In this case, the statement does not explicitly say that Fluoxetine improved the patient's anxiety. It only mentions that the drug was prescribed for it, so no assumption should be made about the relationship between Fluoxetine and the improvement of anxiety, so classify as D.
    - Stick to the Context: Don’t rely on what you know or any outside info. Make your judgment only based on what’s actually said in the text.
```
"""

########################################################
# (CoT- reasoning extraction) 給予 context，推導 reasoning
########################################################
webMD_vanilla_ICL_prompt_user_reasoning = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.

Context: {context}

Q: Given the context, which relation is between "{head_entity}" and "{tail_entity}"?

A: Let's think step by step.
{reasoning}
{anwser_prefix}
"""

##################################################################
# (CoT - reasoning extraction) 給予 context+evidence，推導 reasoning
##################################################################
webMD_S2PR_evidence_prompt_user_reasoning = """Now please solve this Relation Extraction task.
**Note:** Refer to the full text and focus on the evidence provided in the context. Do not use external knowledge or assumptions. Base your decision solely on the information given.

Context: {context}

Evidence sentences in the context:
```
{evidence}
```

Q: Given the context and **focusing on the evidence sentences**, which relation is between "{head_entity}" and "{tail_entity}"?

A: Let's think step by step.
{reasoning}
{anwser_prefix}
"""

##############################################################
# (CoT - reasoning extraction) 給予 context+path，推導 reasoning
##############################################################
webMD_S2PR_KG_path_prompt_user_reasoning = """Now please solve this Relation Extraction task.
Note: Refer to the full text and focus on the provided context for determining the relation, and use the Knowledge Base Evidences as supplementary information to support your reasoning. Make sure to base your decision primarily on the context, using the Knowledge Base Evidences only as a reference.

Context: {context}

Knowledge Base Evidences:
```
{evidence}
```

Q: Given the context and using the knowledge base evidences as supplementary information, which relation is between "{head_entity}" and "{tail_entity}"?

A: Let's think step by step.
{reasoning}
{anwser_prefix}
"""

# ==================================================================================================== #
#                                              Other                                                  #
#                                              Other                                                  #
#                                              Other                                                  #
#                                              Other                                                  #
# ==================================================================================================== #

##########
# 替換代名詞 
##########
replace_pronoun_prompt_system = """
def replace_pronouns_and_noun_phrases_in_text(input_text):
    \"\"\"
    **Task**: You are provided with a text that contains pronouns or noun phrases (such as "this medication") referring to specific nouns mentioned in the passage. Your task is to identify and replace each pronoun or noun phrase with its corresponding noun, ensuring that the meaning and coherence of the text are preserved.
    Make sure to replace all pronouns, including but not limited to "it," "they," "he," "she," "this," "that," "these," and "those," with their corresponding nouns. Similarly, replace noun phrases like "this medication" or "the drug" with the specific noun they refer to. Review the entire text carefully to avoid leaving any pronouns or noun phrases unreplaced.
    Be careful not to alter the original structure or content of the text, except for replacing the pronouns or noun phrases with the correct nouns they refer to.
    Ensure that the number of sentences after replacement matches the original text, i.e., the length of the related_sentence list must be the same as the length of the input_text list.
    \"\"\"

    # The entire text has been divided into sentences and stored in the list
    input_text = ["I was prescribed 150 mg. 4X daily by an endodotist in preparation for a root canal.", "Taking four doses daily felt like too many.", "Initially, I had some mild digestive complaints: acidity, bloating, burping, and hiccups.", "Two days into use, I developed an persistent nagging vise-like headache, not migraine, which I didn't at first link to the medication.", "The headache was accompanied by mental fogginess and some dizziness as well.", "The headache only stopped when I discontinued the antibiotic two days sooner than prescribed--after consulting a pharmacist.", "What ultimately made me stop was a marked rash on the trunk (which was noticeable on the third day into treatment and which subsequently spread to my arms and legs).", "The discomfort from the itching was intense.", "Benedryl has provided some minor relief.", "However, two days after my discontinuing clindamycin HCL, the rash persists and seems to have gained even more ground.", "As of yet, I'm glad to report that it is not on my face and neck--but I've got it pretty much everywhere else.", "Having read the posts by others, many of whom have had severe diarrhea--sometimes well after completing this medication, I am making a serious effort to use probiotics and yogurt to avoid c. difficile.", "If you must take this medication, I would strongly urge you to attend to and report side effects quickly.", "Do not believe that taking an antihistamine to deal with the itching will solve the problem--as I was lulled into believing.I wish I'd stopped using the drug earlier.", "I would also recommend that you request an alternate antibiotic.", "Previous experience with metronidazole proved to be far less problematic for me.", "I will not use this medication again as I understand that side effects can be much more serious at second exposure to the drug."]
    after_replace_sentence_list = []
    # Go through each sentence in input_text, replace all pronouns and noun phrases with their corresponding nouns, and then add the modified sentences to after_replace_sentence_list in the same order.

    after_replace_sentence_list.append({{"sentence_1": "I was prescribed 150 mg. 4X daily by an endodontist in preparation for a root canal."}})
    after_replace_sentence_list.append({{"sentence_2": "Taking four doses daily felt like too many."}})
    after_replace_sentence_list.append({{"sentence_3": "Initially, I had some mild digestive complaints: acidity, bloating, burping, and hiccups."}})
    after_replace_sentence_list.append({{"sentence_4": "Two days into use, I developed a persistent nagging vise-like headache, not migraine, which I didn't at first link to **clindamycin**."}})
    after_replace_sentence_list.append({{"sentence_5": "The headache was accompanied by mental fogginess and some dizziness as well."}})
    after_replace_sentence_list.append({{"sentence_6": "The headache only stopped when I discontinued the **clindamycin** two days sooner than prescribed--after consulting a pharmacist."}})
    after_replace_sentence_list.append({{"sentence_7": "What ultimately made me stop was a marked rash on the trunk (which was noticeable on the third day into treatment and which subsequently spread to my arms and legs)."}})
    after_replace_sentence_list.append({{"sentence_8": "The discomfort from the itching was intense."}})
    after_replace_sentence_list.append({{"sentence_9": "Benedryl has provided some minor relief."}})
    after_replace_sentence_list.append({{"sentence_10": "However, two days after my discontinuing clindamycin HCL, the rash persists and seems to have gained even more ground."}})
    after_replace_sentence_list.append({{"sentence_11": "As of yet, I am glad to report that the rash is not on my face and neck--but I have got **the rash** pretty much everywhere else."}})
    after_replace_sentence_list.append({{"sentence_12": "Having read the posts by others, many of whom have had severe diarrhea--sometimes well after completing **clindamycin**, I am making a serious effort to use probiotics and yogurt to avoid c. difficile."}})
    after_replace_sentence_list.append({{"sentence_13": "If you must take **clindamycin HCL**, I would strongly urge you to attend to and report side effects quickly."}})
    after_replace_sentence_list.append({{"sentence_14": "Do not believe that taking an antihistamine to deal with the itching will solve the problem--as I was lulled into believing. I wish I had stopped using **clindamycin** earlier."}})
    after_replace_sentence_list.append({{"sentence_15": "I would also recommend that you request an alternate antibiotic."}})
    after_replace_sentence_list.append({{"sentence_16": "Previous experience with metronidazole proved to be far less problematic for me."}})
    after_replace_sentence_list.append({{"sentence_17": "I will not use **clindamycin HCL** again as I understand that side effects can be much more serious at second exposure to **clindamycin**."}})
"""

replace_pronoun_prompt_user = """Please complete the following code like the example to complete replace pronouns and noun phrases. (Follow the example dict data format)
def replace_pronouns_and_noun_phrases_in_text(input_text):
    \"\"\"
    **Task**: You are provided with a text that contains pronouns or noun phrases (such as "this medication") referring to specific nouns mentioned in the passage. Your task is to identify and replace each pronoun or noun phrase with its corresponding noun, ensuring that the meaning and coherence of the text are preserved.
    Make sure to replace all pronouns, including but not limited to "it," "they," "he," "she," "this," "that," "these," and "those," with their corresponding nouns. Similarly, replace noun phrases like "this medication" or "the drug" with the specific noun they refer to. Review the entire text carefully to avoid leaving any pronouns or noun phrases unreplaced.
    Be careful not to alter the original structure or content of the text, except for replacing the pronouns or noun phrases with the correct nouns they refer to.
    Ensure that the number of sentences after replacement matches the original text, i.e., the length of the related_sentence list must be the same as the length of the input_text list.
    \"\"\"

    # The entire text has been divided into sentences and stored in the list
    input_text = {input_text}
    after_replace_sentence_list = []
    # Go through each sentence in input_text, replace all pronouns and noun phrases with their corresponding nouns, and then add the modified sentences to after_replace_sentence_list in the same order.


"""

#################################
# 以 natural language 重寫 KG path
#################################
rewrite_KG_path_prompt_system = """
There are some knowledge graph path. They follow entity->relationship->entity format.
\n\n
{Path}
\n\n
Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Knowledge Base Evidence 1, Knowledge Base Evidence 2,...\n\n

Output:
"""

rewrite_KG_path_prompt_user = "{text}"