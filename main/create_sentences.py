# %%
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
# %%

document = """Severe infection resulting in 3 root canals. Currently taking 300mg of clindamycin every 6 hours. Some nausea and diarrhea with use. I am on day 3 of 10. I am also taking hydrocodone 5/325 every 6 hours for pain. Increased dizziness and sleepiness with use. I am allergic to meds in penicillin family, so clindamycin is the go to.  I was also given a iv bag of the antibiotic vancomycin before dental surgery."""

mention_list = ['vancomycin', 'penicillin', 'clindamycin', 'hydrocodone', 'diarrhea', 'Increased dizziness', 'Severe infection', 'nausea', 'pain', 'sleepiness']

def get_sentences(document):
    sentences = sent_tokenize(document)
    return sentences

def get_vertex_set(mention_list, sentences):
    vertexSet = []

    # Iterate over mentions and find their positions
    for mention in mention_list:
        mention_info = []
        for sent_id, sent in enumerate(sentences):
            # Find all occurrences of the mention in the sentence
            start = 0
            while start < len(sent):
                start = sent.find(mention, start)
                if start == -1:
                    break
                end = start + len(mention)
                mention_info.append({
                    "name": mention,
                    "sent_id": sent_id,
                    "pos": [start, end]
                })
                start = end  # Move past this mention for next search

        # 斷句問題可能導致找不到
        # if (mention_info == []):
        #     # print(mention)
        #     # print(sentences)
        vertexSet.append(mention_info)
        # print(json.dumps(vertexSet, indent=4))
    return vertexSet
