import spacy
nlp = spacy.load("en_core_web_sm")

def vec_to_tags(tags, vecs, max_seq_len=256):
    """
    change vector to tags
    """
    idx_to_tag = {key: idx for key, idx in enumerate(tags)}
    print(idx_to_tag)
    tags = []

    for vec in vecs:
        tag = [idx_to_tag.get(idx) for idx in vec[:max_seq_len]]
        tags.append(tag)

    return tags 

def sentences_to_vec(sentence, d_word_id, max_len, max_seq_len=256):
    sent = nlp(sentence)
    vec = [d_word_id.get(str(word), max_len) for word in sent[:max_seq_len]]
    return vec + [0] * (max_seq_len - len(vec))

def tags_to_vec(tags, d_tags_id, max_seq_len=256):
    vec = [d_tags_id.get(tag) for tag in tags[:max_seq_len]]
    return vec + [0] * (max_seq_len - len(vec))

def sentences_to_vec_nopad(sentence, d_word_id, max_len, max_seq_len=256):
    sent = nlp(sentence)
    vec = [d_word_id.get(str(word), max_len) for word in sent[:max_seq_len]]
    return vec

def tags_to_vec_nopad(tags, d_tags_id, max_seq_len=256):
    vec = [d_tags_id.get(tag) for tag in tags[:max_seq_len]]
    return vec