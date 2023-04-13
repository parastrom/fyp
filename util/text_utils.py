import nltk


def wordseg_and_extract_nums(question):

    tokens = nltk.word_tokenize(question)
    candi_nums = []
    candi_nums_index = []

    for idx, token in enumerate(tokens):
        if token.isdigit() or token.replace(".", "").isdigit():
            candi_nums.append(token)
            candi_nums_index.append(idx)

    return tokens, candi_nums_index, candi_nums_index

