import json
import re
import string

import datasets
import numpy as np
import itertools
import stanza

import nltk.corpus
from itertools import combinations

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=False, use_gpu=True)


def preprocess_name(name: str):
    if isinstance(name, str):
        name = name.replace('_', ' ')
        doc = nlp(name)
        processed_toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
        processed_name = " ".join(processed_toks)
        return processed_name, processed_toks


def preprocess_db(db):
    for table in db.tables:
        processed_name, processed_toks = preprocess_name(table.name[0])
        table.processed_name = processed_name
        table.processed_toks = processed_toks

        for column in table.columns:
            processed_name, processed_toks = preprocess_name(column.name[0])
            column.processed_name = processed_name
            column.processed_toks = processed_toks


def preprocess_question(question: str):
    """Tokenize,  lemmatize, lowercase question"""
    question = question.strip()
    doc = nlp(question)
    raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
    toks = [w.lemma.lower() for s in doc.sentences for w in s.words]

    # Compute question relations

    q_num, dtype = len(toks), '<U100'
    max_relative_dist = 2

    if q_num < max_relative_dist + 1:
        dist_vec = ['question-question_dist' + str(i) if i != 0 else 'question-question-identity'
                    for i in range(-max_relative_dist, max_relative_dist+1, 1)]
    else:
        dist_vec = ['question-question-generic'] * (q_num - max_relative_dist - 1) + \
            ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
              for i in range(-max_relative_dist, max_relative_dist+1, 1)] + \
            ['question-question-generic'] * (q_num - max_relative_dist - 1)
        starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)

        return {
            'raw_question_toks': raw_toks,
            'ori_toks': [w.text for s in doc.sentences for w in s.words],
            'processed_question_toks': toks,
            'relations': q_mat.tolist(),
        }


def rasat_schema_linking(question: str, db):
    ""

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    preprocess_db(db)
    question_preproc = preprocess_question(question)

    question_toks = question_preproc['processed_question_toks']
    table_toks = [table.processed_toks for table in db.tables]
    column_toks = [column.processed_toks for column in db.columns]
    table_names = [table.processed_name for table in db.tables]
    column_names = [column.processed_name for column in db.columns]
    q_num, t_num, c_num = len(question_toks), len(table_toks), len(column_toks)


    # relations between questions and tables
    q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype='<U100')
    max_len = max([len(t) for t in table_toks])
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        for idx, name in enumerate(table_names):
            if phrase == name:
                q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
            elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'

    # relations between questions and columns
    q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype='<U100')
    max_len = max([len(c) for c in column_toks if isinstance(c, list)])
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        for idx, name in enumerate(column_names):
            if name is not None:
                if phrase == name:
                    q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_col_mat[range(i, j), idx] = 'question-column-partialmatch'

    return {"q_col_match": q_col_mat, "q_tab_match": q_tab_mat}


def rasat_cell_linking(tokens, db):

    def is_number(word):
        """check if input is a number"""
        try:
            float(word)
            return True
        except:
            return False

    q_val_match = dict()
    num_date_match = dict()
    col_id2list = {col.id: col for col in db.columns}

    for q_id, token in enumerate(tokens):
        token = token.lower()

        # Skip token if it is in STOPWORDS
        if token in STOPWORDS:
            continue

        if is_number(token):
            token = str(float(token))

        for col_id, column in col_id2list.items():
            key = f'{q_id},{col_id}'
            col_values = column.cells
            for cell_value in col_values:
                if is_number(cell_value):
                    cell_value = str(float(cell_value))
                else:
                    cell_value = cell_value.lower()

                if token == cell_value:
                    q_val_match[key] = "CVM"
                    break

            # Add numerical/date relationships to num_date_match
            if column.type in ('number', 'real', 'time'):
                rel = 'NUMBER' if column.type == 'real' else column.type.upper()
                num_date_match[key] = rel

    return {"q_val_match": q_val_match, "num_date_match": num_date_match}


# schema linking, IRNet
def compute_schema_linking(tokens, db):

    def partial_match(x_list, y_list):
        x_str = "".join(y_list)
        y_str = "".join(x_list)

        if x_str in STOPWORDS:
            return False

        if re.match("%s" % re.escape(x_str), y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        """check exact match"""
        x, y = x_list, y_list
        if isinstance(x, list):
            x = "".join(x)
        if isinstance(y, list):
            y = "".join(y)
        return x == y

    def ngrams(tok_list, n):
        """generate n-grams from tok_list
        Args:
            tok_list (TYPE): NULL
            n (TYPE): NULL
        Returns: TODO
        Raises: NULL
        """
        for pos in range(len(tok_list) - n + 1):
            yield tok_list[pos:pos + n]

    def set_q_relation(q_match_dict,
                       q_start,
                       q_match_len,
                       other_id,
                       relation_tag,
                       force=True):
        """set match relation for question
        """

        for q_id in range(q_start, q_start + q_match_len):
            key = f"{q_id},{other_id}"
            if not force and key in q_match_dict:
                continue
            q_match_dict[key] = relation_tag

    columns = [x.name for x in db.columns]
    tables = [x.name for x in db.tables]

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(columns):
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(tables):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = 5
    while n > 0:
        for i, n_gram_list in enumerate(ngrams(tokens, n)):
            if len("".join(n_gram_list).strip()) == 0:
                continue
            # exact match case
            for col_id, col in col_id2list.items():
                if exact_match(n_gram_list, col):
                    set_q_relation(q_col_match, i, n, col_id, "CEM")
            for tab_id, tab in tab_id2list.items():
                if exact_match(n_gram_list, tab):
                    set_q_relation(q_tab_match, i, n, tab_id, "TEM")

            # partial match case
            for col_id, col in col_id2list.items():
                if partial_match(n_gram_list, col):
                    set_q_relation(
                        q_col_match, i, n, col_id, "CPM", force=False)
            for tab_id, tab in tab_id2list.items():
                if partial_match(n_gram_list, tab):
                    set_q_relation(
                        q_tab_match, i, n, tab_id, "TEM", force=False)
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, db):
    """cell-value linking
    """

    def isnumber(word):
        """check if input is a number"""
        try:
            float(word)
            return True
        except:
            return False

    def check_cell_match(word, cells):
        """check if word partial/exact match one of values
        """
        for cell in cells:
            if word in cell:
                return True
        return False

    num_date_match = {}
    cell_match = {}

    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS:
            continue

        num_flag = isnumber(word)
        for col_id, column in enumerate(db.columns):
            # word is number
            if num_flag:
                if column.type in ("number", "real", "time"
                                    ):  # TODO fine-grained date
                    rel = 'NUMBER' if column.type == 'real' else column.type.upper(
                    )
                    num_date_match[f"{q_id},{col_id}"] = rel
            elif column.type.lower(
            ) == 'binary':  # binary condition should use special process
                continue
            elif check_cell_match(word, column.cells):
                cell_match[f"{q_id},{col_id}"] = "CELLMATCH"

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link


def _table_id(db, col):
    if col == 0:
        return None
    else:
        return db.columns[col].table.id


def _foreign_key_id(db, col):
    foreign_col = db.columns[col].foreign_key_for
    if foreign_col is None:
        return None
    return foreign_col.id


def _match_foreign_key(db, col, table):
    foreign_key_id = _foreign_key_id(db, col)
    if foreign_key_id is None:
        return None
    return table == _table_id(db, foreign_key_id)


def clamp(value, abs_max):
    """clamp value"""
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


def convert_tokens_to_question_format(tokens, cell_linking, schema_linking):
    token_mapping = {}

    # Add cell_linking values to the token_mapping
    for key, value in cell_linking['q_val_match'].items():
        r, c = map(int, key.split(','))
        token_mapping[f'question_{r}_{c}'] = value.lower()

    for key, value in cell_linking['num_date_match'].items():
        r, c = map(int, key.split(','))
        token_mapping[f'question_{r}_{c}'] = value.lower()

    # Add schema_linking values to the token_mapping
    for r in range(schema_linking['q_col_match'].shape[0]):
        for c in range(schema_linking['q_col_match'].shape[1]):
            if schema_linking['q_col_match'][r, c] != 'question-column-nomatch':
                token_mapping[f'question_{r}_{c}'] = schema_linking['q_col_match'][r, c].lower()

    for r in range(schema_linking['q_tab_match'].shape[0]):
        for c in range(schema_linking['q_tab_match'].shape[1]):
            if schema_linking['q_tab_match'][r, c] != 'question-table-nomatch':
                token_mapping[f'question_{r}_{c}'] = schema_linking['q_tab_match'][r, c].lower()

    # Replace tokens in the input list with their corresponding question_r_c format
    converted_tokens = [token_mapping.get(token, token) for token in tokens]

    return converted_tokens


def build_relation_matrix(cell_links, schema_links, tokens):
    n_row, n_col, n_tok = len(schema_links['q_col_match']), len(schema_links['q_col_match'][0]), len(tokens)
    relation_matrix = np.zeros((n_tok, n_row * n_col), dtype=np.int64)

    # For cell linking
    for key, value in cell_links['q_val_match'].items():
        r, c = map(int, key.split(','))
        relation_matrix[r, r * n_col + c] = 1

    for key, value in cell_links['num_date_match'].items():
        r, c = map(int, key.split(','))
        relation_matrix[r, r * n_col + c] = 1

    for r in range(schema_links['q_col_match'].shape[0]):
        for c in range(schema_links['q_col_match'].shape[1]):
            match_type = schema_links['q_col_match'][r, c]
            if match_type in {'question-column-exactmatch', 'question-column-partialmatch'}:
                relation_matrix[r, r * n_col + c] = 1

    for r in range(schema_links['q_tab_match'].shape[0]):
        for c in range(schema_links['q_tab_match'].shape[1]):
            match_type = schema_links['q_tab_match'][r, c]
            if match_type in {'question-table-exactmatch', 'question-table-partialmatch'}:
                for col_idx in range(n_col):
                    relation_matrix[r, r * n_col + col_idx] = 1

    return relation_matrix


if __name__ == '__main__':

    from settings import DATASETS_PATH
    from dataproc import utils
    from pathlib import Path

    train_ds = datasets.load_dataset(path="../dataproc/loaders/spider.py", cache_dir=DATASETS_PATH, split='train')
    train_spider_file = Path(train_ds[0]['data_filepath'])
    dbs = utils.process(train_ds)
    with open(train_spider_file) as data_file:
        spider_json = json.load(data_file)

    for idx, sample in enumerate(train_ds):
        db = dbs[sample['db_id']]
        question = sample['question']
        question_toks = question.split()
        rasat_schema = rasat_schema_linking(question, db)
        normal_schema = compute_schema_linking(question_toks, db)
        rasat_cell = rasat_cell_linking(question_toks, db)
        normal_cell = compute_cell_value_linking(question_toks, db)
        relation_matrix = build_relation_matrix(rasat_cell, rasat_schema, question_toks)


