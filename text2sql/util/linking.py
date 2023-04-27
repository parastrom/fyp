import json
import re
import string

import datasets
import numpy as np
import itertools
import stanza
import networkx as nx
import matplotlib.pyplot as plt

import nltk.corpus
from itertools import combinations

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=False, use_gpu=True)


def preprocess_name(name: str):
    if isinstance(name, str):
        name = name.replace('_', ' ')
        doc = nlp(name)
        processed_toks = []
        for s in doc.sentences:
            for w in s.words:
                try:
                    processed_toks.append(w.lemma.lower())
                except AttributeError:
                    pass
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

def create_dependency_graph(sentences):
    G = nx.Graph()
    global_word_index = 1  # Start with 1 for the global index
    for sentence in sentences:
        for word in sentence.words:
            # Add edges using the global_word_index instead of word.id
            head_index = int(word.head) + global_word_index - int(word.id) if word.head != '0' else 0
            G.add_edge(global_word_index, head_index, relation=word.deprel)
            global_word_index += 1
    return G


def preprocess_question(question: str):

    """Tokenize,  lemmatize, lowercase question"""
    try:
        question = question.strip()
        doc = nlp(question)
        raw_toks = []
        toks = []
        for s in doc.sentences:
            for w in s.words:
                try:
                    raw_toks.append(w.text.lower())
                    toks.append(w.lemma.lower())
                except AttributeError:
                    pass
    except Exception as e:
        print(question)
        print(f"An exception occurred: {e}")
        exit(-1)

    # Create graph for depedency relations
    graph = create_dependency_graph(doc.sentences)
    # nx.draw_networkx(graph, with_labels=True)
    # plt.savefig('plotgraph.png', format="PNG")
    # plt.show()
    # Compute question relations

    q_num, dtype = len(toks), int
    max_relative_dist = 2

    q_mat = np.zeros((q_num, q_num), dtype=dtype)

    for id1 in range(1, q_num + 1):
        for id2 in range(1, q_num + 1):
            if id1 != id2:
                shortest_path_length = nx.shortest_path_length(graph, id1, id2)
                q_mat[id1 - 1, id2 - 1] = shortest_path_length

    return {
        'raw_question_toks': raw_toks,
        'ori_toks': [w.text for s in doc.sentences for w in s.words],
        'processed_question_toks': toks,
        'relations': q_mat,
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
    tab_q_mat = np.array([['table-question-nomatch'] * q_num for _ in range(t_num)], dtype='<U100')
    max_len = max([len(t) for t in table_toks])
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        for idx, name in enumerate(table_names):
            if phrase == name:
                q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
                tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
            elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
                tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'

    # relations between questions and columns
    q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype='<U100')
    col_q_mat = np.array([['column-question-nomatch'] * q_num for _ in range(c_num)], dtype='<U100')
    max_len = max([len(c) for c in column_toks if isinstance(c, list)])
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        for idx, name in enumerate(column_names):
            if name is not None:
                if phrase == name:
                    q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-partialmatch'

    return {"q_col_match": q_col_mat, "q_tab_match": q_tab_mat, "col_q_match": col_q_mat, "tab_q_match": tab_q_mat}


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

class Relations(object):
    """Docstring for Relations. """

    def __init__(self,
                 qq_max_dist=2,
                 cc_foreign_key=True,
                 cc_table_match=True,
                 cc_max_dist=2,
                 ct_foreign_key=True,
                 ct_table_match=True,
                 tc_table_match=True,
                 tc_foreign_key=True,
                 tt_max_dist=2,
                 tt_foreign_key=True,
                 merge_types=False,
                 sc_link=True,
                 cv_link=True):
        super(Relations, self).__init__()

        self.qq_max_dist = qq_max_dist
        self.cc_foreign_key = cc_foreign_key
        self.cc_table_match = cc_table_match
        self.cc_max_dist = cc_max_dist
        self.ct_foreign_key = ct_foreign_key
        self.ct_table_match = ct_table_match
        self.tc_table_match = tc_table_match
        self.tc_foreign_key = tc_foreign_key
        self.tt_max_dist = tt_max_dist
        self.tt_foreign_key = tt_foreign_key
        self.merge_types = merge_types
        self.sc_link = sc_link
        self.cv_link = cv_link

        self.relation_ids = {}

        def add_relation(name):
            self.relation_ids[name] = len(self.relation_ids)

        ##< TODO: add_relation('[UNK]')

        def add_rel_dist(name, max_dist):
            for i in range(-max_dist, max_dist + 1):
                add_relation((name, i))

        add_rel_dist('qq_dist', qq_max_dist)

        add_relation('qc_default')
        # if qc_token_match:
        #    add_relation('qc_token_match')

        add_relation('qt_default')
        # if qt_token_match:
        #    add_relation('qt_token_match')

        add_relation('cq_default')
        # if cq_token_match:
        #    add_relation('cq_token_match')

        add_relation('cc_default')
        if cc_foreign_key:
            add_relation('cc_foreign_key_forward')
            add_relation('cc_foreign_key_backward')
        if cc_table_match:
            add_relation('cc_table_match')
        add_rel_dist('cc_dist', cc_max_dist)

        add_relation('ct_default')
        if ct_foreign_key:
            add_relation('ct_foreign_key')
        if ct_table_match:
            add_relation('ct_primary_key')
            add_relation('ct_table_match')
            add_relation('ct_any_table')

        add_relation('tq_default')
        # if cq_token_match:
        #    add_relation('tq_token_match')

        add_relation('tc_default')
        if tc_table_match:
            add_relation('tc_primary_key')
            add_relation('tc_table_match')
            add_relation('tc_any_table')
        if tc_foreign_key:
            add_relation('tc_foreign_key')

        add_relation('tt_default')
        if tt_foreign_key:
            add_relation('tt_foreign_key_forward')
            add_relation('tt_foreign_key_backward')
            add_relation('tt_foreign_key_both')
        add_rel_dist('tt_dist', tt_max_dist)

        # schema linking relations
        # forward_backward
        if sc_link:
            add_relation('qcCEM')
            add_relation('cqCEM')
            add_relation('qtTEM')
            add_relation('tqTEM')
            add_relation('qcCPM')
            add_relation('cqCPM')
            add_relation('qtTPM')
            add_relation('tqTPM')

        if cv_link:
            add_relation("qcNUMBER")
            add_relation("cqNUMBER")
            add_relation("qcTIME")
            add_relation("cqTIME")
            add_relation("qcCELLMATCH")
            add_relation("cqCELLMATCH")

        if merge_types:
            assert not cc_foreign_key
            assert not cc_table_match
            assert not ct_foreign_key
            assert not ct_table_match
            assert not tc_foreign_key
            assert not tc_table_match
            assert not tt_foreign_key

            assert cc_max_dist == qq_max_dist
            assert tt_max_dist == qq_max_dist

            add_relation('xx_default')
            self.relation_ids['qc_default'] = self.relation_ids['xx_default']
            self.relation_ids['qt_default'] = self.relation_ids['xx_default']
            self.relation_ids['cq_default'] = self.relation_ids['xx_default']
            self.relation_ids['cc_default'] = self.relation_ids['xx_default']
            self.relation_ids['ct_default'] = self.relation_ids['xx_default']
            self.relation_ids['tq_default'] = self.relation_ids['xx_default']
            self.relation_ids['tc_default'] = self.relation_ids['xx_default']
            self.relation_ids['tt_default'] = self.relation_ids['xx_default']

            if sc_link:
                self.relation_ids['qcCEM'] = self.relation_ids['xx_default']
                self.relation_ids['qcCPM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTEM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTPM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCEM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCPM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTEM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTPM'] = self.relation_ids['xx_default']
            if cv_link:
                self.relation_ids["qcNUMBER"] = self.relation_ids['xx_default']
                self.relation_ids["cqNUMBER"] = self.relation_ids['xx_default']
                self.relation_ids["qcTIME"] = self.relation_ids['xx_default']
                self.relation_ids["cqTIME"] = self.relation_ids['xx_default']
                self.relation_ids["qcCELLMATCH"] = self.relation_ids[
                    'xx_default']
                self.relation_ids["cqCELLMATCH"] = self.relation_ids[
                    'xx_default']

            for i in range(-qq_max_dist, qq_max_dist + 1):
                self.relation_ids['cc_dist', i] = self.relation_ids['qq_dist',
                                                                    i]
                self.relation_ids['tt_dist', i] = self.relation_ids['tt_dist',
                                                                    i]

        print("relations num is: %d", len(self.relation_ids))

    def __len__(self):
        """size of relations
        Returns: int
        """
        return len(self.relation_ids)


RELATIONS = Relations()

# Maintaining legacy functions so that dill works
def normal_build_relation_matrix():
    pass


def compute_schema_linking():
    pass


def compute_cell_value_linking():
    pass


def _table_id(db, col):
    if col == 0:
        return None
    else:
        return db.columns[col].table.id


def _foreign_key_id(db, col):
    foreign_col = db.columns[col].foreign_key
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
            if match_type == 'question-column-partialmatch':
                relation_matrix[r, r * n_col + c] = 1
            elif match_type == 'question-column-exactmatch':
                relation_matrix[r, r * n_col + c] = 2

    for r in range(schema_links['q_tab_match'].shape[0]):
        for c in range(schema_links['q_tab_match'].shape[1]):
            match_type = schema_links['q_tab_match'][r, c]
            if match_type == 'question-column-partialmatch':
                for col_idx in range(n_col):
                    relation_matrix[r, r * n_col + col_idx] = 1
            elif match_type == 'question-table-exactmatch':
                for col_idx in range(n_col):
                    relation_matrix[r, r * n_col + col_idx] = 2

    return relation_matrix


def new_build_relational_matrix(cell_links, schema_links, db, n_tok):
    c_len = len(db.columns)
    t_len = len(db.tables)
    total_len = n_tok + c_len + t_len
    relation_matrix = np.zeros((total_len, total_len), dtype=np.int64)

    # Helper functions
    def _table_id(col_id):
        if db.columns[col_id].table:
            return db.columns[col_id].table.id
        else:
            return None

    def _foreign_key_id(col_id):
        return db.columns[col_id].foreign_key

    def _match_foreign_key(col_id, table_id):
        foreign_key = _foreign_key_id(col_id)
        return foreign_key is not None and foreign_key == table_id


    def _get_type(idx):
        if idx < n_tok:
            return ('question', idx)
        elif n_tok <= idx < n_tok + c_len:
            return ('column', idx - n_tok)
        else:
            return ('table', idx - n_tok - c_len)

    def clamp(value, abs_max):
        """clamp value"""
        value = max(-abs_max, value)
        value = min(abs_max, value)
        return value

    # Process cell_links and schema_links
    for key, value in cell_links['q_val_match'].items():
        r, c = map(int, key.split(','))
        relation_matrix[r, n_tok + c] = RELATIONS.relation_ids['qcCELLMATCH']

    for key, value in cell_links['num_date_match'].items():
        r, c = map(int, key.split(','))
        relation_matrix[r, n_tok + c] = RELATIONS.relation_ids['qcNUMBER']

    for key, value in schema_links['q_col_match'].items():
        r, c = map(int, key.split(','))
        if value == 'question-column-partialmatch':
            relation_matrix[r, n_tok + c] = RELATIONS.relation_ids['qcCPM']
        elif value == 'question-column-exactmatch':
            relation_matrix[r, n_tok + c] = RELATIONS.relation_ids['qcCEM']

    for key, value in schema_links['q_tab_match'].items():
        r, c = map(int, key.split(','))
        if value == 'question-column-partialmatch':
            for col_idx in range(c_len):
                relation_matrix[r, n_tok + col_idx] = RELATIONS.relation_ids['qtTPM']
        elif value == 'question-table-exactmatch':
            for col_idx in range(c_len):
                relation_matrix[r, n_tok + col_idx] = RELATIONS.relation_ids['qtTEM']

    # Capture schema-schema relations
    for i in range(n_tok, total_len):
        i_type = _get_type(i)
        for j in range(n_tok, total_len):
            j_type = _get_type(j)

            if i_type[0] == 'column' and j_type[0] == 'column':
                col1, col2 = i_type[1], j_type[1]
                if col1 == col2:
                    relation_matrix[i, j] = RELATIONS.relation_ids['cc_default']
                else:
                    if _foreign_key_id(col1) == col2:
                        relation_matrix[i, j] = RELATIONS.relation_ids['cc_foreign_key_forward']
                    if _foreign_key_id(col2) == col1:
                        relation_matrix[i, j] = RELATIONS.relation_ids['cc_foreign_key_backward']
                    if _table_id(col1) == _table_id(col2):
                        relation_matrix[i, j] = RELATIONS.relation_ids['cc_table_match']

            elif i_type[0] == 'column' and j_type[0] == 'table':
                col, table = i_type[1], j_type[1]
                if _match_foreign_key(col, table):
                    relation_matrix[i, j] = RELATIONS.relation_ids['ct_foreign_key']
                if _table_id(col) == table:
                    relation_matrix[i, j] = RELATIONS.relation_ids['ct_table_match']

            elif i_type[0] == 'table' and j_type[0] == 'column':
                table, col = i_type[1], j_type[1]
                if _match_foreign_key(col, table):
                    relation_matrix[i, j] = RELATIONS.relation_ids['tc_foreign_key']
                if _table_id(col) == table:
                    relation_matrix[i, j] = RELATIONS.relation_ids['tc_table_match']

            elif i_type[0] == 'table' and j_type[0] == 'table':
                table1, table2 = i_type[1], j_type[1]
                if table1 == table2:
                    relation_matrix[i, j] = RELATIONS.relation_ids['tt_default']
                else:
                    if table2 in db.tables[table1].foreign_key_tables:
                        relation_matrix[i, j] = RELATIONS.relation_ids['tt_foreign_key_forward']
                    if table1 in db.tables[table2].foreign_key_tables:
                        relation_matrix[i, j] = RELATIONS.relation_ids['tt_foreign_key_backward']

    # Capture question-question relations
    for i in range(n_tok):
        i_type = _get_type(i)
        for j in range(n_tok):
            j_type = _get_type(j)

            if i_type[0] == 'question' and j_type[0] == 'question':
                relation_matrix[i, j] = RELATIONS.relation_ids['qq_dist', clamp(j - i, RELATIONS.qq_max_dist)]

    # Capture schema-question relations
    for key, value in schema_links['col_q_match'].items():
        c, r = map(int, key.split(','))
        if value == 'column-question-partialmatch':
            relation_matrix[n_tok + c, r] = RELATIONS.relation_ids['cqCPM']
        elif value == 'column-question-exactmatch':
            relation_matrix[n_tok + c, r] = RELATIONS.relation_ids['cqCEM']

    for key, value in schema_links['tab_q_match'].items():
        t, r = map(int, key.split(','))
        if value == 'table-question-partialmatch':
            for col_idx in range(c_len):
                if db.columns[col_idx].table and db.columns[col_idx].table.id == t:
                    relation_matrix[n_tok + col_idx, r] = RELATIONS.relation_ids['tqTPM']
        elif value == 'table-question-exactmatch':
            for col_idx in range(c_len):
                if db.columns[col_idx].table and db.columns[col_idx].table.id == t:
                    relation_matrix[n_tok + col_idx, r] = RELATIONS.relation_ids['tqTEM']

    return relation_matrix


if __name__ == '__main__':

    from settings import DATASETS_PATH
    from dataproc import spider_dataset
    from pathlib import Path

    train_ds = datasets.load_dataset(path="../dataproc/loaders/spider.py", cache_dir=DATASETS_PATH, split='train')
    train_spider_file = Path(train_ds[0]['data_filepath'])
    dbs = spider_dataset.process(train_ds)
    with open(train_spider_file) as data_file:
        spider_json = json.load(data_file)

    for idx, sample in enumerate(train_ds):
        db = dbs[sample['db_id']]
        question = sample['question']
        question_toks = question.split()
        rasat_schema = rasat_schema_linking(question, db)
        rasat_cell = rasat_cell_linking(question_toks, db)
        relation_matrix = build_relation_matrix(rasat_cell, rasat_schema, question_toks)


