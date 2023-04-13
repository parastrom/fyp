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
            print('relation: %s --> %d', name, self.relation_ids[name])

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


def normal_build_relation_matrix(other_links, total_length, q_length, c_length,
                          c_boundaries, t_boundaries, db):
    """build relation matrix
    """
    sc_link = other_links.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
    cv_link = other_links.get('cv_link',
                              {'num_date_match': {},
                               'cell_match': {}})

    # Catalogue which things are where
    loc_types = {}
    for i in range(q_length):
        loc_types[i] = ('question', )

    c_base = q_length
    for c_id, (c_start,
               c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
        for i in range(c_start + c_base, c_end + c_base):
            loc_types[i] = ('column', c_id)
    t_base = q_length + c_length
    for t_id, (t_start,
               t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
        for i in range(t_start + t_base, t_end + t_base):
            loc_types[i] = ('table', t_id)

    relations = np.zeros((total_length, total_length), dtype=np.int64)
    for i, j in itertools.product(range(total_length), repeat=2):

        def _set_relation(name):
            """set relation for position (i, j)"""
            relations[i, j] = RELATIONS.relation_ids[name]

        def _get_qc_links(q_id, c_id):
            """get link relation of q and col"""
            coord = "%d,%d" % (q_id, c_id)
            if coord in sc_link["q_col_match"]:
                return sc_link["q_col_match"][coord]
            elif coord in cv_link["cell_match"]:
                return cv_link["cell_match"][coord]
            elif coord in cv_link["num_date_match"]:
                return cv_link["num_date_match"][coord]
            return '_default'

        def _get_qt_links(q_id, c_id):
            """get link relation of q and tab"""
            coord = "%d,%d" % (q_id, c_id)
            if coord in sc_link["q_tab_match"]:
                return sc_link["q_tab_match"][coord]
            else:
                return '_default'

        try:
            i_type, j_type = loc_types[i], loc_types[j]
        except Exception as e:
            print(f'loc_types: {loc_types}. c_boundaries: {c_boundaries}.' + \
                          f'i, j, total_length and q_length: {i}, {j}, {total_length}, {q_length}')
            raise e

        if i_type[0] == 'question':
            ################ relation of question-to-* ####################
            if j_type[0] == 'question':  # relation qq
                _set_relation(('qq_dist', clamp(j - i, RELATIONS.qq_max_dist)))
            elif j_type[0] == 'column':  # relation qc
                j_real = j_type[1]
                rel = _get_qc_links(i, j_real)
                _set_relation('qc' + rel)
            elif j_type[0] == 'table':  # relation qt
                j_real = j_type[1]
                rel = _get_qt_links(i, j_real)
                _set_relation('qt' + rel)
        elif i_type[0] == 'column':
            ################ relation of column-to-* ####################
            if j_type[0] == 'question':  ## relation cq
                i_real = i_type[1]
                rel = _get_qc_links(j, i_real)
                _set_relation('cq' + rel)
            elif j_type[0] == 'column':  ## relation cc
                col1, col2 = i_type[1], j_type[1]
                if col1 == col2:
                    _set_relation(
                        ('cc_dist', clamp(j - i, RELATIONS.cc_max_dist)))
                else:
                    _set_relation('cc_default')
                    # TODO: foreign keys and table match
                    if RELATIONS.cc_foreign_key:
                        if _foreign_key_id(db, col1) == col2:
                            _set_relation('cc_foreign_key_forward')
                        if _foreign_key_id(db, col2) == col1:
                            _set_relation('cc_foreign_key_backward')
                    if (RELATIONS.cc_table_match and
                            _table_id(db, col1) == _table_id(db, col2)):
                        _set_relation('cc_table_match')
            elif j_type[0] == 'table':  ## relation ct
                col, table = i_type[1], j_type[1]
                _set_relation('ct_default')
                if RELATIONS.ct_foreign_key and _match_foreign_key(db, col,
                                                                   table):
                    _set_relation('ct_foreign_key')
                if RELATIONS.ct_table_match:
                    col_table = _table_id(db, col)
                    if col_table == table:
                        if col in db.columns[col].table.primary_keys_id:
                            _set_relation('ct_primary_key')
                        else:
                            _set_relation('ct_table_match')
                    elif col_table is None:
                        _set_relation('ct_any_table')
        elif i_type[0] == 'table':
            ################ relation of table-to-* ####################
            if j_type[0] == 'question':
                i_real = i_type[1]
                rel = _get_qt_links(j, i_real)
                _set_relation('tq' + rel)
            elif j_type[0] == 'column':
                table, col = i_type[1], j_type[1]
                _set_relation('tc_default')

                if RELATIONS.tc_foreign_key and _match_foreign_key(db, col,
                                                                   table):
                    _set_relation('tc_foreign_key')
                if RELATIONS.tc_table_match:
                    col_table = _table_id(db, col)
                    if col_table == table:
                        if col in db.columns[col].table.primary_keys_id:
                            _set_relation('tc_primary_key')
                        else:
                            _set_relation('tc_table_match')
                    elif col_table is None:
                        _set_relation('tc_any_table')
            elif j_type[0] == 'table':
                table1, table2 = i_type[1], j_type[1]
                if table1 == table2:
                    _set_relation(
                        ('tt_dist', clamp(j - i, RELATIONS.tt_max_dist)))
                else:
                    _set_relation('tt_default')
                    if RELATIONS.tt_foreign_key:
                        forward = table2 in db.tables[
                            table1].foreign_key_tables
                        backward = table1 in db.tables[
                            table2].foreign_key_tables
                        if forward and backward:
                            _set_relation('tt_foreign_key_both')
                        elif forward:
                            _set_relation('tt_foreign_key_forward')
                        elif backward:
                            _set_relation('tt_foreign_key_backward')

    return relations


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
    n_row, n_col = len(schema_links['q_col_match']), len(schema_links['q_col_match'][0])
    relation_matrix = np.zeros((n_tok, n_row * n_col), dtype=np.int64)

    # Helper functions
    def _table_id(col_id):
        return db.columns[col_id].table_id

    def _foreign_key_id(col_id):
        return db.columns[col_id].foreign_key_for

    def _match_foreign_key(col_id, table_id):
        foreign_key = _foreign_key_id(col_id)
        return foreign_key is not None and foreign_key == table_id

    # For cell linking
    for r, c in cell_links['q_val_match']:
        relation_matrix[r, r * n_col + c] = RELATIONS.relation_ids['qcCELLMATCH']

    for r, c in cell_links['num_date_match']:
        relation_matrix[r, r * n_col + c] = RELATIONS.relation_ids['qcNUMBER']

    for r in range(schema_links['q_col_match'].shape[0]):
        for c in range(schema_links['q_col_match'].shape[1]):
            match_type = schema_links['q_col_match'][r, c]
            if match_type == 'question-column-partialmatch':
                relation_matrix[r, r * n_col + c] = RELATIONS.relation_ids['qcCPM']
            elif match_type == 'question-column-exactmatch':
                relation_matrix[r, r * n_col + c] = RELATIONS.relation_ids['qcCEM']

    for r in range(schema_links['q_tab_match'].shape[0]):
        for c in range(schema_links['q_tab_match'].shape[1]):
            match_type = schema_links['q_tab_match'][r, c]
            if match_type == 'question-column-partialmatch':
                for col_idx in range(n_col):
                    relation_matrix[r, r * n_col + col_idx] = RELATIONS.relation_ids['qtTPM']
            elif match_type == 'question-table-exactmatch':
                for col_idx in range(n_col):
                    relation_matrix[r, r * n_col + col_idx] = RELATIONS.relation_ids['qtTEM']

    # Consider foreign keys and table matches
    for r in range(n_tok):
        for c in range(n_col):
            col_id = r * n_col + c
            if schema_links['q_col_match'][r, c] in ('question-column-partialmatch', 'question-column-exactmatch'):
                table_id = _table_id(col_id)
                for idx, table_id in enumerate(db.table_ids):
                    if table_id != _table_id(col_id):
                        if _match_foreign_key(col_id, table_id):
                            relation_matrix[r, r * n_col + idx] = RELATIONS.relation_ids['cc_foreign_key_forward']
                        elif _match_foreign_key(idx, table_id):
                            relation_matrix[r, r * n_col + idx] = RELATIONS.relation_ids['cc_foreign_key_backward']
                    else:
                        relation_matrix[r, r * n_col + idx] = RELATIONS.relation_ids['cc_table_match']

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
        normal_schema = compute_schema_linking(question_toks, db)
        rasat_cell = rasat_cell_linking(question_toks, db)
        normal_cell = compute_cell_value_linking(question_toks, db)
        relation_matrix = build_relation_matrix(rasat_cell, rasat_schema, question_toks)
        link_info_dict = {
            'sc_link': normal_schema,
            'cv_link': normal_cell
        }
        q_len = len(question_toks)
        c_len = len(db.columns)
        t_len = len(db.tables)
        total_len = q_len + c_len + t_len
        normal_matrix = normal_build_relation_matrix(
            link_info_dict, total_len, q_len, c_len,
            list(range(c_len + 1)), list(range(t_len + 1)), db)


