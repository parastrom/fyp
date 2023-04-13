import sys
from collections import namedtuple
import numpy as np
from transformers import BertTokenizer
import json
from util import text_utils


from dataproc import spider_dataset

BertInput = namedtuple(
    "BertInput",
    "token_ids sent_ids table_indexes column_indexes value_indexes value_list token_mapping orig_question_tokens candi_nums"
)


class BertInputEncoder(object):

    def __init__(self, model_config):
        """init of class"""
        super(BertInputEncoder, self).__init__()

        self.config = model_config
        self.tokenizer = BertTokenizer.from_pretrained(model_config.pretrain_model)
        self.enc_value_with_col = model_config.enc_value_with_col
        self.special_token_dict = {
            'table': '[unused1]',
            'column': '[unused2]',
            'value': '[unused3]',
            'text': '[unused11]',
            'real': '[unused12]',
            'number': '[unused13]',
            'time': '[unused14]',
            'binary': '[unused15]',
            'boolean': '[unused16]',
            'bool': '[unused17]',
            'others': '[unused18]',
        }
        self._need_bool_value = True if self.config.grammar_type != 'nl2sql' else False

    def encode(self,
               question,
               db,
               column_match_cells=None,
               candi_nums=None,
               col_orders=None):

        question = question.strip()

        if self.config.num_value_col_type == 'q_num':
            original_q_tokens, candi_nums, candi_nums_index = text_utils.wordseg_and_extract_nums(question)
            if '0' not in candi_nums:
                candi_nums.append('0')
                candi_nums_index.append(-1)
            if '1' not in candi_nums:
                candi_nums.append('1')
                candi_nums_index.append(-1)

            tokens, value_list, schema_indexes, token_mapping = \
                self.tokenize(original_q_tokens, db, column_match_cells, candi_nums, candi_nums_index, col_orders)

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            table_indexes, column_indexes, value_indexes, num_value_indexes = schema_indexes
            q_len = column_indexes[0]
            sent_ids = [0] * q_len + [1] * (len(token_ids) - q_len)

            value_indexes += num_value_indexes
            return BertInput(token_ids, sent_ids, table_indexes, column_indexes,
                             value_indexes, value_list, token_mapping,
                             original_q_tokens, candi_nums)

    def tokenize(self,
                 question,
                 db,
                 column_match_cells=None,
                 candi_nums=None,
                 candi_nums_index=None,
                 col_orders=None):

        """
        Tokenize question and columns and concatenate.
        final_tokens will include：Question、Schema（include non-digital value）、digital value
        [CLS] Q tokens [SEP]
        [T] table1 [C] col1 [V] value [C] col2 ... [SEP]
        [V] number [V] ... [SEP]
        """

        if col_orders is None:
            col_orders = np.array(len(db.columns))
        if isinstance(question, str):
            q_tokens_tmp = self.tokenizer.tokenize(question)
            token_idx_mapping = [[i] for i in range(len(q_tokens_tmp))]
        else:
            q_tokens_tmp, token_idx_mapping = self._resplit_words(question)

        final_candi_nums_index = []
        if candi_nums_index is not None:
            for idx in candi_nums_index:
                if idx < 0:
                    final_candi_nums_index.append(0)
                else:
                    final_candi_nums_index.append(token_idx_mapping[idx][0] + 1)

        question_tokesn = ['[CLS]'] + q_tokens_tmp
        final_tokens = question_tokesn[:self.config.max_question_len] + ['[SEP]']

        columns = [db.columns[i] for i in col_orders]
        if column_match_cells is not None:
            column_match_cells = [column_match_cells[i] for i in col_orders]
        else:
            column_match_cells = [None] * len(columns)

        # handling schema tokens

        table_indexes = []
        column_indexes = []
        value_indexes = []
        value_list = []
        universal_value_set = set(['yes', 'no']) if self._need_bool_value else set()
        for idx, (column, match_cells) in enumerate(zip(columns, column_match_cells)):
            if idx == 1 or \
                    idx > 1 and column.table.id != columns[idx - 1].table.id:
                table_indexes.append(len(final_tokens))
                final_tokens.append(self.special_token_dict['table'])
                final_tokens += self.tokenizer.tokenize(self.tokenizer.tokenize(column.table.orig_name))

            if idx == 0:
                col_name = 'any column'
                col_type = self.special_token_dict['text']
            else:
                col_name = column.orig_name
                col_type = self.special_token_dict[column.type]

            column_indexes.append(len(final_tokens))
            final_tokens += [col_type] + self.tokenizer.tokenize(col_name)

            if match_cells is not None and len(match_cells) > 0:
                if column.type in ('text', 'time'):
                    if not self.config.predict_value:
                        match_cells = match_cells[:1]  # the first cell used to complement semantics
                    for mcell in match_cells:
                        value_list.append(mcell)
                        toks = [self.special_token_dict['value']] + self.tokenizer.tokenize(mcell)
                        if self.enc_value_with_col:
                            value_indexes.extend([column_indexes[-1], len(final_tokens)])
                        else:
                            value_indexes.append(len(final_tokens))
                        final_tokens += toks
                elif self.config.predict_value:
                    for mcell in match_cells:
                        universal_value_set.add(mcell)
        final_tokens.append('[SEP]')

        if self.config.predict_value:
            for value in universal_value_set:
                value_list.append(value)
                toks = [self.special_token_dict['value']] + self.tokenizer.tokenize(value)
                if self.enc_value_with_col:
                    value_indexes.extend([0, len(final_tokens)])
                else:
                    value_indexes.append(len(final_tokens))
                final_tokens += toks
            final_tokens.append('[SEP]')

            num_value_indexes = []
            if candi_nums is not None and len(candi_nums) > 0:
                value_list += candi_nums
                for num, index in zip(candi_nums, final_candi_nums_index):
                    if self.enc_value_with_col:
                        # index is the index of current number in question
                        num_value_indexes.extend([index, len(final_tokens)])
                    elif self.config.num_value_col_type == 'q_num':
                        num_value_indexes.append(index)
                    else:
                        num_value_indexes.append(len(final_tokens))
                    final_tokens += [self.special_token_dict['value']
                                     ] + self.tokenizer.tokenize(num)
            else:
                # use fixed special token value/empty
                if self.enc_value_with_col:
                    value_indexes = [0, len(final_tokens), 0, len(final_tokens) + 1]
                else:
                    value_indexes = [len(final_tokens), len(final_tokens) + 1]
                num_value_indexes = []
                value_list = ['value', 'empty']
                final_tokens.extend(value_list)
            final_tokens.append('[SEP]')

            return final_tokens, value_list, [
                table_indexes, column_indexes, value_indexes, num_value_indexes
            ], token_idx_mapping


    def _resplit_words(self, words):
        """resplit words by bert_tokenizer
        """
        lst_new_result = []
        token_idx_mapping = []
        for idx, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            new_id_start = len(lst_new_result)
            new_id_end = new_id_start + len(tokens)
            lst_new_result.extend(tokens)
            token_idx_mapping.append(list(range(new_id_start, new_id_end)))
        return lst_new_result, token_idx_mapping
