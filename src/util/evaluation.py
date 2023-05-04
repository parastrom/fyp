import sqlite3
import sqlparse
from sqlparse.sql import Identifier, Function
from sqlparse.tokens import Keyword, DML, CTE, Comparison, Operator

from sqloxide import parse_sql

from nltk import word_tokenize


def extract_grammar_components(query):
    parsed = sqlparse.parse(query)[0]
    tokens = []

    def extract_token_from_group(group):
        for item in group.tokens:
            if isinstance(item, sqlparse.sql.Comparison):
                tokens.append(item.tokens[1].value.upper())
            elif isinstance(item, sqlparse.sql.Operation):
                tokens.append(item.value.upper())
            elif isinstance(item, sqlparse.sql.Parenthesis):
                extract_token_from_group(item)
            elif item.ttype in (Keyword, DML, CTE):
                tokens.append(item.value.upper())

    extract_token_from_group(parsed)
    return tokens


def count_component1(components):
    component1 = ('WHERE', 'GROUP', 'ORDER', 'LIMIT', 'JOIN', 'OR', 'LIKE')
    count = 0
    for component in component1:
        if component in components:
            count += 1
    return count


def count_component2(components):
    component2 = ('EXCEPT', 'UNION', 'INTERSECT')
    count = 0
    for component in component2:
        if component in components:
            count += 1
    return count


def count_others(components):
    all_components = ('WHERE', 'GROUP', 'ORDER', 'LIMIT', 'JOIN', 'OR', 'LIKE', 'EXCEPT', 'UNION', 'INTERSECT')
    count = 0
    for component in all_components:
        if component in components:
            count += 1
    return count


def eval_hardness(components):
    count_comp1_ = count_component1(components)
    count_comp2_ = count_component2(components)
    count_others_ = count_others(components)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"


def match_components_without_values(orig_components, generated_components):
    component_matches = []

    orig_idx = 0
    gen_idx = 0
    while orig_idx < len(orig_components) and gen_idx < len(generated_components):
        if orig_components[orig_idx] == generated_components[gen_idx]:
            component_matches.append((orig_components[orig_idx], generated_components[gen_idx]))
            orig_idx += 1
        gen_idx += 1

    return component_matches


def compute_metrics(orig_components, gen_components):
    tp = 0
    fp = 0
    fn = 0

    orig_idx = 0
    gen_idx = 0

    while orig_idx < len(orig_components) and gen_idx < len(gen_components):
        if orig_components[orig_idx] == gen_components[gen_idx]:
            tp += 1
            orig_idx += 1
        else:
            fp += 1
        gen_idx += 1

    fn = len(orig_components) - tp

    accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1
    precision = tp / (tp + fp) if tp + fp > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 1

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def evaluate_example(orig_query, gen_query, db_path):
    # Calculate exact matching without values
    orig_components = extract_grammar_components(orig_query)
    gen_components = extract_grammar_components(gen_query)
    matched_components_without_values = match_components_without_values(orig_components, gen_components)
    exact_match_without_values = int(len(matched_components_without_values) == len(orig_components))

    # Calculate execution accuracy with values
    orig_query_result = execute_query(db_path, orig_query)
    gen_query_result = execute_query(db_path, gen_query)
    execution_accuracy_with_values = int(orig_query_result == gen_query_result)

    # Calculate component-wise metrics
    component_wise_metrics = compute_metrics(orig_components, gen_components)

    return {
        "exact_match_without_values": exact_match_without_values,
        "execution_accuracy_with_values": execution_accuracy_with_values,
        "component_wise_metrics": component_wise_metrics
    }


def evaluate_queries(orig_queries, gen_queries, db_paths):
    assert len(orig_queries) == len(gen_queries) == len(db_paths), "Input lists must have the same length."

    total_exact_match_without_values = 0
    total_execution_accuracy_with_values = 0
    component_wise_accuracies = {"accuracy": [], "f1": [], "precision": [], "recall": []}
    level_counts = {'easy': 0, 'medium': 0, 'hard': 0, 'extra': 0}
    level_accuracies = {'easy': 0, 'medium': 0, 'hard': 0, 'extra': 0}

    for orig_query, gen_query, db_path in zip(orig_queries, gen_queries, db_paths):
        orig_components = extract_grammar_components(orig_query)
        level = eval_hardness(orig_components)
        level_counts[level] += 1

        evaluation_result = evaluate_example(orig_query, gen_query, db_path)
        exact_match_without_values = evaluation_result["exact_match_without_values"]
        execution_accuracy_with_values = evaluation_result["execution_accuracy_with_values"]

        total_exact_match_without_values += exact_match_without_values
        total_execution_accuracy_with_values += execution_accuracy_with_values

        level_accuracies[level] += execution_accuracy_with_values

        for key in evaluation_result["component_wise_metrics"].keys():
            component_wise_accuracies[key].append(evaluation_result["component_wise_metrics"][key])

    num_examples = len(orig_queries)
    avg_exact_match_without_values = total_exact_match_without_values / num_examples
    avg_execution_accuracy_with_values = total_execution_accuracy_with_values / num_examples

    for key, values in component_wise_accuracies.items():
        component_wise_accuracies[key] = sum(values) / len(values)

    for level in level_accuracies.keys():
        if level_counts[level] > 0:
            level_accuracies[level] = level_accuracies[level] / level_counts[level]
        else:
            level_accuracies[level] = 0

    return {
        "exact_match_without_values_accuracy": avg_exact_match_without_values,
        "execution_accuracy_with_values": avg_execution_accuracy_with_values,
        "component_wise_accuracies": component_wise_accuracies,
        "level_accuracies": level_accuracies
    }


def execute_query(db, query):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error executing query: {e}")
        results = None

    conn.close()
    return results
