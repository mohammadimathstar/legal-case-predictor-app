import yaml
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from typing import Tuple, Dict

from src.utils import load_model, find_closest_prototypes_without_label
from src.logs import get_logger
from src.utils_embedding import (load_embedding_model,
                                       compute_document_embedding_full,
                                       preprocess_text, get_document_subspace_full)
from src.AChorDSLVQ.model import AChorDSLVQModel
from src.visualization_explainability import plot_top_words



def compute_subspace_contribution(vh, s, word_frequencies, subspace_dim):
    """
    Parameters:
        Vh: (D x n)-matrix - (n: number of distinct words in the doc, D: embedding dimensionality e.g.300).
        S: (n x n)- matrix - a vector containing  singular values.
        word_frequencies (1 x n)-matrix - The number of times a word in the text appears

    Return:
         RS^(-1) : it is a (n x d) matrix
    It also shows the coordinate of each word on the reconstructed subspace
    """

    if word_frequencies.ndim == 1 :
        word_frequencies = np.expand_dims(word_frequencies, axis=1)
    RS_1 = word_frequencies * vh[:subspace_dim,:].T @ np.diag(1 / s[:subspace_dim])

    return RS_1


def compute_documbent_embedding(txt: str,
                                embedding_model: KeyedVectors,
                                subspace_dim: int) -> Dict:
    """
    Compute document embedding using a word embedding method e.g. GloVe

    Parameters:
        txt (str): the text of a document
        embedding_model: the word embedding model e.g. GloVe or Word2Vec
        subspace_dim (int): the dimensionality of subspace embedding

    Returns:
        a dictionary containing document embedding and the impact of each word on it.
    """

    # Preprocess data
    tokens = preprocess_text(txt)
    word_embeddings, word_frequencies, words = compute_document_embedding_full(embedding_model,
                                                                             tokens,
                                                                             subspace_dim)
    # Compute Document Embedding
    doc_embedding, s, vh = get_document_subspace_full(word_embeddings, word_frequencies, subspace_dim)
    word_impact_on_doc_embedding = compute_subspace_contribution(vh, s, word_frequencies, subspace_dim)

    return {
        'doc_embedding': doc_embedding,
        'word_impact_on_doc_embedding': word_impact_on_doc_embedding,
        'words': words,
        'word_embeddings': word_embeddings,
    }


def prediction_with_winners(doc_embedding: np.ndarray,
                            achords_model: AChorDSLVQModel) -> Tuple[Dict, int, int]:
    """
    Find the closest two closest prototypes to a document

    Parameters:
        doc_embedding: the embedding subspace of the document
        achords_model: the classifier model

    Returns:
        a dictionary containing the closest two prototypes and model's prediction
    """

    # Prediction and Word Weights
    first_winner, second_winner = find_closest_prototypes_without_label(doc_embedding,
                                                                        achords_model.prototype_features,
                                                                        achords_model.prototype_labels,
                                                                        achords_model.lambda_value)
    return {
        'closest_prototype': first_winner,
        'second_closest_prototype': second_winner,
    }, achords_model.prototype_labels[first_winner['index']], achords_model.prototype_labels[second_winner['index']]


def get_word_importances(txt: str,
                         embedding_model: KeyedVectors,
                         achords_model: AChorDSLVQModel):

    subspace_dim = achords_model.prototype_features.shape[-1]

    # Compute document subspace embedding
    doc_repr = compute_documbent_embedding(txt, embedding_model, subspace_dim)

    # Do prediction and find two closest prototypes
    winners, pred, second_pred = prediction_with_winners(doc_repr['doc_embedding'], achords_model)

    word_importances_winners = {}
    for winner_type, dic in winners.items():
        X, M = doc_repr['word_embeddings'], doc_repr['word_impact_on_doc_embedding'] @ dic['Q']
        W = X.T @ achords_model.prototype_features[dic['index']] @ dic['Qw']
        txt_words_impact = achords_model.lambda_value * M * W
        word_importances_winners[winner_type] = txt_words_impact.sum(axis=1)

    return word_importances_winners, doc_repr['words'], pred, second_pred


def get_top_words(words,
                  word_importance_for_winners,
                  num_of_top_words,
                  ):

    positive_scores = word_importance_for_winners['closest_prototype']
    negative_scores = word_importance_for_winners['second_closest_prototype']
    diff = positive_scores - negative_scores
    sort_idx_pos = np.argsort(positive_scores)
    sort_idx_neg = np.argsort(negative_scores)
    sort_idx_diff = np.argsort(diff)
    topwords_diff = [(i, words[i]) for i in sort_idx_diff[-1:-num_of_top_words-1:-1]]
    top_words = [(i, words[i]) for i in sort_idx_pos[-1:-num_of_top_words-1:-1]]
    top_words.extend([(i, words[i]) for i in sort_idx_neg[-1:-num_of_top_words-1:-1]])
    top_words.extend(topwords_diff)
    top_words_set = set(top_words)

    dic = dict()
    for i, w in top_words_set:
        dic[w] = {
            'closest_prototype_score': positive_scores[i],
            'second_closest_prototype_score': negative_scores[i],
            'decision': diff[i],
        }

    return pd.DataFrame.from_dict(dic, orient='index'), topwords_diff



def explain_decision(txt: str,
                     num_of_top_words: int,
                     config_path: str,
                     save_it: bool = True) -> Dict:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    label2class_housing = {i: name for i, name in enumerate(config['evaluate']['target_names_housing'])}
    label2class_eviction = {i: name for i, name in enumerate(config['evaluate']['target_names_eviction'])}

    logger = get_logger("EXPLAIN",
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    # Load embedding model
    logger.info("Load embedding model")
    embedding_model = load_embedding_model(
        embedding_model_path=config['featurize']['embedding_model'])

    # Load AChorDS-LVQ model
    housing_achords_model = load_model(filepath=config['model']['housing_model_path'])
    eviction_achords_model = load_model(filepath=config['model']['eviction_model_path'])

    # Compute words' importances
    word_importances_for_winners_housing, words_housing, pred_housing, second_pred_housing = get_word_importances(txt,
                                                                                                                  embedding_model,
                                                                                                                  housing_achords_model)
    word_importances_for_winners_eviction, words_eviction, pred_eviction, second_pred_eviction = get_word_importances(txt,
                                                                                                                      embedding_model,
                                                                                                                      eviction_achords_model)

    logger.info(f"Prediction of 'housing' model: '{label2class_housing[pred_housing]}', \t Prediction of 'eviction' "
                f"model: '{label2class_eviction[pred_eviction]}'.")


    housing_top_words_df, housing_topwords_diff = get_top_words(words_housing,
                                                word_importances_for_winners_housing,
                                                num_of_top_words,
                                                )
    eviction_top_words_df, eviction_topwords_diff = get_top_words(words_eviction,
                                                word_importances_for_winners_eviction,
                                                num_of_top_words,
                                                )

    housing_top_words_df.rename(columns={'closest_prototype_score' : label2class_housing[pred_housing],
                                         'second_closest_prototype_score': label2class_housing[second_pred_housing]},
                                inplace=True)
    eviction_top_words_df.rename(columns={'closest_prototype_score': label2class_eviction[pred_eviction],
                                         'second_closest_prototype_score': label2class_eviction[second_pred_eviction]},
                                inplace=True)

    housing_top_words_df.index.names = ['Words']
    eviction_top_words_df.index.names = ['Words']

    if save_it:
        housing_top_words_df.to_csv(config['explainability']['housing_table_path'])
        eviction_top_words_df.to_csv(config['explainability']['eviction_table_path'])

        logger.info(
            f"Housing model's top {num_of_top_words} words are save in '{config['explainability']['housing_table_path']}'")
        logger.info(
            f"Eviction model's top {num_of_top_words} words are save in '{config['explainability']['eviction_table_path']}'")

        plot_top_words(housing_top_words_df, config['explainability']['housing_fig_path'])
        plot_top_words(eviction_top_words_df, config['explainability']['eviction_fig_path'])
        logger.info(
            f"The visualization of {num_of_top_words} words for the 'housing' model is saved in '{config['explainability']['housing_fig_path']}'")
        logger.info(
            f"The visualization of {num_of_top_words} words for the 'eviction' model is save in '{config['explainability']['eviction_fig_path']}'")

    return {'housing_pred': label2class_housing[pred_housing],
            'eviction_pred': label2class_eviction[pred_eviction],
            'housing_viz': os.path.basename(config['explainability']['housing_fig_path']),
            'eviction_viz': os.path.basename(config['explainability']['eviction_fig_path'])}


