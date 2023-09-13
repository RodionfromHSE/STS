import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
from omegaconf import DictConfig
import os
import torch
import json
from typing import List


def extract_by_key(l: List, key: str, transform=lambda x: x):
    return [transform(item[key]) for item in l]


def truncate_file_name(file_name):
    return os.path.join(
        os.path.basename(os.path.dirname(file_name)), os.path.basename(file_name)
    )


class Scorer:
    def __init__(self, config: DictConfig, model, tokenizer) -> None:
        "Fields required: scores (file, func, edit, err, msg)"
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _calculate_edit_distances(strings_a, strings_b):
        """Calculate edit distances between two lists of strings."""
        n = len(strings_a)
        edit_distances = [
            1 - levenshtein_distance(strings_a[i], strings_b[i]) / (2 * n)
            for i in range(n)
        ]
        return edit_distances

    def _string_similarity(self, a: str, b: str):
        tokens_a = self.tokenizer(a, return_tensors="pt", padding=True, truncation=True)
        tokens_b = self.tokenizer(b, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            embeddings_a = self.model(**tokens_a).last_hidden_state.mean(dim=1)
            embeddings_b = self.model(**tokens_b).last_hidden_state.mean(dim=1)

        similarity = cosine_similarity(embeddings_a, embeddings_b).diagonal()[0]
        return similarity

    def _calculate_similarities(self, strings_a, strings_b):
        """Calculate cosine similarities between two lists of strings."""
        tokens_a = self.tokenizer(
            strings_a, return_tensors="pt", padding=True, truncation=True
        )
        tokens_b = self.tokenizer(
            strings_b, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            embeddings_a = self.model(**tokens_a).last_hidden_state.mean(dim=1)
            embeddings_b = self.model(**tokens_b).last_hidden_state.mean(dim=1)

        similarities = cosine_similarity(embeddings_a, embeddings_b).diagonal()
        return similarities

    def trace_similarity(self, obj_a, obj_b):
        """Calculate similarity between two trace objects."""
        trace_a = obj_a["trace_list"]
        trace_b = obj_b["trace_list"]

        # truncate traces to same length
        if len(trace_a) != len(trace_b):
            diff_len = abs(len(trace_a) - len(trace_b))
            if len(trace_a) > len(trace_b):
                trace_a = trace_a[diff_len:]
            else:
                trace_b = trace_b[diff_len:]

        func_names_a = extract_by_key(trace_a, "func_name")
        func_names_b = extract_by_key(trace_b, "func_name")
        # truncate file name (in order not to give redundant information)
        file_names_a = extract_by_key(trace_a, "file_name", truncate_file_name)
        file_names_b = extract_by_key(trace_b, "file_name", truncate_file_name)

        func_name_similarities = self._calculate_similarities(
            func_names_a, func_names_b
        )
        file_name_similarities = self._calculate_similarities(
            file_names_a, file_names_b
        )
        edit_distances = self._calculate_similarities(func_names_a, func_names_b)

        return file_name_similarities, func_name_similarities, edit_distances

    def _get_mean_by_config(
        self, file_name_similarities, func_name_similarities, edit_distances
    ):
        scores = self.config.scores
        file_mn = np.mean(file_name_similarities)
        func_mn = np.mean(func_name_similarities)
        edit_mn = np.mean(edit_distances)
        return scores.file * file_mn + scores.func * func_mn + scores.edit * edit_mn

    def calculate_trace_score(self, obj_a, obj_b):
        (
            file_name_similarities,
            func_name_similarities,
            edit_distances,
        ) = self.trace_similarity(obj_a, obj_b)
        score = self._get_mean_by_config(
            file_name_similarities, func_name_similarities, edit_distances
        )
        return score

    def get_full_score(self, obj1, obj2):
        trace_score = self.calculate_trace_score(obj1, obj2)
        err_score = (
            self._string_similarity(obj1["error_type"], obj2["error_type"])
            * self.config.scores.err
        )
        print(err_score)
        msg_score = (
            self._string_similarity(obj1["error_msg"], obj2["error_msg"])
            * self.config.scores.msg
        )
        return trace_score + err_score + msg_score
