# -*- coding: utf-8 -*-
import re
import string
import pickle
import numpy as np

VOWELS = set("aeiou")
CONSONANTS = set(string.ascii_lowercase) - set("aeiou")


class Features:
    @staticmethod
    def url_path_to_dict(path: str):
        pattern = (r'^'
                   r'((?P<schema>.+?)://)?'
                   r'((?P<user>.+?)(:(?P<password>.*?))?@)?'
                   r'(?P<host>.*?)'
                   r'(:(?P<port>\d+?))?'
                   r'(?P<path>/.*?)?'
                   r'(?P<query>[?].*?)?'
                   r'$'
                   )
        regex = re.compile(pattern)
        m = regex.match(path)
        d = m.groupdict() if m is not None else None
        return d

    @staticmethod
    def extract_doc(s):
        return " ".join(re.split("[" + string.punctuation + "]+", s))

    @staticmethod
    def vowels_pct(s):
        count = 0
        s = s.lower()

        for ch in s:
            if ch in VOWELS:
                count = count + 1

        return count / len(s)

    @staticmethod
    def consonants_pct(s):
        count = 0
        s = s.lower()

        for ch in s:
            if ch in CONSONANTS:
                count = count + 1

        return count / len(s)

    @staticmethod
    def is_ip(d: dict):
        if not d:
            return False

        res_s = re.sub(r'[/.]', '', d['host'])
        return int(res_s.isnumeric())

    @staticmethod
    def contains_port(d: dict):
        if not d:
            return False

        if d['port']:
            return 1
        else:
            return 0

    @staticmethod
    def contains_username(d: dict):
        if not d:
            return False

        if d['user']:
            return 1
        else:
            return 0

    @staticmethod
    def url_length(s: str):
        return len(s)

    @staticmethod
    def count_dots(s):
        return s.count('.')

    @staticmethod
    def count_slash(s):
        return s.count('/')

    @staticmethod
    def count_digits(s):
        return len(re.sub(r"\D", "", s))

    @staticmethod
    def count_punctuation(s):
        return len(re.sub(r"[^" + string.punctuation + "]+", "", s))

    @staticmethod
    def hostname_length(d: dict):
        if not d:
            return 0

        if not d['host']:
            return 0
        else:
            return len(d['host'])

    @staticmethod
    def path_length(d: dict):
        if not d:
            return 0

        if not d['path']:
            return 0
        else:
            return len(d['path'])

    @staticmethod
    def query_length(d: dict):
        if not d:
            return 0

        if not d['query']:
            return 0
        else:
            return len(d['query'])


class Inference:

    @staticmethod
    def infer(sample: np.array,
              model_filename: str,
              vectorizer_filename: str,
              scaler_filename: str):
        model = pickle.load(open(model_filename, "rb"))
        tf_idf_vec = pickle.load(open(vectorizer_filename, "rb"))
        sc = pickle.load(open(scaler_filename, "rb"))

        feature_vec = np.array([])

        url_info = Features.url_path_to_dict(sample)
        doc = Features.extract_doc(sample)

        feature_vec = np.append(feature_vec, tf_idf_vec.transform(
            np.array([doc])).toarray())

        feature_vec = np.append(feature_vec, Features.vowels_pct(sample))
        feature_vec = np.append(feature_vec, Features.consonants_pct(sample))
        feature_vec = np.append(feature_vec, Features.is_ip(url_info))
        feature_vec = np.append(feature_vec, Features.contains_port(url_info))
        feature_vec = np.append(feature_vec,
                                Features.contains_username(url_info))

        feature_vec = np.append(feature_vec,
                                sc.transform(
                                    np.array([[Features.url_length(sample),
                                               Features.count_dots(sample),
                                               Features.count_slash(sample),
                                               Features.count_digits(sample),
                                               Features.count_punctuation(
                                                   sample),
                                               Features.hostname_length(
                                                   url_info),
                                               Features.path_length(url_info),
                                               Features.query_length(
                                                   url_info)]])))

        y_hat = model.predict(feature_vec.reshape(1, -1))

        map = {0: "Phishing URL", 1: "Legitimate URL"}

        return map[y_hat[0]]
