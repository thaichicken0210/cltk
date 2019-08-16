"""Tokenization utilities"""

__author__ = ['Patrick J. Burns <patrick@diyclassics.org>']
__license__ = 'MIT License.'

import pickle
from typing import List

from nltk.tokenize.punkt import PunktLanguageVars
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktTrainer


class BaseSentenceTokenizerTrainer():
    """Train sentence tokenizer"""

    def __init__(self: object,  # pylint: disable=too-many-arguments
                 language: str = None,
                 punctuation: List[str] = None,
                 strict: bool = False,
                 strict_punctuation: List[str] = None,
                 abbreviations: List[str] = None):
        """Initialize stoplist builder with option for language specific parameters.

        :type language: str
        :param language: text from which to build the stoplist
        :type punctuation: list
        :param punctuation: list of punctuation used to train sentence tokenizer
        :type strict: bool
        :param strict: option for including additional punctuation for tokenizer
        :type strict: list
        :param strict: list of additional punctuation used to train sentence tokenizer if strict is used  # pylint: disable=line-too-long
        :type abbreviations: list
        :param abbreviations: list of abbreviations used to train sentence tokenizer
        """
        if language:
            self.language = language.lower()

        self.strict = strict
        self.punctuation = punctuation
        self.strict_punctuation = strict_punctuation
        self.abbreviations = abbreviations

    def train_sentence_tokenizer(self: object, text: str):
        """Train sentence tokenizer."""
        language_punkt_vars = PunktLanguageVars

        # Set punctuation
        if self.punctuation:
            if self.strict:
                language_punkt_vars.sent_end_chars = self.punctuation + self.strict_punctuation
            else:
                language_punkt_vars.sent_end_chars = self.punctuation

        # Set abbreviations
        trainer = PunktTrainer(text, language_punkt_vars)
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.INCLUDE_ABBREV_COLLOCS = True

        tokenizer = PunktSentenceTokenizer(trainer.get_params())

        if self.abbreviations:
            for abbreviation in self.abbreviations:
                tokenizer._params.abbrev_types.add(abbreviation)  # pylint: disable=protected-access

        return tokenizer

    @staticmethod
    def pickle_sentence_tokenizer(filename: str, tokenizer: object):
        """Dump pickled tokenizer."""
        with open(filename, 'wb') as file_open:
            pickle.dump(tokenizer, file_open)
