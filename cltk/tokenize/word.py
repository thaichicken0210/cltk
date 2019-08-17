"""Language-specific word tokenizers, for example for handling enclitics."""

__author__ = ['Patrick J. Burns <patrick@diyclassics.org>',
              'Kyle P. Johnson <kyle@kyle-p-johnson.com>',
              'Natasha Voake <natashavoake@gmail.com>',
              'Clément Besnier <clemsciences@aol.com>',
              'Andrew Deloucas <adeloucas@g.harvard.edu>']

__license__ = 'MIT License. See LICENSE.'

from typing import List
from abc import abstractmethod
import re

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize.treebank import TreebankWordTokenizer

from cltk.corpus.arabic.utils.pyarabic import araby
from cltk.tokenize.akkadian.word import tokenize_akkadian_signs
from cltk.tokenize.akkadian.word import tokenize_akkadian_words
from cltk.tokenize.greek.sentence import GreekRegexSentenceTokenizer
from cltk.tokenize.middle_english.params import MiddleEnglishTokenizerPatterns
from cltk.tokenize.middle_high_german.params import MiddleHighGermanTokenizerPatterns
from cltk.tokenize.old_norse.params import OldNorseTokenizerPatterns
from cltk.tokenize.old_french.params import OldFrenchTokenizerPatterns
from cltk.utils.cltk_logger import logger


class WordTokenizer:  # pylint: disable=too-few-public-methods
    """Tokenize according to rules specific to a given language."""

    def __init__(self, language):
        """Take language as argument to the class. Check availability and
        setup class variables."""
        self.language = language
        self.available_languages = ['akkadian',
                                    'arabic',
                                    'french',  # defaults to old_french
                                    'greek',
                                    'latin',  # deprecated, raises warning
                                    'middle_english',
                                    'middle_french',
                                    'middle_high_german',
                                    'old_french',
                                    'old_norse',
                                    'sanskrit',
                                    'multilingual'
                                    ]
        assert_message = f"Specific tokenizer not available for '{self.language}'. Only available for: '{self.available_languages}'."  # pylint: disable=line-too-long
        assert self.language in self.available_languages, assert_message
        # raise deprecation or other warnings
        if self.language == 'french':
            self.language = 'old_french'
            message = "`french` defaults to `old_french`. `middle_french` also available."  # pylint: disable=line-too-long
            print(f'WARNING: {message}')
            logger.warning(message)
        elif self.language == 'latin':
            message = "This method for Latin is deprecated. Instead, use `from cltk.tokenize.latin.word import WordTokenizer`."  # pylint: disable=line-too-long
            print(f'WARNING: {message}')
            logger.warning(message)

    def tokenize(self, string):
        """Tokenize incoming string."""
        if self.language == 'akkadian':
            tokens = tokenize_akkadian_words(string)
        elif self.language == 'arabic':
            tokenizer = BaseArabyWordTokenizer('arabic')
            tokens = tokenizer.tokenize(string)
        elif self.language == 'french':
            tokenizer = BaseRegexWordTokenizer('old_french',
                                               OldFrenchTokenizerPatterns)
            tokens = tokenizer.tokenize(string)
        elif self.language == 'greek':
            tokenizer = BasePunktWordTokenizer('greek',
                                               GreekRegexSentenceTokenizer)
            tokens = tokenizer.tokenize(string)
        elif self.language == 'latin':
            tokenizer = TreebankWordTokenizer()
            tokens = tokenizer.tokenize(string)
        elif self.language == 'old_norse':
            tokenizer = BaseRegexWordTokenizer('old_norse',
                                               OldNorseTokenizerPatterns)
            tokens = tokenizer.tokenize(string)
        elif self.language == 'middle_english':
            tokenizer = BaseRegexWordTokenizer('middle_english',
                                               MiddleEnglishTokenizerPatterns)
            tokens = tokenizer.tokenize(string)
        elif self.language == 'middle_french':
            tokenizer = BaseRegexWordTokenizer('old_french',
                                               OldFrenchTokenizerPatterns)
            tokens = tokenizer.tokenize(string)
        elif self.language == 'middle_high_german':
            tokenizer = BaseRegexWordTokenizer('middle_high_german',
                                               MiddleHighGermanTokenizerPatterns)  # pylint: disable=line-too-long
            tokens = tokenizer.tokenize(string)
        elif self.language == 'old_french':
            tokenizer = BaseRegexWordTokenizer('old_french',
                                               OldFrenchTokenizerPatterns)
            tokens = tokenizer.tokenize(string)
        else:
            message = f"Selected language '{self.language}' has not been assigned a specific word tokenizer. Using the NLTK's default `TreebankWordTokenizer()`."  # pylint: disable=line-too-long
            print(f'WARNING: {message}')
            logger.warning(message)
            tokenizer = TreebankWordTokenizer()
            tokens = tokenizer.tokenize(string)
        return tokens

    def tokenize_sign(self, word):
        """This is for tokenizing cuneiform signs."""
        if self.language == 'akkadian':
            sign_tokens = tokenize_akkadian_signs(word)
        else:
            sign_tokens = 'Language must be written using cuneiform.'
        return sign_tokens


class BaseWordTokenizer:
    """Base class for word tokenization

    TODO: Document this and its usefulness. If useful, then comment this and its subclasses better.
    """

    def __init__(self, language: str = None):
        """Constructor for BaseWordTokenizer

        :param language : language for word tokenization
        :type language: str
        """
        if language:
            self.language = language.lower()

    @abstractmethod
    def tokenize(self, text: str, model: object = None):
        """Create a list of tokens from a string. This method should
        be overridden by subclasses of BaseWordTokenizer.
        """
        pass


class BasePunktWordTokenizer(BaseWordTokenizer):
    """Base class for punkt word tokenization"""

    def __init__(self, language: str = None, sent_tokenizer: object = None):
        """Constructor for BasePunktWordTokenizer.

        :param language : language for sentence tokenization
        :type language: str
        """
        self.language = language
        super().__init__(language=self.language)
        if sent_tokenizer:
            self.sent_tokenizer = sent_tokenizer()
        else:
            punkt_param = PunktParameters()
            self.sent_tokenizer = PunktSentenceTokenizer(punkt_param)

    def tokenize(self, text: str):
        """Tokenize input str and return a list of word token str.

        :rtype: list
        :param text: text to be tokenized into sentences
        :type text: str
        """
        sents = self.sent_tokenizer.tokenize(text)
        tokenizer = TreebankWordTokenizer()
        return [item for sublist in tokenizer.tokenize_sents(sents) for item in sublist]


class BaseRegexWordTokenizer(BaseWordTokenizer):
    """Base class for regex word tokenization"""

    def __init__(self, language: str = None, patterns: List[str] = None):
        """
        :param language : language for sentence tokenization
        :type language: str
        :param patterns: regex patterns for word tokenization
        :type patterns: list of strings
        """
        self.language = language
        self.patterns = patterns
        super().__init__(language=self.language)

    def tokenize(self, text: str):
        """Tokenize input str and return a list of word token str.

        :rtype: list
        :param text: text to be tokenized into sentences
        :type text: str
        :param model: tokenizer object to used # Should be in init?
        :type model: object
        """
        for pattern in self.patterns:
            text = re.sub(pattern[0], pattern[1], text)
        return text.split()


class BaseArabyWordTokenizer(BaseWordTokenizer):
    """Base class for word tokenizer using the pyarabic package:
    https://pypi.org/project/PyArabic/
    """

    def __init__(self, language: str = None):
        """
        :param language : language for sentence tokenization
        :type language: str
        """
        self.language = language
        super().__init__(language=self.language)

    def tokenize(self, text: str):
        """
        :rtype: list
        :param text: text to be tokenized into sentences
        :type text: str
        """
        return araby.tokenize(text)
