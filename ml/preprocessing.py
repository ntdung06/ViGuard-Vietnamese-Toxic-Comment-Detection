"""Shared training/inference preprocessing. Never strip Vietnamese diacritics."""
import re
import unicodedata

LABEL_NAMES = ['CLEAN', 'OFFENSIVE', 'HATE']
TEENCODE = {'ko': 'không', 'khong': 'không', 'k0': 'không', 'dc': 'được'}
MODES = ('basic', 'teencode', 'segmented', 'enhanced')


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).lower()).strip()


def preprocess(text, mode='basic'):
    if mode not in MODES:
        raise ValueError(f'Unknown preprocessing mode: {mode}')
    text = normalize(text)
    if mode in ('teencode', 'enhanced'):
        text = re.sub(r'\b\w+\b', lambda m: TEENCODE.get(m[0], m[0]), text)
    if mode in ('segmented', 'enhanced'):
        from underthesea import word_tokenize
        text = word_tokenize(text, format='text')
    return text
