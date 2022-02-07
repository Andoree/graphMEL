import pandas as pd


def read_mrconso(fpath):
    columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def read_mrsty(fpath):
    columns = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def read_mrrel(fpath):
    columns = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI", "SAB", "RSAB", "SAB",
               "SL", "RG", "DIR", "SUPPRESS", "CVF"]
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')
