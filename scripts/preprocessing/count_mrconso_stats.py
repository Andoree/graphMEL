import codecs
import os.path
from argparse import ArgumentParser
# from utils import read_mrrel, read_mrconso
import pandas as pd


def read_mrconso(fpath):
    columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')




def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df_mrconso = read_mrconso(args.mrconso)
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        print(df_mrconso.columns)
        for col in df_mrconso.columns:
            print(col, ' : ', len(df_mrconso[col].unique()))
            out_file.write(f"{col} : {len(df_mrconso[col].unique())}")


if __name__ == '__main__':
    main()
