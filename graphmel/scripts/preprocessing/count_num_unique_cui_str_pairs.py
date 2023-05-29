import codecs
import logging
import os.path
from argparse import ArgumentParser

from graphmel.scripts.utils.io import read_mrconso


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    logging.info("Loading MRCONSO")
    df_mrconso = read_mrconso(args.mrconso)[["CUI", "STR", "LAT"]]
    logging.info(f"MRCONSO: {df_mrconso.shape}")
    df_mrconso_unique_cui_str_lat = df_mrconso.drop_duplicates(subset=("CUI", "STR", "LAT"), inplace=False)
    logging.info(f"Drooped non-unique <CUI, STR, LAT> MRCONSO: {df_mrconso_unique_cui_str_lat.shape}")

    gb_cui_str_lat_df = df_mrconso_unique_cui_str_lat.groupby(['CUI', 'STR', 'LAT'])[['CUI', 'STR']].count()
    gb_cui_str_lat_path = os.path.join(output_dir, "gb_cui_str_lat.tsv")
    gb_cui_str_lat_df.to_csv(gb_cui_str_lat_path, sep='\t',)

    df_mrconso_unique_cui_str = df_mrconso[["CUI", "STR"]].drop_duplicates(subset=("CUI", "STR"), inplace=False)

    gb_cui_str_df = df_mrconso_unique_cui_str.groupby(['CUI', 'STR']).count()
    gb_cui_str_path = os.path.join(output_dir, "gb_cui_str.tsv")
    gb_cui_str_df.to_csv(gb_cui_str_path, sep='\t', )



if __name__ == '__main__':
    main()
