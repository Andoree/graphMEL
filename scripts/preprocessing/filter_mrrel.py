import pandas as pd
from argparse import ArgumentParser
from graphmel.scripts.utils.io import read_mrrel

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mrrel')
    parser.add_argument('--df', default=None)
    parser.add_argument('--concept_id_column', default='CUI')
    parser.add_argument('--ontology', default=None, nargs='+')
    parser.add_argument('--save_to')
    parser.add_argument('--save_all', action='store_true')
    args = parser.parse_args()

    df_mrrel = read_mrrel(args.mrrel)
    
    if args.ontology is not None: 
        df_mrrel = df_mrrel[df_mrrel.SAB.isin(args.ontology)]
       
    if args.df is not None:
        df = pd.read_csv(args.df, sep="\t")
        ids = df[args.concept_id_column]
        df_mrrel = df_mrrel[df_mrrel.SAB.isin(args.ontology)]
    
    final = df_mrrel
    if not args.save_all:
        final = final[[args.concept_id_column, 'STR']]
    final.drop_duplicates().to_csv(args.save_to,  index=False, header=True, sep='\t')