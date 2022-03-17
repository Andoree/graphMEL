import os.path
from argparse import ArgumentParser
from graphmel.scripts.utils.io import read_mrconso, read_mrsty

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrsty')
    parser.add_argument('--types', nargs='+', default=[])
    parser.add_argument('--langs', nargs='+', default=['ENG', 'FRE', 'GER', 'SPA', 'DUT', ])
    parser.add_argument('--ontology', default=None, nargs='+')
    parser.add_argument('--concept_id_column', default='CUI')
    parser.add_argument('--filter_unique_str', action="store_true")
    parser.add_argument('--save_to')
    parser.add_argument('--save_all', action='store_true')
    args = parser.parse_args()

    output_path = args.save_to
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_fname = os.path.basename(output_path)
    mrconso = read_mrconso(args.mrconso)

    if len(args.types) > 0:
        mrsty = read_mrsty(args.mrsty)
        filtered_concepts = mrsty[mrsty.TUI.isin(args.types)]['CUI'].drop_duplicates()
        filtered_umls = mrconso[(mrconso.CUI.isin(filtered_concepts)) & (mrconso.LAT.isin(args.langs))]
    else:

        filtered_umls = mrconso[mrconso.LAT.isin(args.langs)]
        output_fname = f"{'_'.join(args.langs)}_{output_fname}"
    if args.ontology is not None:
        filtered_umls = filtered_umls[filtered_umls.SAB.isin(args.ontology)]
        output_fname = f"{'_'.join(args.ontology)}_{output_fname}"
    if args.filter_unique_str:
        filtered_umls.drop_duplicates(keep="first", subset=("CUI", "STR"), inplace=True)

    final = filtered_umls
    if not args.save_all:
        final = final[[args.concept_id_column, 'STR']]
    output_path = os.path.join(output_dir, output_fname)
    final.drop_duplicates().to_csv(output_path, index=False, header=False, sep='|')
