import logging
import os
from argparse import ArgumentParser

from graphMEL.scripts.utils.io import save_tuples, save_dict, save_node_id2terms_list
from graphMEL.scripts.utils.io import read_mrconso, read_mrrel
from graphMEL.scripts.utils.umls2graph import get_concept_list_groupby_cui, extract_umls_edges


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrrel')
    parser.add_argument('--output_node_id2synonyms_path', )
    parser.add_argument('--output_node_id2cui_path', )
    parser.add_argument('--output_edges_path', )
    args = parser.parse_args()

    output_node_id2terms_list_path = args.output_node_id2synonyms_path
    output_node_id2cui_path = args.output_node_id2cui_path
    output_edges_path = args.output_edges_path
    output_paths = (output_node_id2terms_list_path, output_node_id2cui_path, output_edges_path)
    for path in output_paths:
        output_dir = os.path.dirname(path)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)
    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)
    mrconso_df["CUI"] = mrconso_df["CUI"].fillna('')
    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(args.mrrel)

    logging.info("Generating node index....")
    node_id2terms_list, node_id2cui = get_concept_list_groupby_cui(mrconso_df=mrconso_df)
    cui2node_id = {cui: node_id for node_id, cui in node_id2cui.items()}
    assert len(node_id2cui) == len(cui2node_id)
    logging.info("Generating edges....")
    edges = extract_umls_edges(mrrel_df, cui2node_id)
    logging.info("Saving the result....")
    save_node_id2terms_list(save_path=output_node_id2terms_list_path, mapping=node_id2terms_list, )
    save_dict(save_path=output_node_id2cui_path, dictionary=node_id2cui)
    save_tuples(save_path=output_edges_path, tuples=edges)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main()
