import os.path
from argparse import ArgumentParser

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from read_umls import read_mrconso


def get_concept_embeddings(concepts, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        # Tokenize concepts
        model_input = tokenizer(concepts,
                                add_special_tokens=True,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_token_type_ids=True,
                                return_attention_mask=True)
        # Pass tokenized input into model
        last_hidden_states = model.forward(input_ids=torch.LongTensor(model_input["input_ids"]).to(device),
                                           token_type_ids=torch.LongTensor(model_input["token_type_ids"]).to(device),
                                           attention_mask=torch.LongTensor(model_input["attention_mask"])
                                           .to(device))["last_hidden_state"]
        # Get [CLS] token embeddings
        cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])

    return cls_embeddings


class MrconsoConceptDataset(Dataset):
    def __init__(self, mrconso_df):
        self.mrconso = mrconso_df
        self.concepts = mrconso_df["STR"].values

    def __getitem__(self, idx):
        return self.concepts[idx]

    def __len__(self):
        return len(self.concepts)


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--encoder_name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_embeddings_path')
    parser.add_argument('--output_vocab_path')
    args = parser.parse_args()

    mrconso_path = args.mrconso
    encoder_name = args.encoder_name
    batch_size = args.batch_size
    output_embeddings_path = args.output_embeddings_path
    output_dir = os.path.dirname(output_embeddings_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_vocab_path = args.output_vocab_path
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    mrconso_df = read_mrconso(mrconso_path)
    device = "cuda" if torch.cuda.is_available else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, )
    model = AutoModel.from_pretrained(encoder_name, ).to(device)

    concept_dataset = MrconsoConceptDataset(mrconso_df)
    concept_loader = torch.utils.data.DataLoader(
        concept_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
    )

    with open(output_embeddings_path, 'w+', encoding="utf-8") as output_emb_file, \
            open(output_vocab_path, 'w+', encoding="utf-8") as output_vocab_file:
        with torch.no_grad():
            i = 0
            for concept_batch in tqdm(concept_loader):
                concept_embeddings = get_concept_embeddings(concept_batch, model, tokenizer, device).cpu()
                for emb in concept_embeddings:
                    concept_emb_str = " ".join(str(x) for x in emb)
                    output_emb_file.write(f"{i} {concept_emb_str} 0\n")
                    concept_cui = concept_dataset.mrconso.iloc[i].CUI
                    concept_str = concept_dataset.mrconso.iloc[i].STR
                    output_vocab_file.write(f"{i}\t{concept_str}\t{concept_cui}\n")
                    i += 1
            assert i == mrconso_df.shape[0]


if __name__ == '__main__':
    main()
