import os
import torch
from transformers import BertForMaskedLM, BertConfig, BertTokenizer, BertModel
from helpers.lexsubgen import BertProbEstimator
from helpers.predict import analyze_tagged_text
from helpers.masked_token_predictor_bert import MaskedTokenPredictorBert

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class GraphBERTToBERTPredictor:
    CACHE_DIR = './workdir/cache'

    def __init__(self, use_cuda=True, device='cuda:0'):
        cuda_device = device[-1] if use_cuda else -1
        self.bert = BertProbEstimator(mask_type='masked', cuda_device=cuda_device)
        config = BertConfig.from_pretrained('bert-large-cased', output_hidden_states=True)
        self.model = BertModel.from_pretrained('bert-large-cased', cache_dir=self.CACHE_DIR, config=config)
        self.modellm = BertForMaskedLM.from_pretrained('bert-large-cased', cache_dir=self.CACHE_DIR, config=config)
        if use_cuda:
            self.model.cuda()
        self.bpe_tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True,
                                                           cache_dir=self.CACHE_DIR)
        self.use_tokenizer = False
        self.predictor = MaskedTokenPredictorBert(self.bert, self.bpe_tokenizer, max_len=250, device=device)

    def predict_candidates(self, tagged_text, embedding):
        return list(zip(*analyze_tagged_text(tagged_text, self.predictor, self.use_tokenizer,
                                             n_top=50,
                                             n_units=1,
                                             n_tokens=[1, 2, 3],
                                             max_multiunit=1,
                                             mask_token=True,
                                             fix_multiunit=False,
                                             multiunit_lookup=200,
                                             embedding=embedding)[:2]))

    def get_true_bert_embedding(self, input_text):
        return self.model(**self.bpe_tokenizer(input_text, return_tensors="pt"))[0][:,3:-2,:]\
            .mean(dim=1, keepdims=True).to('cuda:0')

    def test_embedding(self, embedding):
        prediction_scores = self.modellm.cls(embedding.cpu())
        prediction_scores = prediction_scores.detach().squeeze(0)
        probs = torch.softmax(prediction_scores, dim=-1)
        topk_prob, topk_indices = torch.topk(probs[0, :], 50)
        topk_tokens = self.bpe_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())
        for prob, tok in zip(topk_prob, topk_tokens):
            print('{}\t{}'.format(tok, prob))


if __name__ == '__main__':
    gb2b = GraphBERTToBERTPredictor(use_cuda=False)
    true_embedding = gb2b.get_true_bert_embedding('I like caramel .')
    print(gb2b.test_embedding(true_embedding))
    print(gb2b.predict_candidates(f'__[MASK]__', true_embedding))
