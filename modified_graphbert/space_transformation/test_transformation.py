


from nltk.corpus import wordnet as wn

word = "coffee.n.01"
list_kids = list(set([j.name().replace("_", " ").lower() for i in wn.synset(word).hyponyms() for j in i.lemmas()]))

ft.wv.similar_by_vector(fasttext_dict[word], topn=20)
list_preds = list(set([i[0].lower() for i in ft.wv.similar_by_vector(fasttext_dict[word], topn=10)]))
print(apk(list_kids, list_preds))

model.eval()
out = model(torch.tensor(train_src_matrix[train_synsets_ordered.index(word)], dtype=torch.float32).to(DEVICE))
print(ft.wv.similar_by_vector(out.detach().cpu().numpy(), topn=10))

list_preds = list(set([i[0].lower() for i in ft.wv.similar_by_vector(out.detach().cpu().numpy(), topn=10)]))
print(apk(list_kids, list_preds))

# testing projection on test set

from collections import defaultdict
from tqdm import tqdm

def sort_candidates(parents, dict_of_candidates):
    results = defaultdict(list)
    for synset_name, synset_candidates in dict_of_candidates.items():
        for parent in wn.synset(synset_name).hypernyms():
            if parent.name() in parents:
                results[parent.name()].extend(synset_candidates)
    final_results = defaultdict(list)
    for k, v in results.items():
        sorted_answers = list(reversed(sorted(v, key=lambda x: x[1])))
        for answer, score in sorted_answers:
            if answer not in final_results[k]:
                final_results[k].append(answer)
    return final_results

model.eval()
test_src_matrix = torch.tensor(test_src_matrix, dtype=torch.float32).to(DEVICE)
out = model(test_src_matrix)
out = out.detach().cpu().numpy()

child_candid_preds = {}
for i in range(len(test_children_ordered)):
    pred_child_emb = out[i, :]
    candidates = ft.wv.similar_by_vector(pred_child_emb, topn=10)
    #print(test_children_ordered[i], [j[0] for j in candidates])
    child_candid_preds[test_children_ordered[i]] = candidates

final_results = sort_candidates(test_parents, child_candid_preds)

apks5 = []
apks10 = []
apks20 = []

rrs5 = []
rrs10 = []
rrs20 = []

precs = []
recs = []
fmes = []

count = 0

with open("pred_res_500_cosine.txt", "w") as f:
    for tp in test_parents:
        list_kids = list(
            set([j.name().replace("_", " ").lower() for i in wn.synset(tp).hyponyms() for j in i.lemmas()]))

        final_preds = final_results[tp]

        apks5.append(apk(list_kids, final_preds, k=5))
        apks10.append(apk(list_kids, final_preds, k=10))
        apks20.append(apk(list_kids, final_preds, k=20))

        rrs5.append(compute_rr(list_kids, final_preds, k=5))
        rrs10.append(compute_rr(list_kids, final_preds, k=10))
        rrs20.append(compute_rr(list_kids, final_preds, k=20))

        precs.append(compute_pr(list_kids, final_preds))
        recs.append(compute_rec(list_kids, final_preds))
        fmes.append(compute_f_score(list_kids, final_preds))

        f.write(f"{tp}, {wn.synset(tp).hyponyms()}\n {list_kids}\n {final_preds}\n")
        f.write(f"AP@5: {apks5[-1]}\t AP@10: {apks10[-1]}\t AP@20: {apks20[-1]}\n")
        f.write(f"RR@5: {rrs5[-1]}\t RR@10: {rrs10[-1]}\t RR@20: {rrs20[-1]}\n")
        f.write(f"Precision: {precs[-1]}\t Recall: {recs[-1]}\t F-score: {fmes[-1]}\n")

        f.write("=" * 40 + "\n")

    f.write(
        f"MAP@5: {sum(apks5) / len(apks5)}\t MAP@10: {sum(apks20) / len(apks20)}\t MAP@20: {sum(apks20) / len(apks20)}\n")
    f.write(f"MRR@5: {sum(rrs5) / len(rrs5)}\t MRR@10: {sum(rrs10) / len(rrs10)}\t MRR@20: {sum(rrs20) / len(rrs20)}\n")
    f.write(
        f"Mean precision: {sum(precs) / len(precs)}\t Mean recall: {sum(recs) / len(recs)}\t Mean F-score: {sum(fmes) / len(fmes)}\n")

