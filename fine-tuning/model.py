import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel

from tqdm import tqdm
from task9_scorer import average_precision
from utils import load_vocabs


class HypertModel(nn.Module):
    def __init__(self, model_path, revision=None, use_auth_token=None):
        super(HypertModel, self).__init__()
        self.n_proj = 24
        self.dim = 768
        if revision is None:
            self.bert = AutoModel.from_pretrained(model_path)
        elif use_auth_token is None:
            self.bert = AutoModel.from_pretrained(model_path, revision=revision)
        else:
            self.bert = AutoModel.from_pretrained(
                model_path, revision=revision, use_auth_token=use_auth_token
            )

        self.dropout = nn.Dropout(0.2)
        self.proj_matrix = self._make_proj_matrix(self.n_proj, 200)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.query_proj = nn.Linear(768, 200)
        self.cand_proj = nn.Linear(768, 200)
        self.cls_proj = nn.Linear(768, self.n_proj)

        self.classifier = nn.Linear(self.n_proj * 2, 1)

    def forward(self, input_ids, attention_mask, query_mask, hyper_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        cls_token_embedding = last_hidden_state[:, 0]
        expand_query_mask = query_mask.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1])
        expand_hyper_mask = hyper_mask.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1])

        query_mean_embedding = torch.sum(
            torch.mul(last_hidden_state, expand_query_mask), 1
        ) / torch.sum(query_mask, -1).unsqueeze(-1)
        hyper_mean_embedding = torch.sum(
            torch.mul(last_hidden_state, expand_hyper_mask), 1
        ) / torch.sum(hyper_mask, -1).unsqueeze(-1)

        query_emb = self.query_proj(query_mean_embedding)
        cand_emb = self.cand_proj(hyper_mean_embedding)
        cls_emb = self.cls_proj(cls_token_embedding)

        ## Projection
        query_emb = self.dropout(query_emb)
        projections = torch.matmul(self.proj_matrix, query_emb.transpose(0, 1))
        projections = self.dropout(projections)
        projections = projections.transpose(0, 1).transpose(0, 2)

        cand_emb = self.dropout(cand_emb)
        cand_emb = cand_emb.unsqueeze(-1)

        features = torch.bmm(projections, cand_emb)
        full_features = torch.cat([features.squeeze(), cls_emb], 1)

        # concatenate embeddings
        x = self.classifier(full_features)

        return self.sigmoid(x)

    def freeze_bert(self):
        for param in self.bert.named_parameters():
            param[1].requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.named_parameters():
            param[1].requires_grad = True

    def _make_proj_matrix(self, n_proj, dim):
        var = 2 / (dim + dim)
        noise = torch.zeros([n_proj, dim, dim], dtype=torch.float32)
        noise.normal_(0, var)
        proj_mat = noise + torch.cat([torch.eye(dim, dim).unsqueeze(0) for _ in range(n_proj)])
        return nn.Parameter(proj_mat)


class HypertPredictor(object):
    def __init__(self, model, tokenizer, batch_size, subtask, n_preds):
        self.predictor = Predictor(model, tokenizer, batch_size, subtask)
        self.n_preds = n_preds

    def pred_hypernyms(self, queries):
        hypernyms = []
        for query in tqdm(queries):
            _, preds = self.predictor.predict(query, self.n_preds)
            hypernyms.append(preds)

        return hypernyms


class Evaluator(object):
    def __init__(self, model, tokenizer, batch_size, subtask, n_preds):
        self.evaluator = Predictor(model, tokenizer, batch_size, subtask)
        self.n_preds = n_preds

    def get_MAP(self, queries, golds):
        APs = []  # Average Precision
        for query in tqdm(queries):
            GT_hypernyms = golds[query]
            _, preds = self.evaluator.predict(query, self.n_preds)
            answer = [0 for _ in range(self.n_preds)]
            for i, pred in enumerate(preds):
                if pred in GT_hypernyms:
                    answer[i] = 1
            APs.append(average_precision(answer, len(GT_hypernyms)))

        MAP = np.mean(APs)
        return MAP


class Predictor(object):
    def __init__(self, model, tokenizer, batch_size, subtask):
        self.model = model
        self.tokenizer = tokenizer
        self.vocabs = load_vocabs(subtask)
        self.batch_size = batch_size
        self.type_token_dict = {"Concept": "[CON]", "Entity": "[ENT]"}

    def _generate_sentences(self, query):
        sentences = []  # generate all possible sentences
        vocabs = self.vocabs

        query, types = query.split("\t")

        if query in vocabs:  # remove query in vocbas
            vocabs.remove(query)

        self.idx2vocab = {idx: vocab for idx, vocab in enumerate(vocabs)}
        for vocab in vocabs:
            sentence = f"{self.type_token_dict[types]} {query} [SEP] {vocab}"
            sentences.append(sentence)

        return sentences

    def _generate_batch_sentences(self, sentences):
        size = len(sentences)
        batch_size = self.batch_size
        for idx in range(0, size, batch_size):
            yield sentences[idx : min(idx + batch_size, size)]

    def predict(self, query, n_preds=15):
        self.model.eval()
        # self.model.cuda()
        sentences = self._generate_sentences(query)

        scores = None
        for batched_sents in self._generate_batch_sentences(sentences):
            encodings = self.tokenizer(batched_sents, padding=True, truncation=True)

            ids = np.array(encodings["input_ids"])
            _, indices = np.where(ids == 102)
            indices = indices.reshape(-1, 2)
            query_mask = np.zeros_like(ids)
            hyper_mask = np.zeros_like(ids)

            for r_idx, m_idx in enumerate(indices):
                query_mask[r_idx, 2 : m_idx[0]] = 1
                hyper_mask[r_idx, m_idx[0] + 1 : m_idx[1]] = 1

            input_ids = torch.tensor(encodings["input_ids"]).cuda()
            attention_mask = torch.tensor(encodings["attention_mask"]).cuda()
            query_mask = torch.tensor(query_mask).cuda()
            hyper_mask = torch.tensor(hyper_mask).cuda()

            with torch.no_grad():
                output = self.model(input_ids, attention_mask, query_mask, hyper_mask)

            if scores is None:
                scores = output

            else:
                scores = torch.cat((scores, output), dim=0)

        # sors
        scores = torch.flatten(scores)
        scores, word_indices = torch.sort(scores, descending=True)

        scores, word_indices = (
            scores[:n_preds],
            word_indices[:n_preds],
        )
        scores = scores.detach().cpu().tolist()
        word_indices = word_indices.detach().cpu().tolist()

        words = []
        for idx in word_indices:
            words.append(self.idx2vocab[idx])

        return scores, words
