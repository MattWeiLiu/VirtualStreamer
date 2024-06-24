import time
import random
import torch

from collections import deque
from .cache import LRUCache
from bisect import insort_left
from transformers import AutoTokenizer, BertForNextSentencePrediction
from torch.nn import functional


def shuffle(cache, sentence):
    cache.append(sentence)
    random.shuffle(cache)
    while len(cache) > 10:
        cache.pop()
    return cache


class WeightedRankingCache:
    def __init__(self, capacity):
        # Model initial
        model_name = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)

        self.w_t = 0.25    # time weight
        self.w_l = 0.25    # length weight
        self.w_nsp = 0.25  # Next Sentence Prediction weight
        self.w_v = 0.25    # viewer weight

        # prev response
        self.prev_response = ""

        # Viewer cache
        self.lru_cache = LRUCache(capacity)

    def rerank(self, cache, sentence, open_id, timestamp):
        """
        cache[[int, object]] = [score, {"content": xxxxxx,
                                      "timestamp": xxxxxx,
                                      "length": 0.5,
                                      "openID": xxxxxx}]
        """
        if len(cache) > 0:
            next_cache = deque([])
            now = time.time()
            for _, info in cache:
                score = 0
                if now - info["timestamp"] < 90:
                    # Time
                    score += self.w_t / (now - info["timestamp"])

                    # Length
                    score += self.w_l * info["length"]

                    # NSP
                    score += self.w_nsp * self.next_sentence_prediction(info["content"])

                    # Viewer
                    prev_timestamp = self.lru_cache.get(open_id)
                    if prev_timestamp and prev_timestamp > info["timestamp"]:
                        continue
                    elif prev_timestamp:
                        score += self.w_v / (info["timestamp"] - prev_timestamp)

                    insort_left(next_cache, [score, info], key=lambda x: -1 * x[0])

            info = {"content": sentence, "timestamp": timestamp, "openID": open_id}
            score = self.w_t
            info["length"] = self.w_l * (min(len(sentence), 50) / 50)
            score += info["length"]
            score += self.w_nsp * self.next_sentence_prediction(sentence)
            prev_timestamp = self.lru_cache.get(open_id)
            if prev_timestamp:
                score += self.w_v / (now - prev_timestamp)
            insort_left(next_cache, [score, info], key=lambda x: -1 * x[0])
            return next_cache
        else:
            info = {"content": sentence, "timestamp": timestamp, "openID": open_id}
            info["length"] = self.w_l * (min(len(sentence), 50) / 50)
            return deque([[0, info]])

    def put_viewer(self, open_id, timestamp):
        self.lru_cache.put(open_id, timestamp)

    def next_sentence_prediction(self, next_sentence):
        encoding = self.tokenizer(self.prev_response, next_sentence, return_tensors="pt")
        outputs = self.model(**encoding, labels=torch.LongTensor([1]))
        logits = outputs.logits
        log_prob = functional.log_softmax(logits)
        return torch.exp(log_prob).data.cpu().numpy()[0, 0]
