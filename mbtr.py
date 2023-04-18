from typing import TypeAlias, TypeVar
from collections.abc import Iterable, Hashable, Callable
from itertools import chain
from functools import partial, cache
from operator import itemgetter
import heapq

from sentence_transformers.SentenceTransformer import SentenceTransformer
import nltk
import numpy as np
from numpy import linalg as LA
from numpy.typing import NDArray
import scipy.spatial.distance as spd
from scipy.special import log_softmax
import networkx as nx
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import deal

Sentence:   TypeAlias = str
NDArrayF64: TypeAlias = NDArray[np.float64]
NDArrayF16: TypeAlias = NDArray[np.float16]

H = TypeVar('H', bound=Hashable)
def unique_tuple(seq: Iterable[H]) -> tuple[H, ...]:
    return tuple(dict.fromkeys(seq))

l2_norm_ax1 = partial(LA.norm, ord=2, axis=1, keepdims=True)

class MultiBiasTextRank:

    # Pre-condition contracts on the user-facing methods
    init_contract = deal.chain(
        deal.pre(lambda _: _.n_out_sentences > 0),
        deal.pre(lambda _: _.max_iterations > 0),
        deal.pre(lambda _: 0 <= _.sim_threshold <= 1,
                 message="similarity threshold must be between 0 and 1"),
        deal.pre(lambda _: _.batch_size > 0 and (_.batch_size & (_.batch_size - 1) == 0),
                 message="batch size must be positive and a power of 2"),
    )
    summarizer_contract = deal.chain(
        deal.pre(lambda _: bool(_.documents), message="documents cannot be empty"),
        deal.pre(lambda _: bool(_.queries),   message="queries cannot be empty"),
        deal.pre(lambda _: 0 <= _.alpha <= 1, message="alpha must be between 0 and 1"),
        deal.pre(lambda _: 0 <= _.beta <= 1,  message="beta must be between 0 and 1"),
    )

    @init_contract
    def __init__(self,
        n_out_sentences:         int                  = 1,
        reduction:               str                  = "sum",
        max_iterations:          int                  = 100,
        sim_threshold:           float                = 0.65, # (Kazemi et al. 2020)
        convergence_threshold:   float                = 1e-5,
        batch_size:              int                  = 16,
        sentencized:             bool                 = False,
        encoder:                 str                  = (
            "sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1"),
        specificity_examples:    tuple[str, ...]|None = None,
        sentiment_classifier:    str|None             = (
            # "LiYuan/amazon-review-sentiment-analysis"),
        "ett1112/amazon_sentiment_sample_of_1900_with_summary")
            # ("cardiffnlp/twitter-xlm-roberta-base-sentiment"),
    ) -> None :

        self.n_out_sentences  = n_out_sentences
        self.sim_threshold    = sim_threshold
        self._setup_sentences = (MultiBiasTextRank._curate_sentences
                                 if sentencized
                                 else MultiBiasTextRank._sentencize)
        self._pagerank        = partial(
            nx.pagerank,
            tol      = convergence_threshold,
            max_iter = max_iterations,
        )
        self._encode          = cache(partial(
            SentenceTransformer(encoder).encode,
            batch_size = batch_size,
        ))
        self._polyq_reduce    = MultiBiasTextRank._setup_reduction(reduction)
        self._target_ic       = self._compute_ic(specificity_examples)
        self._senti_pipeline  = MultiBiasTextRank._setup_senti_pipeline(sentiment_classifier)

    @summarizer_contract
    def __call__(self,
        documents:      tuple[str, ...],
        queries:        tuple[str, ...],
        alpha:          float     = 0.1, # (Moubtahij et al. 2023)
        beta:           float     = 0.2, # (Moubtahij et al. 2023)
        sentiment:      bool|None = None,
    ) -> list[str]:

        sentences      = self._setup_sentences(documents)
        sentences_encs = self._encode(sentences)
        queries_encs   = self._encode(queries)
        query_biases   = MultiBiasTextRank._cos_sim_f16(
            queries_encs, sentences_encs
        )
        delta_ic       = self._delta_ic(sentences_encs)
        compound_bias  = self._polyq_reduce(
            query_biases - (beta * delta_ic),
            axis=0,
        ).squeeze()

        if sentiment is not None:
            queries_ic = np.mean(l2_norm_ax1(queries_encs))
            senti_biases = (
                self._sentiment_biases(sentences, sentiment).squeeze()
                / queries_ic # favor the contribution of high information queries
            )
            compound_bias += senti_biases

        compound_bias /= compound_bias.sum()

        if alpha == 0:
            return self._bias_only_sentences(sentences, compound_bias)

        ranked_sentences = self._textrank(
            sentences, compound_bias, sentences_encs, alpha
        )
        return self._summary(ranked_sentences)

    def _summary(self,
        ranked_sentences: dict[Sentence, float]
    ) -> list[Sentence]:

        top_ranked_sentences = heapq.nlargest(
            self.n_out_sentences,
            ranked_sentences.items(),
            key=itemgetter(1)
        )
        return [s for s, _ in top_ranked_sentences]

    def _textrank(self,
        sentences:      tuple[Sentence, ...],
        compound_bias:  NDArrayF16,
        sentences_encs: NDArrayF64,
        alpha:          float,
    ) -> dict[Sentence, float]:

        biased_nodes   = (
            None if np.allclose(compound_bias, 0)
            else
            dict(zip(sentences, compound_bias))
        )
        adj_mat        = MultiBiasTextRank._adjacency_matrix(sentences_encs)
        adj_mat[adj_mat < self.sim_threshold] = 0

        textrank_graph = nx.relabel_nodes(
            nx.from_numpy_array(adj_mat),
            dict(enumerate(sentences)),
            copy=False
        )
        return self._pagerank(
            textrank_graph,
            alpha           = alpha,
            personalization = biased_nodes, # normalized in nx.pagerank
        )

    def _compute_ic(self, specificity_examples: tuple[str, ...]|None) -> float|None:

        if specificity_examples is None:
            return None

        _examples_encs = self._encode(specificity_examples)
        _ic_hints      = l2_norm_ax1(_examples_encs)

        return np.mean(_ic_hints.squeeze()).item()

    def _bias_only_sentences(self,
        sentences: tuple[Sentence, ...],
        compound_bias: NDArrayF16,
    ) -> list[Sentence]:

        top_ranked_sentences = heapq.nlargest(
            self.n_out_sentences,
            zip(sentences, compound_bias),
            key=itemgetter(1),
        )
        return [s for s, _ in top_ranked_sentences]

    def _delta_ic(self, sentence_encodings: NDArrayF64) -> NDArrayF16:

        out_shape = (1, len(sentence_encodings))

        if self._target_ic is None:
            return np.zeros(out_shape, dtype=np.float16)

        return np.abs(
            l2_norm_ax1(sentence_encodings) - self._target_ic,
            dtype=np.float16,
        ).reshape(out_shape)

    def _sentiment_biases(self,
        sentences: tuple[str, ...], sentiment: bool
    ) -> NDArrayF16:

        assert self._senti_pipeline is not None,\
            "Did you forget to provide a sentiment classifier?"

        with torch.no_grad():
            senti_biases_ = self._senti_pipeline(list(sentences))

        # NOTE: HF sentiment classifiers don't have a standard API.
        # Loosen this coupling if there are users that request it.
        queried_senti = "LABEL_1" if sentiment else "LABEL_0"
        senti_biases = [
            s["score"] if s["label"] == queried_senti
            else 0
            for s in senti_biases_
        ]
        return np.array(senti_biases, dtype=np.float16)

    @staticmethod
    def _setup_reduction(reduction: str) -> Callable:

        if reduction in {"jointprob", "logsum"}:
            return (lambda bias_weights, axis=0:
                    log_softmax(bias_weights, axis=1).sum(axis=axis))
                    # Softmax handles large and negative entries to log

        # Numpy reductions e.g. sum, max, min, mean, median...
        return getattr(np, reduction)

    @staticmethod
    def _setup_senti_pipeline(
        sentiment_classifier: str|None
    ) -> Callable[[Iterable[Sentence]], NDArrayF64] | None:

        if sentiment_classifier is None:
            return None

        _bias_tokenizer          = partial(
            AutoTokenizer.from_pretrained(sentiment_classifier),
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 512,
        )
        _bias_model              = AutoModelForSequenceClassification.from_pretrained(
            sentiment_classifier,
            num_labels=2 # NOTE: Loosen coupling if requested
        )

        return pipeline(
            "sentiment-analysis",
            tokenizer = _bias_tokenizer,
            model     = _bias_model,
        )

    @staticmethod
    def _cos_sim_f16(m: NDArrayF64, n: NDArrayF64) -> NDArrayF16:
        return np.matmul(
            m / l2_norm_ax1(m),
            (n / l2_norm_ax1(n)).T,
            dtype=np.float16,
        )

    @staticmethod
    def _adjacency_matrix(sentences_encs: NDArrayF64) -> NDArrayF16:
        return spd.squareform(
            1 - spd.pdist(sentences_encs, metric="cosine").astype(np.float16)
        )

    @staticmethod
    def _curate_sentences(sentences: Iterable[Sentence]) -> tuple[Sentence, ...]:
        return unique_tuple(filter(None, map(str.strip, sentences)))

    @staticmethod
    def _sentencize(documents: tuple[str, ...]) -> tuple[Sentence, ...]:

        _sentences = chain.from_iterable(
            map(nltk.sent_tokenize, documents)
        )
        return MultiBiasTextRank._curate_sentences(_sentences)
