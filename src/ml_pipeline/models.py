from typing import Literal


class EmbeddingModel:
    def __new__(
        cls,
        method: Literal[
            "sentence_transformer",
            "model2vec",
            "huggingface",
            "flair",
            "spacy",
            "use",
            "gensim",
            "sklearn",
            "openai",
            "cohere",
            "multimodal",
            "customback",
            "customembed",
            "tfidf",
        ],
        **kwargs,
    ):
        if method == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            try:
                return SentenceTransformer(**kwargs)
            except Exception:
                raise ValueError(
                    f"Недопустимое имя модели: '{kwargs['model_name_or_path']}'"
                )
        elif method == "model2v ec":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "huggingface":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "flair":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "spacy":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "use":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "gensim":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "sklearn":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "openai":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "cohere":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "multimodal":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "customback":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "customembed":
            print("In process, plese use sentence_transformer")
            return None
        elif method == "tfidf":
            print("In process, plese use sentence_transformer")
            return None
        else:
            raise ValueError(f"Недопустимое имя метода: '{method}'")


class ReductionModel:
    def __new__(
        cls,
        method: Literal["umap", "svd", "pca", "skip", "custom"] = "umap",
        **kwargs,
    ):
        if method == "umap" or method == "custom":
            from umap import UMAP

            if method == "custom":
                print("In process, now used umap")
            return UMAP(**kwargs)
        elif method == "svd":
            from sklearn.decomposition import TruncatedSVD

            return TruncatedSVD(**kwargs)
        elif method == "pca":
            from sklearn.decomposition import PCA

            return PCA(**kwargs)
        elif method == "skip":
            from bertopic.dimensionality import BaseDimensionalityReduction

            return BaseDimensionalityReduction()
        else:
            raise ValueError(f"Недопустимый метод: {method}")


class ClusteringModel:
    def __new__(
        cls,
        model_name: Literal["hdbscan", "kmeans", "agglomerative", "custom"] = "hdbscan",
        **kwargs,
    ):
        if model_name == "hdbscan" or model_name == "custom":
            from sklearn.cluster import HDBSCAN

            if model_name == "custom":
                print("In process, now used hdbscan")
            return HDBSCAN(**kwargs)
        elif model_name == "kmeans":
            from sklearn.cluster import KMeans

            return KMeans(**kwargs)
        elif model_name == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            return AgglomerativeClustering(**kwargs)
        else:
            raise ValueError(f"Недопустимый метод: {model_name}")


class VectorizerModel:
    def __new__(
        cls,
        model_name: Literal["count", "countonline"] = "count",
        **kwargs,
    ):
        if model_name == "count":
            from sklearn.feature_extraction.text import CountVectorizer

            return CountVectorizer(**kwargs)
        elif model_name == "countonline":
            from bertopic.vectorizers import OnlineCountVectorizer

            return OnlineCountVectorizer(**kwargs)
        else:
            raise ValueError(f"Недопустимый метод: {model_name}")


class RepresentationModel:
    def __new__(
        cls,
        method: Literal[
            "keybert",
            "PartOfSpeech",
            "MaximalMarginalRelevance",
            "zero-shot",
            "custom",
            "openai",
            "transformers",
            "gguf",
            "llama-manual-quantization",
        ] = "keybert",
        **kwargs,
    ):
        if method == "keybert":
            from bertopic.representation import KeyBERTInspired

            return KeyBERTInspired()
        elif method == "countonline":
            pass
