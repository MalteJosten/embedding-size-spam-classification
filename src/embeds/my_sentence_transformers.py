import os
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from sacremoses import MosesTokenizer
from sentence_transformers import SentenceTransformer
from embeds.para.models import load_model
from embeds.para.utils import Example

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments: True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SenTransformer(ABC):
    def __init__(self, device: str | None):
        self.device = device
        self.model = None

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    def clean(self) -> None:
        if self.model:
            del self.model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class ALLMiniLML6(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        result = [tensor.item() for tensor in embedding]
        return result


class AllMpnetBaseV2(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode([text])[0]
        result = [tensor.item() for tensor in embedding]
        return result

class Mxbai(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=64)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        result = [tensor.item() for tensor in embedding]
        old_norm = np.linalg.norm(result)
        new_result = [float(part / old_norm) for part in result]

        return new_result


class NomicEmbedText768(SenTransformer):
    def __init__(self, device):
        import torch.nn.functional as F

        super().__init__(device=device)
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(["classification:" + text], convert_to_tensor=True)
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :768]
        embedding = F.normalize(embedding, p=2, dim=1)

        result = embedding.cpu().numpy().tolist()[0]
        return result


class NomicEmbedText512(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(["classification:" + text], convert_to_tensor=True)
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :512]
        embedding = F.normalize(embedding, p=2, dim=1)

        result = embedding.cpu().numpy().tolist()[0]
        return result


class NomicEmbedText256(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(["classification:" + text], convert_to_tensor=True)
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :256]
        embedding = F.normalize(embedding, p=2, dim=1)

        result = embedding.cpu().numpy().tolist()[0]
        return result


class NomicEmbedText128(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(["classification:" + text], convert_to_tensor=True)
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :128]
        embedding = F.normalize(embedding, p=2, dim=1)

        result = embedding.cpu().numpy().tolist()[0]
        return result


class NomicEmbedText64(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(["classification:" + text], convert_to_tensor=True)
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = embedding[:, :64]
        embedding = F.normalize(embedding, p=2, dim=1)

        result = embedding.cpu().numpy().tolist()[0]
        return result


class MultiMiniLML12(SenTransformer):
    def __init__(self, device):
        super().__init__(device=device)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        result = [tensor.item() for tensor in embedding]
        return result


class ParaEmbed(SenTransformer):
    def __init__(self, device: str | None):
        super().__init__(device=device)
        self.model, _ = load_model()


    def embed(self, text: str) -> list[float]:
        entok = MosesTokenizer(lang='en')

        from argparse import Namespace

        new_args = Namespace(batch_size=32, entok=entok, sp=self.model.sp,
                        model=self.model, lower_case=self.model.args.lower_case,
                        tokenize=self.model.args.tokenize)

        embedding = self.para_embed(new_args, self.batcher, [text])
        result = embedding.astype(float)
        result = result / np.linalg.norm(result)
        result = result.tolist()
        return result[0]

    def buffered_read(self, fp, buffer_size):
        buffer = []
        for src_str in fp:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

        if len(buffer) > 0:
            yield buffer

    def para_embed(self, params, batcher, sentences):
        results = []
        for ii in range(0, len(sentences), params.batch_size):
            batch1 = sentences[ii:ii + params.batch_size]
            results.extend(batcher(params, batch1))
        return np.vstack(results)

    def batcher(self, params, batch):
        new_batch = []
        for p in batch:
            if params.tokenize:
                tok = params.entok.tokenize(p, escape=False)
                p = " ".join(tok)
            if params.lower_case:
                p = p.lower()
            p = params.sp.EncodeAsPieces(p)
            p = " ".join(p)
            p = Example(p, params.lower_case)
            p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
            new_batch.append(p)
        x, l = params.model.torchify_batch(new_batch)
        vecs = params.model.encode(x, l)
        return vecs.detach().cpu().numpy()


def get_model(model_name, device=DEVICE):
    if model_name == "lml6":
        model = ALLMiniLML6(device)
    elif model_name == "mpnet":
        model = AllMpnetBaseV2(device)
    elif model_name == "nomic768":
        model = NomicEmbedText768(device)
    elif model_name == "nomic512":
        model = NomicEmbedText512(device)
    elif model_name == "nomic256":
        model = NomicEmbedText256(device)
    elif model_name == "nomic128":
        model = NomicEmbedText128(device)
    elif model_name == "nomic64":
        model = NomicEmbedText64(device)
    elif model_name == "lml12":
        model = MultiMiniLML12(device)
    elif model_name == "mxbai":
        model = Mxbai(device)
    elif model_name == "para":
        model = ParaEmbed(device)
    else:
        print("Specify model name")
        exit()

    return model
