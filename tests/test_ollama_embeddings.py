from __future__ import annotations

import pytest

from kaliok.embeddings import ollama


class FakeResponse:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def raise_for_status(self):
        return None

    def json(self):
        return {"embeddings": self._embeddings}


def test_embed_texts_empty_input_does_not_call_ollama(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Aucun appel HTTP attendu pour un lot vide.")

    monkeypatch.setattr(ollama.requests, "post", fail)

    assert ollama.embed_texts([], model="qwen3-embedding:4b") == []


def test_embed_text_keeps_single_text_behavior(monkeypatch):
    calls = []
    expected = [0.0] * ollama.EMBEDDING_DIMENSIONS

    def fake_post(url, *, json, timeout):
        calls.append(json)
        return FakeResponse([expected])

    monkeypatch.setattr(ollama.requests, "post", fake_post)

    assert ollama.embed_text("texte courant") == expected
    assert calls == [
        {"model": ollama.EMBEDDING_MODEL, "input": ["texte courant"]}
    ]


@pytest.mark.parametrize(
    ("text_count", "expected_batch_sizes"),
    [
        (32, [32]),
        (33, [32, 1]),
        (240, [*([32] * 7), 16]),
        (478, [*([32] * 14), 30]),
    ],
)
def test_embed_texts_batches_preserve_size_order_and_explicit_model(
    monkeypatch,
    text_count,
    expected_batch_sizes,
):
    texts = [f"texte-{index}" for index in range(text_count)]
    calls = []

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse(
            [
                [float(text.removeprefix("texte-")), 1.0]
                for text in json["input"]
            ]
        )

    monkeypatch.setattr(ollama.requests, "post", fake_post)

    actual = ollama.embed_texts(texts, model="qwen3-embedding:4b")

    assert actual == [[float(index), 1.0] for index in range(text_count)]
    assert [len(call[1]["input"]) for call in calls] == expected_batch_sizes
    assert [text for call in calls for text in call[1]["input"]] == texts
    assert all(
        call[0] == f"{ollama.OLLAMA_URL}/api/embed"
        and call[1]["model"] == "qwen3-embedding:4b"
        and call[2] == 300
        for call in calls
    )


def test_embed_texts_stops_on_intermediate_batch_error(monkeypatch):
    calls = 0

    def fake_post(url, *, json, timeout):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("lot intermédiaire en échec")
        return FakeResponse(
            [[float(index), 1.0] for index, _ in enumerate(json["input"])]
        )

    monkeypatch.setattr(ollama.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="lot intermédiaire"):
        ollama.embed_texts(
            [f"texte-{index}" for index in range(70)],
            model="qwen3-embedding:4b",
        )

    assert calls == 2


def test_embed_texts_rejects_dimensions_inconsistent_between_batches(
    monkeypatch,
):
    calls = 0

    def fake_post(url, *, json, timeout):
        nonlocal calls
        calls += 1
        dimensions = 2 if calls == 1 else 3
        return FakeResponse(
            [[0.0] * dimensions for _ in json["input"]]
        )

    monkeypatch.setattr(ollama.requests, "post", fake_post)

    with pytest.raises(ValueError, match="Dimensions.*hétérogènes"):
        ollama.embed_texts(
            [f"texte-{index}" for index in range(33)],
            model="qwen3-embedding:4b",
        )


def test_embed_texts_validates_returned_count_for_each_batch(monkeypatch):
    calls = 0

    def fake_post(url, *, json, timeout):
        nonlocal calls
        calls += 1
        returned = len(json["input"]) - (1 if calls == 2 else 0)
        return FakeResponse([[0.0, 1.0] for _ in range(returned)])

    monkeypatch.setattr(ollama.requests, "post", fake_post)

    with pytest.raises(ValueError, match="lot 2/2"):
        ollama.embed_texts(
            [f"texte-{index}" for index in range(33)],
            model="qwen3-embedding:4b",
        )


def test_embed_texts_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="strictement positif"):
        ollama.embed_texts(["texte"], batch_size=0)
