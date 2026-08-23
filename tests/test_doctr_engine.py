from types import SimpleNamespace

import pytest
from PIL import Image

from kaliok.ocr.base import OcrEngine
from kaliok.ocr.doctr_engine import DocTrEngine


def test_doctr_engine_implements_ocr_engine():
    assert issubclass(DocTrEngine, OcrEngine)


def test_doctr_returns_one_result_per_line_with_pixel_geometry():
    engine = DocTrEngine.__new__(DocTrEngine)

    words_line_1 = [
        SimpleNamespace(
            value="PERMISSION",
            confidence=0.90,
        ),
        SimpleNamespace(
            value="DE",
            confidence=0.80,
        ),
        SimpleNamespace(
            value="VOIRIE",
            confidence=1.00,
        ),
    ]

    words_line_2 = [
        SimpleNamespace(
            value="2026",
            confidence=0.95,
        ),
    ]

    line_1 = SimpleNamespace(
        words=words_line_1,
        geometry=(
            (0.10, 0.20),
            (0.60, 0.30),
        ),
    )

    line_2 = SimpleNamespace(
        words=words_line_2,
        geometry=(
            (0.70, 0.50),
            (0.90, 0.60),
        ),
    )

    fake_result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                blocks=[
                    SimpleNamespace(
                        lines=[
                            line_1,
                            line_2,
                        ]
                    )
                ]
            )
        ]
    )

    class FakePredictor:
        def __call__(self, images):
            assert len(images) == 1

            assert images[0].shape == (
                1000,
                2000,
                3,
            )

            return fake_result

    engine.engine = FakePredictor()

    image = Image.new(
        "RGB",
        (2000, 1000),
    )

    results = engine.recognize(image)

    assert len(results) == 2

    first = results[0]

    assert first.text == "PERMISSION DE VOIRIE"
    assert first.confidence == pytest.approx(0.9)
    assert first.coordinate_system == "pixels"

    assert first.bbox_x == pytest.approx(200.0)
    assert first.bbox_y == pytest.approx(200.0)
    assert first.bbox_width == pytest.approx(1000.0)
    assert first.bbox_height == pytest.approx(100.0)

    second = results[1]

    assert second.text == "2026"
    assert second.confidence == pytest.approx(0.95)
    assert second.coordinate_system == "pixels"

    assert second.bbox_x == pytest.approx(1400.0)
    assert second.bbox_y == pytest.approx(500.0)
    assert second.bbox_width == pytest.approx(400.0)
    assert second.bbox_height == pytest.approx(100.0)


def test_doctr_line_without_geometry_keeps_text():
    engine = DocTrEngine.__new__(DocTrEngine)

    fake_result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                blocks=[
                    SimpleNamespace(
                        lines=[
                            SimpleNamespace(
                                words=[
                                    SimpleNamespace(
                                        value="Texte",
                                        confidence=0.8,
                                    ),
                                    SimpleNamespace(
                                        value="sans géométrie",
                                        confidence=1.0,
                                    ),
                                ],
                                geometry=None,
                            )
                        ]
                    )
                ]
            )
        ]
    )

    engine.engine = lambda images: fake_result

    results = engine.recognize(
        Image.new(
            "RGB",
            (100, 100),
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.text == "Texte sans géométrie"
    assert result.confidence == pytest.approx(0.9)

    assert result.bbox_x is None
    assert result.bbox_y is None
    assert result.bbox_width is None
    assert result.bbox_height is None
    assert result.coordinate_system is None