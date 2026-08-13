from kaliok.core import Kaliok


def test_start():
    kaliok = Kaliok()

    message = kaliok.start()

    assert kaliok.running is True
    assert message == "kaliok V2 0.1.0 est prêt."
