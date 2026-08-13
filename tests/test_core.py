from kaliok.config import APP_NAME, APP_VERSION
from kaliok.core import Kaliok


def test_start():
    kaliok = Kaliok()

    message = kaliok.start()

    assert kaliok.running is True
    assert message == f"{APP_NAME} {APP_VERSION} est prêt."


def test_stop():
    kaliok = Kaliok()
    kaliok.start()

    message = kaliok.stop()

    assert kaliok.running is False
    assert message == f"{APP_NAME} est arrêté."

def test_status_when_stopped():
    kaliok = Kaliok()

    assert kaliok.status() == f"{APP_NAME} est arrêté."


def test_status_when_running():
    kaliok = Kaliok()
    kaliok.start()

    assert kaliok.status() == f"{APP_NAME} est en cours d'exécution."
