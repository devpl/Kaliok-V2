from .config import APP_NAME, APP_VERSION


class Kaliok:
    def __init__(self):
        self.name = APP_NAME
        self.version = APP_VERSION

    def start(self):
        return f"{self.name} {self.version} est prêt."
