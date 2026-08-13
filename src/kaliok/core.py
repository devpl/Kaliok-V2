from .config import APP_NAME, APP_VERSION


class Kaliok:
    def __init__(self):
        self.name = APP_NAME
        self.version = APP_VERSION
        self.running = False

    def start(self):
        self.running = True
        return f"{self.name} {self.version} est prêt."

    def stop(self):
        self.running = False
        return f"{self.name} est arrêté."
