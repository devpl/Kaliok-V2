# Kaliok V2

Kaliok V2 est le nouveau socle du projet Kaliok.

## Version

0.1.0

## Python

Python 3.13 ou supérieur.

## Démarrage

```bash
python main.py
...
```

## Observabilité OpenTelemetry

L'observabilité OpenTelemetry reste facultative. Sans endpoint OTLP explicite,
`create_opentelemetry_observer()` retourne un observer inactif.

Variables standard prises en charge :

```text
OTEL_SDK_DISABLED=false
OTEL_SERVICE_NAME=kaliok
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_ENDPOINT=<endpoint du collector>
```

`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` et
`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` peuvent être utilisés pour une
configuration spécifique aux traces. Aucune adresse de collector n'est
définie par Kaliok.
