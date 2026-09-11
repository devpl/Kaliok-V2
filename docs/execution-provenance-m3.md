# Migration 3 — provenance d'exécution

Migration 3 ajoute une couche d'orchestration persistée sans changer la
sémantique ni le runtime actuel de `ProcessingRun`.

```text
Execution
  └─ ExecutionStep (un lancement réel de RagTemplateNode)
       ├─ ProcessingRun 0..N
       ├─ INPUT  → artefact réel
       └─ OUTPUT → artefact réel
```

`ProcessingRun` reste un traitement technique producteur. `Execution` est
l'orchestration globale d'un sous-graphe ou du graphe RAG. `ExecutionStep`
capture le node, le binding éventuel, la ressource éventuelle et la
configuration effectivement utilisée. Une `ConfigurationRevision` reste la
version canonique ; le JSONB du step est son snapshot effectif d'exécution.

La chaîne fonctionnelle est donc représentable sans invariant de run partagé :

```text
ProcessingRun N --OUTPUT--> NormalizedContentUnit
ExecutionStep D --INPUT---> NormalizedContentUnit
ProcessingRun D --OUTPUT--> DiscoveredCandidate
ExecutionStep R --INPUT---> DiscoveredCandidate
ProcessingRun R --OUTPUT--> Entity

N != D != R
```

Le périmètre spécialisé de M3 couvre `ContentBlock` (entrée pratique de la
normalisation), `NormalizedContentUnit`, `DiscoveredCandidate` et `Entity`.
Chaque lien pointe vers la vraie PK PostgreSQL. L'enveloppe commune porte le
rôle `input` ou `output` et l'`ArtifactType`. L'exclusivité « exactement une
table spécialisée par enveloppe » est garantie par le service d'écriture, pas
par un trigger inter-table ; ce renforcement reste une dette explicite.

Il n'existe aucun backfill : aucun faux `ProcessingRun`, `Execution`,
`ExecutionStep` ou lien d'artefact n'est créé. Les lignes historiques, y
compris les unités normalisées sans run, restent valides avec une provenance
partielle. `execution_group_id` est conservé comme champ transitoire.

Les exécutions de scope `lab` restent identifiables et ne rendent jamais un
artefact « current ». La promotion production sera une mutation métier
séparée et auditée ultérieurement. Les changements techniques de statut et la
production d'artefacts ne créent pas d'`audit_event` : ce journal demeure
réservé aux mutations métier/utilisateur.

Les suppressions fonctionnelles d'Execution ne font pas partie de l'API. Les
enfants purement techniques peuvent être supprimés en cascade lors d'une
opération administrative exceptionnelle ; supprimer un step détache ses
`ProcessingRun` (`ON DELETE SET NULL`) et ne supprime jamais les artefacts
métier ciblés.

Ces primitives sont directement réutilisables par le futur parcours « Tester
cette étape ». Le planner de graphe, l'exécution de zone et la promotion sont
hors périmètre de M3.
