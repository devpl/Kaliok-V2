# Composition pipeline — migration 2B

La composition descriptive est portée par `pipeline_binding_nodes`. Un lien
associe un `PipelineBinding` à un `RagTemplateNode` d’une même révision de
pipeline et de template. La capability n’est pas dupliquée : elle est dérivée
du node puis vérifiée contre les `ComponentCapability` du binding.

Les sources de vérité sont séparées ainsi :

- topologie : `rag_template_nodes` et `rag_template_edges` ;
- ressources concrètes : `resource_instances` ;
- composition descriptive : `pipeline_binding_nodes` ;
- projection legacy contrôlée : `pipeline_binding_capabilities` ;
- exécution actuelle : `PipelineManifest` et sa projection legacy.

Une écriture graph-managed doit modifier les liens node, valider la
composition, régénérer explicitement la projection legacy et comparer le
résultat dans la même transaction. La validation ne répare jamais une
divergence silencieusement.

Les trois niveaux sont distincts :

- `structure_valid` vérifie les révisions, les capabilities, les liens, la
  configuration et la cohérence de projection ;
- `activatable` ajoute la couverture des nodes requis et les exigences
  minimales de qualification des ressources ;
- `runtime_ready` ajoute la disponibilité instantanée, la santé des ressources,
  les connexions et la résolution des credentials.

Une ressource momentanément `unreachable` peut donc laisser la composition
structurellement valide tout en empêchant `runtime_ready`.

Le backfill Alembic 2B est une projection technique et ne crée ni audit event,
ni resource instance, ni node. Les futures mutations applicatives utilisent
les actions d’audit `binding_node_added`, `binding_node_removed`,
`binding_node_selection_changed`, `binding_node_configuration_changed`,
`binding_resource_instance_changed` et `binding_node_enabled_changed`.
