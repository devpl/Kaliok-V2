# Expérimenter Candidate Discovery sous PowerShell

Le dictionnaire JSON est lu directement en UTF-8. Ne transmettez pas du code Python contenant des accents avec un here-string PowerShell envoyé sur l’entrée standard : selon la configuration de la console, `Nouméa` peut alors devenir `Noum?a` avant même d’atteindre Python.

Configurez Python et la sortie de console en UTF-8, puis appelez le fichier CLI directement :

```powershell
$env:PYTHONUTF8 = "1"
chcp 65001
.\.venv\Scripts\python.exe -m tools.run_candidate_discovery --filename "rapport-d-activit--s-2013-de-la-ctc-NC_1.pdf" --dictionary config\discovery\development_dictionary.json --dry-run
```

La recherche par nom évite de recopier un UUID. Si plusieurs versions correspondent, l’outil les affiche et demande un UUID explicite :

```powershell
.\.venv\Scripts\python.exe -m tools.run_candidate_discovery --document-version-id "UUID" --normalization-run-id "UUID" --dictionary config\discovery\development_dictionary.json --dry-run
```

Le mode par défaut est `--dry-run` : le résultat est affiché puis la transaction est annulée. Pour rendre volontairement l’expérience visible dans le Lab :

```powershell
.\.venv\Scripts\python.exe -m tools.run_candidate_discovery --filename "nom-exact.pdf" --dictionary config\discovery\development_dictionary.json --commit
```

Ouvrez ensuite **Laboratoire RAG → Découverte documentaire**, choisissez le document, puis l’analyse la plus récente. L’outil affiche toujours le document, les générations de normalisation et de découverte, les détecteurs, les comptes par type, les valeurs regroupées et les pages concernées.
