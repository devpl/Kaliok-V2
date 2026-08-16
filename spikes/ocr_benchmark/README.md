# Benchmark OCR - kaliok V2

## Objectif

Comparer plusieurs moteurs OCR sur de vrais PDF scannés afin de choisir le moteur principal de kaliok.

## Configuration commune

- PDF rendus avec PDFium
- Scale : 1
- CPU
- Documents administratifs réels
- 3 PDF
- 14 pages au total

## RapidOCR + ONNX Runtime

### Document 1
- Pages : 4
- Temps total : 8.591 s
- Moyenne/page : 2.148 s

### Document 2
- Pages : 6
- Temps total : 15.658 s
- Moyenne/page : 2.610 s

### Document 3
- Pages : 4
- Temps total : 11.339 s
- Moyenne/page : 2.835 s

### Total
- Pages : 14
- Temps total : 35.588 s
- Moyenne/page : 2.542 s

## PaddleOCR

### Document 1
- Pages : 4
- Temps total : 35.702 s
- Moyenne/page : 8.925 s

### Document 2
- Pages : 6
- Temps total : 55.955 s
- Moyenne/page : 9.326 s

### Document 3
- Pages : 4
- Temps total : 44.733 s
- Moyenne/page : 11.183 s

### Total
- Pages : 14
- Temps total : 136.390 s
- Moyenne/page : 9.742 s

## Conclusion

RapidOCR + ONNX Runtime est retenu comme candidat principal pour l'OCR CPU de kaliok.

Avantages observés :
- environ 3.8 fois plus rapide que PaddleOCR sur ce corpus ;
- très bonne qualité sur les informations structurées ;
- installation simple avec pip ;
- intégration directe possible dans FastAPI ;
- pas de dépendance Docker nécessaire pour l'exécution courante.

PaddleOCR reste intéressant comme moteur de référence ou fallback qualité.

## Pipeline retenu provisoirement

PDF
→ détection couche texte
→ PDFium si texte natif
→ PDFium scale 1 + RapidOCR si scan
→ structure commune
→ chunking
→ embeddings
→ pgvector
→ RAG
