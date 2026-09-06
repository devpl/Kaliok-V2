# PyCharm Development Run Configurations

Use these run configurations from the PyCharm `Run` menu:

- `Kaliok API` starts FastAPI on `http://127.0.0.1:8001/docs`
- `Kaliok Lab` starts Django on `http://127.0.0.1:8000/rag/laboratory/`

Recommended workflow:

1. Run `Kaliok API`
2. Run `Kaliok Lab`
3. In PyCharm, optionally create a compound configuration named `Kaliok V2`
4. Add both configs to the compound so they start together

Environment used by the shared configs:

- Working directory: project root
- Python interpreter: `.venv`
- `PYTHONPATH`: `.\src`
- `KALIOK_API_BASE_URL`: `http://127.0.0.1:8001` for the lab only
