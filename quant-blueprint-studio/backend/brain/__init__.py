"""Segundo cerebro: prompt → validated blueprint, and grounded explanations of results.

Local only: BM25 retrieval in pure Python and an optional Ollama model on 127.0.0.1.
The model never writes code or invents node kinds; its plan is compiled against the
catalog the interface sends, and every graph passes the engine's structural checks.
"""
