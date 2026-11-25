"""Mismo entorno OpenMP que el servidor (ver backend/server.py): evita el segfault de libomp en macOS."""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
