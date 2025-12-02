"""
Módulo de algoritmos de deconvolução.
"""

from src.algorithms.wiener import Wiener
from .base import DeconvolutionAlgorithm
from .richardson_lucy import RichardsonLucy

# Registro de algoritmos disponíveis
ALGORITHMS = {
    'richardson_lucy': RichardsonLucy,
    'wiener': Wiener,
}

def get_algorithm(name):
    """
    Retorna uma instância do algoritmo pelo nome.
    
    Args:
        name: Nome do algoritmo
    
    Returns:
        Instância do algoritmo
    
    Raises:
        ValueError: Se o algoritmo não for encontrado
    """
    if name not in ALGORITHMS:
        available = ', '.join(ALGORITHMS.keys())
        raise ValueError(f"Algoritmo '{name}' não encontrado. Algoritmos disponíveis: {available}")
    
    return ALGORITHMS[name]()

def list_algorithms():
    """
    Retorna lista de nomes de algoritmos disponíveis.
    
    Returns:
        Lista de strings com nomes dos algoritmos
    """
    return list(ALGORITHMS.keys())

