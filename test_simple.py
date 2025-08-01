#!/usr/bin/env python3

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.queens_solver import QueensSolver

def test_simple():
    solver = QueensSolver(verbose=False)
    result = solver.solve_from_image('tests/boards/solvable/board4.png', quiet_mode=False)
    print(f"Test result: {result}")

if __name__ == "__main__":
    test_simple()
