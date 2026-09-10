#!/usr/bin/env python3
"""
CLI entry point to interact with the AI Customer Support Agent.
Usage:
    python run_agent.py --message "Where is my order? It was supposed to arrive yesterday!"
"""

import sys
from src.agent import main

if __name__ == "__main__":
    main()
