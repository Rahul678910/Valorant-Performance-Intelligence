# Contributing to Valorant Performance Intelligence

Thank you for your interest in contributing to **Valorant Performance Intelligence**! We welcome community contributions, from reporting bugs and expanding tactical features to improving model accuracy and dashboard visualizations.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/valorant-performance-intelligence.git
   cd valorant-performance-intelligence
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Tests:**
   ```bash
   pytest -v tests/
   ```

---

## Workflow & Coding Standards

1. Create a descriptive feature branch from `main`:
   ```bash
   git checkout -b feature/agent-synergy-matrix
   ```
2. Adhere to PEP 8 styling conventions with explicit type hints and docstrings.
3. Ensure no data leakage when implementing new ML features.
4. Add corresponding unit tests in `tests/` for any new logic.
5. Submit a Pull Request following the PR template.
