# AGI Knowledge Graph Processing

This project focuses on processing and transforming knowledge graphs, particularly those utilizing Sparse Priming Representations (SPRs). The goal is to parse unstructured knowledge graph data, apply SPR-related transformations, and output structured formats like JSON.

## Overview

The core idea involves using scripts to handle custom knowledge graph formats that incorporate SPRs – a method for concisely encoding complex information. This project includes tools to parse these formats and convert them into standard graph representations or other structured data.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/imndevmodeai/aGintFlux.git
   cd aGintFlux
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The primary script for parsing the custom KG format is `kg_spr_formatter.py`.

**Example: Convert unstructured KG to JSON**

```bash
python kg_spr_formatter.py --input blueprints/unStructured_KG.txt --output blueprints/spr_formatted_kg.json
```

**Example: Convert to edgelist**

```bash
python kg_spr_formatter.py --input blueprints/unStructured_KG.txt --output blueprints/kg_edgelist.txt --format edgelist
```

Refer to the script's help for more options:
```bash
python kg_spr_formatter.py --help
```

## Project Structure

- `code/`: Core implementation files
  - `mastermind_ai.py`: Main AI agent implementation
  - `system_orchestrator.py`: System coordination
  - `workflow_engine.py`: Workflow management
  - `cfp_framework.py`: Comparative Fluxual Processing
  - `quantum_utils.py`: Quantum computing utilities
- `tools/`: Various AI tools and utilities
- `blueprints/`: System configuration files
- `app/`: Web application components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Quantum computing principles inspired by quantum mechanics
- CFP framework based on fluid dynamics and information theory
- Special thanks to contributors and maintainers 