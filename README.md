# Monte Carlo Simulation

2D Ising Model using Metropolis Algorithm

### Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup

Clone the repository and navigate to the project root directory.

**Install dependencies**

Run `poetry install` to install dependencies.

Run `python setup.py build_ext --inplace` if using Cython to build the module. **Recommended**

## Run the Simulation

Activate the virtual environment:
Ensure you're in the poetry shell:

`poetry shell`

Execute the simulation:

`python monte-carlo/simulation.py`

To stop the simulation, press `⌘+C` on macOS or `Ctrl+C` on Windows/Linux.

##### Installing Prerequisites

Install Homebrew (for macOS):

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
