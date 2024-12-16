# Monte Carlo Simulation

2D Ising Model using Metropolis Algorithm

### Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup

Clone the repository and navigate to the project root directory.

Run the Docker Container `docker run -p 8888:8888 -v /path/to/ising-simulation:/app ising-simulation` **_If using JupyterNotebook_**

**Install dependencies**

Run `poetry install` to install dependencies.

## Run the Simulation

Activate the virtual environment:

`poetry shell`

Run `python setup.py build_ext --inplace` if using Cython to build the module. **Recommended**

Execute the simulation:

`python monte-carlo/simulation.py` / `python monte-carlo/no-cy-sim.py` if not using Cython.

To stop the simulation, press `⌘+C` on macOS or `Ctrl+C` on Windows/Linux.

##### Installing Prerequisites

Install Homebrew (for macOS):

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
