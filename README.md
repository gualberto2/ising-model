# Monte Carlo Simulation

2D Ising Model using Metropolis Algorithm

### Prerequisites

- Docker

### Setup

Make sure you are in the directory of the repository.

`docker build -t ising-simulation .`

`docker run --rm -v "$(pwd)/plots:/app/plots" ising-simulation`

```
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  ising-simulation \
  python monte-carlo/simulation.py \
  --full-csv-path "output/full_results.csv" \
  --top-n 10 \
  --csv-path "output/top_results.csv"
```
