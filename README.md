# DeepSyncNet

A python  implementation of our manuscript "Deep synergetic modeling of slow-fast dynamics in complex systems”. Thank you for reviewing our manuscript.

## Environment Setup
```bash
conda create --name <env> --file requirement.txt
```

## Usage

choose the system: `FHN`, `FHNv`, `HalfMoon`, `Lorenz`

```shell
# ours
./scripts/{system}/OURS.sh

# baseline
./scripts/{system}/LED.sh
./scripts/{system}/NeuralODE.sh
./scripts/{system}/DeepKoopman.sh
./scripts/{system}/SINDy.sh
./scripts/{system}/RC.sh
```

We recommend turning on the **--parallel** option to enable parallel execution of programs with different random seeds to improve test efficiency. Please be careful to choose the suitable number of random seeds  **--seed_num** according to your computational and cache resources. The result of the experiment should be an average of multiple random seeds.
