# Running Locally:
python src/run_adiabatic.py --L 4 --dt 0.5 --ramp_times 10 20 50

# Running on Cluster:
sbatch scripts/submit_adiabatic.sh

# Data Saving
Data is saved at: data/adiabatic/adiabatic_L{L}_dt{dt}_filling{filling}.npz