#!/bin/bash
# -----------------------------
# Fully self-contained SLURM submission
# -----------------------------

# Parameters
L_LIST=(4 8 16 32 64 128 256)
DT_LIST=(0.5)

RAMP_MIN=0.1
RAMP_MAX=10
RAMP_STEPS=1000
FILLING=0.5

# Total number of tasks
N_L=${#L_LIST[@]}
N_DT=${#DT_LIST[@]}
TOTAL_TASKS=$(( N_L * N_DT ))

# Submit the array job dynamically
sbatch --array=0-$((TOTAL_TASKS-1)) <<'EOF'
#!/bin/bash
#SBATCH --job-name=adiabatic
#SBATCH --time=6-23:00:00
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%A_%a.err

# -----------------------------
# Array index
# -----------------------------
ARRAY_IDX=$SLURM_ARRAY_TASK_ID

# Parameters must be repeated inside the SLURM task
L_LIST=(4 8 16 32 64 128 256)
DT_LIST=(0.5)

N_DT=${#DT_LIST[@]}
L_IDX=$((ARRAY_IDX / N_DT))
DT_IDX=$((ARRAY_IDX % N_DT))

L=${L_LIST[$L_IDX]}
DT=${DT_LIST[$DT_IDX]}

# Ramp times
RAMP_MIN=0.1
RAMP_MAX=10
RAMP_STEPS=1000
FILLING=0.5

# -----------------------------
# Load Python 3 safely
# -----------------------------
module purge
module load anaconda
source activate base
unset PYTHONHOME PYTHONPATH

# -----------------------------
# Run Python script
# -----------------------------
python src/run_adiabatic.py \
    --L $L \
    --dt $DT \
    --ramp_min $RAMP_MIN \
    --ramp_max $RAMP_MAX \
    --ramp_steps $RAMP_STEPS \
    --filling $FILLING
EOF