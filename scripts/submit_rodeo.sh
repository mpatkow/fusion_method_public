#!/bin/bash
# ============================================================
# SLURM submission for run_rodeo.py
#
# Two independent submission blocks:
#   1) submit_no_precond   – adiabatic_time = 0
#   2) submit_precond      – adiabatic_time > 0 (state is cached
#                            automatically; re-runs with different
#                            cuts never redo the adiabatic ramp)
#
# Uncomment whichever block(s) you want to submit, or call
# the functions directly:
#   bash submit_rodeo.sh no_precond
#   bash submit_rodeo.sh precond
#   bash submit_rodeo.sh all
# ============================================================

# ============================================================
# ===  SHARED PARAMETERS  ====================================
# ============================================================

L_LIST=(4 8 16)

FILLING=0.5

# Rodeo total-time grid  (linspace; use rodeo_spacing=log for logspace)
RODEO_MIN=1
RODEO_MAX=200
RODEO_STEPS=100
RODEO_SPACING="log"   # "linear" | "log"
RODEO_TIMES_IN_T0="false" # 'false' | "true"

RODEO_DT=0.01

# alpha = 1 + ALPHA_COEFF * pi / (gap * T) * 1/L
# Emperically we see that we want alpha-1 to be:
#   L=4   -> 0.7
#   L=8   -> 0.35
#   L=16  -> 0.175
#   L=32  -> 0.0875
#   L=64  -> 0.04375
#   L=128 -> 0.021875
# 
# This suggests we should take ALPHA_COEFF=2.8
ALPHA_COEFF=2.8

# All cut values you might ever want (script saves one file per cut;
# you can submit only a subset via TIME_CUTS_NO_PRECOND /
# TIME_CUTS_PRECOND below without re-running other cuts).
ALL_CUTS=(3 5 7 9 11)

# ============================================================
# === NO-PRECONDITIONING block ================================
# ============================================================

# Which cuts to run in the no-precond jobs
TIME_CUTS_NO_PRECOND=(3)

# (adiabatic_time is fixed at 0 for this block)

submit_no_precond() {
    local N_L=${#L_LIST[@]}
    local TOTAL_TASKS=$N_L

    echo "Submitting no-precond array: ${TOTAL_TASKS} tasks"

    sbatch --array=0-$((TOTAL_TASKS - 1)) <<HEREDOC
#!/bin/bash
#SBATCH --job-name=rodeo_noprecond
#SBATCH --time=6-23:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/rodeo_noprecond-%A_%a.out
#SBATCH --error=slurm_logs/rodeo_noprecond-%A_%a.err

ARRAY_IDX=\$SLURM_ARRAY_TASK_ID

L_LIST=(${L_LIST[@]})
L=\${L_LIST[\$ARRAY_IDX]}

module purge
module load anaconda
source activate base
unset PYTHONHOME PYTHONPATH

mkdir -p slurm_logs data/rodeo data/precond

python src/run_rodeo.py \\
    --L \$L \\
    --filling ${FILLING} \\
    --adiabatic_time 0.0 \\
    --adiabatic_dt 0.5 \\
    --rodeo_dt ${RODEO_DT} \\
    --alpha_coeff ${ALPHA_COEFF} \\
    --rodeo_min ${RODEO_MIN} \\
    --rodeo_max ${RODEO_MAX} \\
    --rodeo_steps ${RODEO_STEPS} \\
    --rodeo_spacing ${RODEO_SPACING} \\
    --rodeo_times_in_T0 ${RODEO_TIMES_IN_T0} \\
    --time_cuts ${TIME_CUTS_NO_PRECOND[@]} \\
    --output_dir data/rodeo \\
    --cache_dir data/precond
HEREDOC
}

# ============================================================
# === PRECONDITIONED block ====================================
# ============================================================

# Adiabatic ramp time is computed per-job as TAE = TAE_SLOPE * L + TAE_INTERCEPT.
# Modify the coefficients here to change the formula globally.
TAE_SLOPE=0.84
TAE_INTERCEPT=1.19
ADIABATIC_DT=0.5

# Which cuts to run for the preconditioned jobs.
# You can pass a single cut or a batch; outputs are separate files
# so you can add new cuts later without rerunning anything.
TIME_CUTS_PRECOND=(7 11)

submit_precond() {
    local N_L=${#L_LIST[@]}
    local TOTAL_TASKS=$N_L

    echo "Submitting precond array: ${TOTAL_TASKS} tasks (TAE = ${TAE_SLOPE}*L + ${TAE_INTERCEPT})"

    sbatch --array=0-$((TOTAL_TASKS - 1)) <<HEREDOC
#!/bin/bash
#SBATCH --job-name=rodeo_precond
#SBATCH --time=6-23:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/rodeo_precond-%A_%a.out
#SBATCH --error=slurm_logs/rodeo_precond-%A_%a.err

ARRAY_IDX=\$SLURM_ARRAY_TASK_ID

L_LIST=(${L_LIST[@]})
L=\${L_LIST[\$ARRAY_IDX]}

module purge
module load anaconda
source activate base
unset PYTHONHOME PYTHONPATH

mkdir -p slurm_logs data/rodeo data/precond

# ----------------------------------------------------------------
# TAE = ${TAE_SLOPE} * L + ${TAE_INTERCEPT}  (resolved inside run_rodeo.py)
# The adiabatic ramp is computed ONCE and cached in data/precond/.
# Re-runs with different cuts load the cache and skip the ramp.
# ----------------------------------------------------------------
python src/run_rodeo.py \\
    --L \$L \\
    --filling ${FILLING} \\
    --adiabatic_time_auto \\
    --tae_slope ${TAE_SLOPE} \\
    --tae_intercept ${TAE_INTERCEPT} \\
    --adiabatic_dt ${ADIABATIC_DT} \\
    --rodeo_dt ${RODEO_DT} \\
    --alpha_coeff ${ALPHA_COEFF} \\
    --rodeo_min ${RODEO_MIN} \\
    --rodeo_max ${RODEO_MAX} \\
    --rodeo_steps ${RODEO_STEPS} \\
    --rodeo_spacing ${RODEO_SPACING} \\
    --rodeo_times_in_T0 ${RODEO_TIMES_IN_T0} \\
    --time_cuts ${TIME_CUTS_PRECOND[@]} \\
    --output_dir data/rodeo \\
    --cache_dir data/precond
HEREDOC
}

# ============================================================
# === OPTIONAL: submit only specific cuts for an existing   ===
# === preconditioned state (no ramp re-computation)        ===
# ============================================================
# Example: add cut=13 for L=128 using the TAE formula
#
#   python src/run_rodeo.py \
#       --L 128 --filling 0.5 \
#       --adiabatic_time_auto --tae_slope 0.84 --tae_intercept 1.19 \
#       --adiabatic_dt 0.5 \
#       --rodeo_dt 0.01 --alpha_coeff 2.8 \
#       --rodeo_min 1 --rodeo_max 100 --rodeo_steps 100 \
#       --rodeo_times_in_T0 \
#       --time_cuts 13 \
#       --output_dir data/rodeo --cache_dir data/precond
#
# Or with an explicit adiabatic_time (e.g. if you ran with a custom value):
#
#   python src/run_rodeo.py \
#       --L 128 --filling 0.5 \
#       --adiabatic_time 108.31 --adiabatic_dt 0.5 \
#       --rodeo_dt 0.01 --alpha_coeff 2.8 \
#       --rodeo_min 1 --rodeo_max 100 --rodeo_steps 100 \
#       --rodeo_times_in_T0 \
#       --time_cuts 13 \
#       --output_dir data/rodeo --cache_dir data/precond

# ============================================================
# === DISPATCH  ==============================================
# ============================================================

MODE=${1:-all}

case "$MODE" in
    no_precond)
        submit_no_precond
        ;;
    precond)
        submit_precond
        ;;
    all)
        submit_no_precond
        submit_precond
        ;;
    *)
        echo "Usage: $0 [no_precond | precond | all]"
        exit 1
        ;;
esac