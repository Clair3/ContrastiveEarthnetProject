#!/bin/bash
#SBATCH --job-name=preprocess_samples
#SBATCH --array=0-7392%200          
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/preprocess_%A_%a.log
#SBATCH --error=logs/preprocess_%A_%a.log

source /Net/Groups/BGI/scratch/crobin/miniconda3/bin/activate ExtremesEnv2

# Add project root to PYTHONPATH
export PYTHONPATH=/Net/Groups/BGI/scratch/crobin/PythonProjects/ContrastiveEarthnetProject:$PYTHONPATH

CHUNK_SIZE=5
LIST="sample_paths.txt"
OUTDIR="datasets/S2_evi_5d_update/"
echo "Output directory: $OUTDIR"

START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE - 1 ))

echo "Task $SLURM_ARRAY_TASK_ID processes samples index $START to $END"

# Extract lines START..END from the file
sed -n "$((START+1)),$((END+1))p" "$LIST" | while read SAMPLE; do
    echo "Processing $SAMPLE"
    python process_sample.py \
    --input-path "$SAMPLE" \
    --output-dir "$OUTDIR" \
    --mode single
done
