#!/bin/bash

#SBATCH --job-name=ts
#SBATCH --account=project_2004994
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1,nvme:500
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=85G
#SBATCH --time=36:00:00

# argparse. it is used by a submitting script (`./scripts/submit_job.sh`) and can be ignored
for i in "$@"; do
  case $i in
    -n=*|--now=*)
      NOW="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

# exit when any command fails
set -e

## The following will assign a master port (picked randomly to avoid collision) and an address for ddp.
# We want names of master and slave nodes. Make sure this node (MASTER_ADDR) comes first
MASTER_ADDR=`/bin/hostname -s`
if (( $SLURM_JOB_NUM_NODES > 1 )); then
    WORKERS=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR`
fi
# Get a random unused port on this host(MASTER_ADDR)
MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=$MASTER_ADDR
echo "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT "WORKERS" $WORKERS

# load conda environment
# source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
# conda activate sparse_sync
export PATH="/projappl/project_2004994/SparseSync/conda_env/bin:$PATH"

## select the dataset
# DATASET="LRS3_face_crop"
DATASET="LRS3_no_face_crop"
# DATASET="VGGSoundSparsePicked"

SCRATCH="/scratch/project_2000936"

if [[ "$DATASET" == "VGGSound" ]] || [[ "$DATASET" == "VGGSoundSparsePicked" ]]; then
    DATASET_TARGET="dataset.vggsound.$DATASET"
    VIDS_PATH="$SCRATCH/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/"
elif [[ "$DATASET" == "LRS3_face_crop" ]]; then
    DATASET_TARGET="dataset.lrs.LRS3"
    VIDS_PATH="$SCRATCH/vladimir/data/lrs3/h264_orig_strict_crop_25fps_224side_16000hz_aac/"
elif [[ "$DATASET" == "LRS3_no_face_crop" ]]; then
    DATASET_TARGET="dataset.lrs.LRS3"
    VIDS_PATH="$SCRATCH/vladimir/data/lrs3/h264_uncropped_25fps_256side_16000hz_aac/"
fi


srun python main.py start_time="$NOW" \
    config="configs/sparse_sync_lrs3_2cls_5sec.yaml" \
    logging.logdir="/scratch/project_2004994/viertoli/logs/sync/sync_model" \
    data.vids_path="$VIDS_PATH" \
    data.dataset.target="$DATASET_TARGET" \
    training.base_batch_size="10"

# srun python main.py \
#     config="./configs/sparse_sync_gh_2cls.yaml" \
#     logging.logdir="/scratch/project_2004994/viertoli/logs/sync/sync_model" \
#     data.vids_path="/scratch/project_2004994/greatesthit/vis-data-256_h264_video_25fps_256side_24000hz_aac_len_5_splitby_random" \
#     data.dataset.params.meta_path="/scratch/project_2004994/greatesthit/vis-data-256_h264_video_25fps_256side_24000hz_aac_len_5_splitby_random/metadata.csv" \
#     data.dataset.target="dataset.greatesthit.GreatestHitDataset" \
#     training.base_batch_size="10"
