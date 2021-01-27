DATASET=$1
ARCH=$2
ID=$3
GPU=$4
iter=best
python test.py \
--gpus "${GPU}" \
--cfg "outputs/phys/${DATASET}/${ID}/config.yaml" \
--predictor-arch "${ARCH}" \
--predictor-init "outputs/phys/${DATASET}/${ID}/ckpt_${iter}.path.tar"
