DATASET=$1
ARCH=$2
PRED_NAME=$3
GPU=$4
EVAL=$5
SID=$6
EID=$7
iter=latest
python test.py --gpus "${GPU}" \
--cfg "outputs/phys/${DATASET}/${PRED_NAME}/config.yaml" \
--predictor-arch "${ARCH}" \
--predictor-init "outputs/phys/${DATASET}/${PRED_NAME}/ckpt_${iter}.path.tar" \
--eval ${EVAL} \
--start-id "${SID:=0}" \
--end-id "${EID:=25}"
