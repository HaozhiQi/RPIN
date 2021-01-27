DATASET=$1
ARCH=$2
PRED_NAME=$3
CLS_NAME=$4
GPU=$5
EVAL=$6
SID=$7
EID=$8
iter=best
python test.py --gpus "${GPU}" \
--cfg "outputs/phys/${DATASET}/${PRED_NAME}/config.yaml" \
--predictor-arch "${ARCH}" \
--predictor-init "outputs/phys/${DATASET}/${PRED_NAME}/ckpt_${iter}.path.tar" \
--cls-init "outputs/cls/${DATASET}/${CLS_NAME}/last.ckpt" \
--eval ${EVAL} \
--start-id "${SID:=0}" \
--end-id "${EID:=25}"
