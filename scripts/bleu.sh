ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

python ${ROOT_DIR}/qwen/evaluate/bleu.py \
    --input /content/test.km-vi.csv \
    --start-index 0 \