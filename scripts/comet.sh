ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

python ${ROOT_DIR}/qwen/evaluate/comet.py \
    --input /content/test.km-vi.csv \
    --output /content/test.km-vi.comet.csv