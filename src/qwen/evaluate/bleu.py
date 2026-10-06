import argparse

import pandas as pd
import sacrebleu


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tính BLEU từ file CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn đến file CSV.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="prediction",
        help="Tên cột chứa câu dự đoán.",
    )
    parser.add_argument(
        "--reference-column",
        type=str,
        default="tgt",
        help="Tên cột chứa câu tham chiếu.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=2000,
        help="Vị trí bắt đầu tính BLEU.",
    )
    parser.add_argument(
        "--tokenize",
        type=str,
        default="13a",
        help="Phương pháp tokenization của SacreBLEU.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_columns = {
        args.prediction_column,
        args.reference_column,
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"File CSV thiếu các cột: {sorted(missing_columns)}. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )

    evaluation_df = df.iloc[args.start_index:].copy()

    evaluation_df = evaluation_df.dropna(
        subset=[
            args.prediction_column,
            args.reference_column,
        ]
    )

    predictions = (
        evaluation_df[args.prediction_column]
        .astype(str)
        .tolist()
    )
    references = (
        evaluation_df[args.reference_column]
        .astype(str)
        .tolist()
    )

    if not predictions:
        raise ValueError(
            f"Không có dữ liệu để tính BLEU từ index {args.start_index}."
        )

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
        tokenize=args.tokenize,
    )

    print(f"Số câu đánh giá: {len(predictions)}")
    print(bleu)
    print(f"BLEU score: {bleu.score:.2f}")


if __name__ == "__main__":
    main()