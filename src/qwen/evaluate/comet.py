import argparse

import pandas as pd
import torch
from qwen.evaluate.bleu import download_model, load_from_checkpoint
from tqdm import tqdm


def compute_comet_scores(
    df: pd.DataFrame,
    comet_model,
    source_column: str = "src",
    reference_column: str = "tgt",
    prediction_column: str = "prediction",
    batch_size: int = 128,
    use_gpu: bool = True,
) -> pd.DataFrame:
    required_columns = {
        source_column,
        reference_column,
        prediction_column,
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"File CSV thiếu các cột: {sorted(missing_columns)}. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )

    evaluation_df = df.dropna(
        subset=[
            source_column,
            reference_column,
            prediction_column,
        ]
    ).copy()

    comet_data = [
        {
            "src": str(row[source_column]),
            "mt": str(row[prediction_column]),
            "ref": str(row[reference_column]),
        }
        for _, row in evaluation_df.iterrows()
    ]

    all_comet_scores = []

    for start_idx in tqdm(
        range(0, len(comet_data), batch_size),
        desc="Computing COMET",
    ):
        batch = comet_data[start_idx : start_idx + batch_size]

        model_output = comet_model.predict(
            batch,
            batch_size=batch_size,
            gpus=1 if use_gpu else 0,
        )

        all_comet_scores.extend(model_output.scores)

    if len(all_comet_scores) != len(evaluation_df):
        raise ValueError(
            "Số lượng COMET score không khớp với số dòng dữ liệu."
        )

    evaluation_df["comet_score"] = all_comet_scores

    average_score = sum(all_comet_scores) / len(all_comet_scores)

    print(f"Số câu đánh giá: {len(evaluation_df)}")
    print(f"COMET score trung bình: {average_score:.4f}")

    return evaluation_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tính COMET score cho các bản dịch trong file CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn đến file CSV đầu vào.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comet_scores.csv",
        help="Đường dẫn lưu file CSV kết quả.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Unbabel/wmt22-comet-da",
        help="Tên mô hình COMET.",
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default="src",
        help="Tên cột chứa câu nguồn.",
    )
    parser.add_argument(
        "--reference-column",
        type=str,
        default="tgt",
        help="Tên cột chứa câu tham chiếu.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="prediction",
        help="Tên cột chứa bản dịch dự đoán.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Kích thước batch.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Vị trí bắt đầu đánh giá.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Chạy bằng CPU thay vì GPU.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.iloc[args.start_index:].copy()

    use_gpu = torch.cuda.is_available() and not args.cpu

    print(f"Thiết bị: {'GPU' if use_gpu else 'CPU'}")
    print(f"Đang tải mô hình: {args.model}")

    comet_model_path = download_model(args.model)
    comet_model = load_from_checkpoint(comet_model_path)

    result_df = compute_comet_scores(
        df=df,
        comet_model=comet_model,
        source_column=args.source_column,
        reference_column=args.reference_column,
        prediction_column=args.prediction_column,
        batch_size=args.batch_size,
        use_gpu=use_gpu,
    )

    result_df.to_csv(args.output, index=False)

    print(f"Đã lưu kết quả tại: {args.output}")


if __name__ == "__main__":
    main()