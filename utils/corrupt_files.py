import pandas as pd
import torchaudio
import os

def corrupt_audio(csv_path, path_column="path_to_audio"):
    """
    Removes corrupted audio files from a CSV and rewrites the file in-place.

    :param csv_path: Path to the CSV file.
    :param path_column: Name of the column containing audio file paths.
    """
    df = pd.read_csv(csv_path)
    print(f"Checking {len(df)} entries in {csv_path}")

    valid_rows = []
    for i, row in df.iterrows():
        audio_path = row[path_column]
        try:
            torchaudio.load("fma_small/"+audio_path)
            valid_rows.append(row)
        except Exception as e:
            print(f"[CORRUPT] {audio_path}: {e}")

    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_df.to_csv(csv_path, index=False)
    print(f"Cleaned CSV written to {csv_path} ({len(cleaned_df)} valid entries)")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        csv_file = f"csv_files/{split}.csv"
        if os.path.exists(csv_file):
            corrupt_audio(csv_file)
        else:
            print(f"{csv_file} not found.")
