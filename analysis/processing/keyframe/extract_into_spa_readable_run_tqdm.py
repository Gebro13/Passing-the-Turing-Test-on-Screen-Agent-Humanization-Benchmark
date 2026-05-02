from argparse import ArgumentParser
from pathlib import Path
import subprocess
import time
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser(description="Monitor the progress of frame extraction by counting the number of session folders created in the output directory.")
    parser.add_argument("--inspected_folder", type=str, required=True, help="The folder to be inspected for the number of session folders created during frame extraction. This should be the output folder where the extracted frames are being saved.")
    parser.add_argument("--target_count", type=int, required=True, help="The target number of session folders to be created. The progress will be monitored until this number is reached.")

    args = parser.parse_args()
    target: int = args.target_count
    current: int = 0
    inspected_folder: Path = Path(args.inspected_folder)

    with tqdm(total=target, desc="Progress", unit="value", smoothing=1.0) as pbar:
        while current < target:

            command = f"ls -d {inspected_folder}/*/ 2>/dev/null | wc -l"
            list_command = ["bash", "-c", command]

            result = subprocess.run(list_command, capture_output=True, text=True)
            try:
                current = int(result.stdout.strip())
            except ValueError:
                continue
            pbar.n = current
            pbar.refresh()
            time.sleep(0.1)