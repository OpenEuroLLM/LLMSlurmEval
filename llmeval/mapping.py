import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass
class Config:
    arguments_block: dict
    log_file: Path
    num_layers: int | None = None
    hidden_size: int | None = None
    ffn_hidden_size: int | None = None
    num_attention_heads: int | None = None
    seq_length: int | None = None
    max_position_embeddings: int | None = None
    tied_embedding: bool | None = None
    global_batch_size: int | None = None
    train_iters: int | None = None
    lr_decay_style: str | None = None
    lr_warmup_iters: int | None = None
    lr_decay_iters: int | None = None
    lr: float | None = None
    min_lr: float | None = None
    log_file_name: str | None = None
    model_size: str | None = None
    original_log_dir_name: str | None = None
    dataset: str | None = None
    tokenizer: str | None = None

    def __post_init__(self):
        # Parse arguments block into a dictionary of parameters
        self.log_file_name: str = self.log_file.name
        self.model_size: str = self._extract_model_size()
        self.num_layers: int = self._get_int("num_layers")
        self.hidden_size: int = self._get_int("hidden_size")
        self.ffn_hidden_size: int = self._get_ffn_hidden_size()
        self.num_attention_heads: int = self._get_int("num_attention_heads")
        self.seq_length: int = self._get_int("seq_length")
        self.max_position_embeddings: int = self._get_int("max_position_embeddings")
        self.tied_embedding: bool = not self.arguments_block.get(
            "untie_embeddings_and_output_weights", False
        )
        self.global_batch_size: int = self._get_int("global_batch_size")
        self.train_iters: int = self._get_int("train_iters")
        self.lr_decay_style: str = self.arguments_block["lr_decay_style"]
        self.lr_warmup_iters: int = self._get_int("lr_warmup_iters")
        self.lr_decay_iters: int = self._get_int("lr_decay_iters")
        self.lr: float = self._get_float("lr")
        self.min_lr: float = self._get_float("min_lr")
        self.original_log_dir_name: str | None = self._get_original_log_dir_name()
        self.dataset: str = self._get_dataset()
        self.tokenizer: str | None = self._get_tokenizer()

    def _get_int(self, key: str, default: int | None = None) -> int:
        value = self.arguments_block.get(key, default)
        if value is None:
            raise ValueError(f"Missing expected integer value for key: {key}")
        return int(value)

    def _get_float(self, key: str, default: float | None = None) -> float:
        value = self.arguments_block.get(key, default)
        if value is None:
            raise ValueError(f"Missing expected float value for key: {key}")
        return float(value)

    def _get_ffn_hidden_size(self) -> int:
        ffn_hidden_size = self.arguments_block.get("ffn_hidden_size")
        if ffn_hidden_size is not None:
            return int(ffn_hidden_size)
        if "1.7b" in str(self.log_file):
            return 8192
        if "1.3b" in str(self.log_file):
            return 5440
        return 0

    def _get_original_log_dir_name(self) -> str | None:
        save_path = self.arguments_block.get("save")
        return save_path.split("/")[-1] if save_path else None

    def _get_dataset(self) -> str:
        data_path_list = self.arguments_block.get("data_path", [])
        if not data_path_list:
            return "unknown"
        data_path = data_path_list[0].lower()
        if "dclm" in data_path:
            return "DCLM"
        if "c4" in data_path:
            return "C4"
        if "pile" in data_path:
            return "Pile"
        if "commoncorpus" in data_path or "common_corpus" in data_path:
            return "CommonCorpus"
        if "fineweb-edu-1.4t" in data_path:
            return "FineWeb-Edu-1.4T"
        if "slimpajama" in data_path:
            return "SlimPajama"
        if "hplt-2.0" in data_path:
            return "HPLT-2.0"
        if "nemotron-cc-2024-hq-real-synth-mix" in data_path:
            return "Nemotron-cc-2024-HQ-real-synth-mix"
        return "unknown"

    def _get_tokenizer(self) -> str | None:
        tokenizer_path = self.arguments_block.get("tokenizer_model", "")
        if "gpt-neox-20b" in tokenizer_path:
            return "gpt-neox-20b"
        return None

    def _extract_model_size(self) -> str:
        num_layers = self.arguments_block.get("num_layers", 0)
        hidden_size = self.arguments_block.get("hidden_size", 0)
        ffn_hidden_size = self.arguments_block.get("ffn_hidden_size")
        num_attention_heads = self.arguments_block.get("num_attention_heads", 0)

        # Convert to int if they're strings
        try:
            num_layers = int(num_layers)
            hidden_size = int(hidden_size)
            if ffn_hidden_size is not None:
                ffn_hidden_size = int(ffn_hidden_size)
            num_attention_heads = int(num_attention_heads)
        except (ValueError, TypeError):
            pass

        # Check for known architecture patterns
        match (num_layers, hidden_size, ffn_hidden_size, num_attention_heads):
            case (24, 2048, 8192, 32):
                return "1.7b"
            case (24, 2048, 5440, 32):
                return "1.3b"
            case (22, 1024, 3840, 16):
                return "0.4b"
            case (22, 512, 2256, 8):
                return "0.13b"
            case _ if (
                hidden_size <= 640
                and ffn_hidden_size is not None
                and ffn_hidden_size <= 2560
                and num_attention_heads <= 10
            ):
                return "0.1b"

        # Fallback: return "unknown" instead of assuming 1.3
        logging.debug(
            f"Warning: Could not determine model size for {self.log_file.name}"
        )
        return "unknown"

    def __repr__(self) -> str:
        tokens = self.seq_length * self.global_batch_size * self.train_iters
        tokens_billions = round(tokens / 1_000_000_000)
        tokens_str = f"{tokens_billions}B"

        return (
            f"open-sci-ref_model-"
            f"{self.model_size}"
            f"_data-{self.dataset}"
            f"_tokenizer-{self.tokenizer}"
            f"_samples-{tokens_str}"
            f"_global_bs-{self.global_batch_size}"
            f"_context-{self.seq_length}"
            f"_schedule-{self.lr_decay_style}"
            f"_lr-{self.lr:.0e}"
            f"_warmup-{self.lr_warmup_iters}"
            f"_tied_embedding-{self.tied_embedding}"
        )


def parse_arguments_block(content: str) -> dict | None:
    """Parse the arguments block from log file content."""
    lines = content.split("\n")
    arguments = {}
    in_arguments_block = False

    for line in lines:
        # Check for start of arguments block
        if "------------------------ arguments ------------------------" in line:
            in_arguments_block = True
            continue

        # Check for end of arguments block
        if "-------------------- end of arguments ---------------------" in line:
            break

        # Parse arguments within the block
        if in_arguments_block and ":" in line:
            # Remove the node prefix like "[lrdn0598:0]:" and extract argument
            if "]:" in line:
                line = line.split("]:", 1)[1]

            # Split on dots to separate argument name and value
            if "..." in line:
                parts = line.split("...")
                if len(parts) >= 2:
                    arg_name = parts[0].strip()
                    # Get everything after the last set of dots, but clean up any remaining dots
                    arg_value = parts[-1].strip()

                    # Clean up any leading dots that might remain
                    while arg_value.startswith("."):
                        arg_value = arg_value[1:].strip()

                    # Handle special cases for list values
                    if (
                        arg_name == "data_path"
                        and arg_value.startswith("[")
                        and arg_value.endswith("]")
                    ):
                        # Parse list format like ['path1', 'path2']
                        try:
                            arg_value = eval(arg_value)  # Safe for simple list format
                        except Exception:
                            # Fallback to string if eval fails
                            pass

                    # Convert boolean values first (before numeric conversion)
                    if isinstance(arg_value, str):
                        match arg_value.lower():
                            case "true":
                                arg_value = True
                            case "false":
                                arg_value = False
                            case "none":
                                arg_value = None
                            case _ if (
                                arg_value.replace(".", "")
                                .replace("-", "")
                                .replace("e", "")
                                .isdigit()
                            ):
                                try:
                                    if "." in arg_value or "e" in arg_value.lower():
                                        arg_value = float(arg_value)
                                    else:
                                        arg_value = int(arg_value)
                                except ValueError:
                                    pass  # Keep as string if conversion fails

                    arguments[arg_name] = arg_value

    return arguments if arguments else None


def process_log_file(log_file: Path, processed_files: set) -> dict | None:
    """Processes a single log file and returns a configuration dictionary."""
    if str(log_file) in processed_files:
        return None
    processed_files.add(str(log_file))

    with open(log_file, "r") as f:
        content = f.read()

    if "[after training is done]" not in content:
        return None

    arguments_block = parse_arguments_block(content)
    if not arguments_block:
        return None

    try:
        cfg = Config(arguments_block=arguments_block, log_file=log_file)
    except (ValueError, KeyError) as e:
        logging.debug(f"Error processing {log_file}: {e}")
        return None

    if f"validation loss at iteration {cfg.train_iters}" not in content:
        logging.debug(f"Training did not finish; skipping job: {log_file.name}")
        return None

    return {
        "log_file_name": cfg.log_file_name,
        "extracted_model_name": repr(cfg),
        "config": {
            "num_layers": cfg.num_layers,
            "hidden_size": cfg.hidden_size,
            "ffn_hidden_size": cfg.ffn_hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "seq_length": cfg.seq_length,
            "max_position_embeddings": cfg.max_position_embeddings,
            "global_batch_size": cfg.global_batch_size,
            "train_iters": cfg.train_iters,
            "lr_decay_style": cfg.lr_decay_style,
            "lr_warmup_iters": cfg.lr_warmup_iters,
            "lr_decay_iters": cfg.lr_decay_iters,
            "lr": cfg.lr,
            "min_lr": cfg.min_lr,
            "tied_embedding": cfg.tied_embedding,
            "model_size": cfg.model_size,
            "original_log_dir_name": cfg.original_log_dir_name,
            "dataset": cfg.dataset,
            "tokenizer": cfg.tokenizer,
        },
    }


def main():
    """Main function to parse logs and create mapping."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        default="/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    log_files = list(log_path.glob("open-sci-ref*.out"))

    mapping = {}
    processed_files = set()
    with tqdm(total=len(log_files), desc="Processing log files") as pbar:
        for log_file in log_files:
            result = process_log_file(log_file, processed_files)
            if result:
                file_key = log_file.name
                if file_key not in mapping:
                    mapping[file_key] = result
            pbar.update(1)

    # Save the mapping to a JSON file
    logging.info(f"Saving mapping with {len(mapping)} checkpoints")
    jsonl_mapping = list(mapping.values())
    with open("log_dir_name_mapping.jsonl", "w") as f:
        for item in jsonl_mapping:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
