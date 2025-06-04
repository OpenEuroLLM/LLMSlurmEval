import json
import os
import shutil
import re
import logging
import argparse


def main(output_base_dir: str, base_evals_dir: str):
    """
    Reorganizes evaluation results from a flat directory structure into a hierarchical
    structure organized by model, task, and shot configuration.

    This function traverses evaluation result directories and reorganizes results_*.json files
    into a clean hierarchy: model_name/task_name/n_shot/results.json

    Args:
        output_base_dir (str): Target directory for the reorganized results
        base_evals_dir (str): Source directory containing evaluation run directories

    Logic Overview:
        1. Scans base_evals_dir for run directories (ignoring 'model-symlink')
        2. Within each run directory, looks for model result subdirectories
        3. For each results_*.json file found:
           - Extracts task name from results data (e.g., 'copa', 'hellaswag')
           - Determines shot configuration from n-shot data (e.g., '0-shot', '5-shot')
           - Resolves actual model name by:
             * Following pretrained model symlinks from metadata
             * Reading config.json to get _name_or_path
             * Parsing local model paths with pattern matching
             * Falling back to symlink path if config unavailable
           - Creates organized directory structure:
             * For local models: model_name/hf/iter_X/task/n-shot/results.json
             * For other models: sanitized_model_name/task/n-shot/results.json
           - Copies the results file to the new organized location

    Example transformation:
        Input:  evals/run_123/model_dir/results_copa.json
        Output: evals/results/model-1.7b/copa/0-shot/results.json

    The function handles symlink resolution, model name extraction from configs,
    and creates a consistent directory structure regardless of the original
    evaluation run organization.
    """
    os.makedirs(output_base_dir, exist_ok=True)

    potential_run_dirs = [
        d
        for d in os.listdir(base_evals_dir)
        if os.path.isdir(os.path.join(base_evals_dir, d)) and d != "model-symlink"
    ]
    for run_dir_name in potential_run_dirs:
        current_run_path = os.path.join(base_evals_dir, run_dir_name)

        model_result_dirs = [
            d
            for d in os.listdir(current_run_path)
            if os.path.isdir(os.path.join(current_run_path, d))
        ]
        for model_dir_name in model_result_dirs:
            model_dir_path = os.path.join(current_run_path, model_dir_name)

            any_results_processed_in_dir = False
            for current_results_filename in os.listdir(model_dir_path):
                if not (
                    current_results_filename.startswith("results_")
                    and current_results_filename.endswith(".json")
                ):
                    continue

                any_results_processed_in_dir = True
                results_json_path = os.path.join(
                    model_dir_path, current_results_filename
                )

                try:
                    with open(results_json_path, "r") as f:
                        results_data = json.load(f)

                    if not results_data.get("results"):
                        continue
                    if not results_data["results"]:
                        continue

                    task_name = list(results_data["results"].keys())[0]

                    n_shot_info_str = "unknown_shot"  # Default directory name component
                    n_shot_data = results_data.get("n-shot")
                    if n_shot_data is not None:
                        num_shots = n_shot_data.get(task_name)
                        if (
                            num_shots is not None
                        ):  # num_shots can be 0, which is a valid value
                            n_shot_info_str = f"{num_shots}-shot"

                    configs_data = results_data.get("configs")
                    if not configs_data:
                        continue

                    # Try to find a config key that starts with the task_name
                    task_config_data = None
                    for k, v in configs_data.items():
                        if k.startswith(task_name):
                            task_config_data = v
                            break
                    if not task_config_data:
                        continue

                    metadata = task_config_data.get("metadata")
                    if not metadata:
                        continue

                    pretrained_model_symlink_path = metadata.get("pretrained")
                    if not pretrained_model_symlink_path:
                        continue

                    name_or_path_to_use = None
                    potential_config_path = None

                    path_candidate_for_config = os.path.join(
                        str(pretrained_model_symlink_path), "config.json"
                    )

                    if os.path.exists(path_candidate_for_config) and os.path.isfile(
                        path_candidate_for_config
                    ):
                        potential_config_path = path_candidate_for_config
                    elif os.path.islink(str(pretrained_model_symlink_path)):
                        try:
                            resolved_symlink_target = os.path.realpath(
                                str(pretrained_model_symlink_path)
                            )

                            if os.path.isdir(resolved_symlink_target):
                                path_in_resolved_dir = os.path.join(
                                    resolved_symlink_target, "config.json"
                                )
                                if os.path.exists(
                                    path_in_resolved_dir
                                ) and os.path.isfile(path_in_resolved_dir):
                                    potential_config_path = path_in_resolved_dir
                            elif (
                                os.path.isfile(resolved_symlink_target)
                                and os.path.basename(resolved_symlink_target)
                                == "config.json"
                            ):
                                potential_config_path = resolved_symlink_target
                        except OSError as e:
                            logging.debug(
                                f"Warning: Error resolving symlink {pretrained_model_symlink_path}: {e}"
                            )

                    if potential_config_path:
                        try:
                            with open(potential_config_path, "r") as f:
                                model_config_data = json.load(f)
                            name_or_path_to_use = model_config_data.get("_name_or_path")
                            if not name_or_path_to_use:
                                name_or_path_to_use = str(pretrained_model_symlink_path)
                        except Exception:
                            name_or_path_to_use = str(pretrained_model_symlink_path)
                    else:
                        name_or_path_to_use = str(pretrained_model_symlink_path)

                    if not name_or_path_to_use:
                        continue

                    local_path_match = re.match(
                        r".*/(.*?)/hf/(iter_\d+)$", name_or_path_to_use
                    )

                    if local_path_match:
                        model_str_part = local_path_match.group(1)
                        model_name_segment = model_str_part.split("_machine")[0]
                        iter_name = local_path_match.group(2)
                        new_dir_path = os.path.join(
                            output_base_dir,
                            model_name_segment,
                            "hf",
                            iter_name,
                            task_name,
                            n_shot_info_str,
                        )
                    else:
                        sanitized_model_name = name_or_path_to_use.replace(
                            "/", "_"
                        ).replace("\\", "_")
                        new_dir_path = os.path.join(
                            output_base_dir,
                            sanitized_model_name,
                            task_name,
                            n_shot_info_str,
                        )

                    os.makedirs(new_dir_path, exist_ok=True)

                    destination_file_path = os.path.join(new_dir_path, "results.json")
                    shutil.copy2(results_json_path, destination_file_path)

                except Exception as e:
                    logging.error(f"Error processing {results_json_path}: {e}")

            if not any_results_processed_in_dir:
                logging.info(f"No results_*.json files found in {model_dir_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Reorganize evaluation results")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/results",
        help="Base directory for organized results",
    )
    parser.add_argument(
        "--evals-dir",
        type=str,
        default="evals",
        help="Base directory for evaluation results",
    )
    args = parser.parse_args()

    main(args.output_dir, args.evals_dir)
