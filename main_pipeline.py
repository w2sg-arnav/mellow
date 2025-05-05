import argparse
import os
import logging
import sys
import json
from pathlib import Path
import time # For timing inference

# Essential imports for audio and model handling
import torch
import yaml     # Potentially useful for inspecting config, but wrapper handles loading

# --- Configuration ---
# Set level=logging.DEBUG to see the sys.path printout
logging.basicConfig(
    level=logging.INFO, # Change to logging.DEBUG for more verbose output if needed
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Helper Functions ---

# Determine device *before* initializing wrapper
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu" # MellowWrapper expects device index or "cpu"
# Log device info early
logging.info(f"CUDA Available: {USE_CUDA}, Using device: {DEVICE}")


def parse_arguments():
    """Parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Audio Evaluation Pipeline: Compares two audio samples using Mellow."
    )
    parser.add_argument(
        "--sample1", required=True, type=str,
        help="Path to the first audio sample file."
    )
    parser.add_argument(
        "--sample2", required=True, type=str,
        help="Path to the second audio sample file."
    )
    parser.add_argument(
        "--reference", required=False, default=None, type=str,
        help="[OPTIONAL] Path to the reference speaker audio file. If provided, enables speaker similarity check."
    )
    parser.add_argument(
        "--mellow_dir", required=True, type=str,
        help="Path to the root directory of the *cloned* Mellow GitHub repository (needed for imports)."
    )
    parser.add_argument(
        "--mellow_config", default="v0", type=str, choices=["v0"],
        help="Mellow configuration name."
    )
    parser.add_argument(
        "--mellow_model", default="v0", type=str, choices=["v0", "v0_s"],
        help="Mellow model/checkpoint name ('v0' or 'v0_s')."
    )
    parser.add_argument(
        "--output_dir", default="results", type=str,
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--max_new_tokens", default=350, type=int, # Default allows longer answers
        help="Maximum number of new tokens for Mellow to generate."
    )
    parser.add_argument(
        "--top_p", default=0.8, type=float,
        help="Top-p (nucleus) sampling parameter for generation."
    )
    parser.add_argument(
        "--temperature", default=0.7, type=float, # Default slightly lower
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--input_text", default=None, type=str,
        help="[OPTIONAL] The text that was fed to the TTS models to generate the samples. May help Mellow evaluate pronunciation accuracy."
    )
    return parser.parse_args()

def validate_inputs(args):
    """Checks if the provided audio file paths and Mellow directory exist."""
    valid = True
    # Use Path objects for easier path manipulation
    args.sample1 = Path(args.sample1).resolve()
    args.sample2 = Path(args.sample2).resolve()

    for name, path_obj in [("Sample 1", args.sample1), ("Sample 2", args.sample2)]:
        if not path_obj.is_file():
            logging.error(f"Input file not found: {path_obj}")
            valid = False
        else:
            logging.info(f"Found {name} file: {path_obj}")

    # Handle optional reference file
    if args.reference:
        args.reference = Path(args.reference).resolve()
        if not args.reference.is_file():
            logging.warning(f"Optional reference file specified but not found: {args.reference}. Speaker similarity prompt will be skipped.")
            args.reference = None # Ensure it's None if not found
        else:
            logging.info(f"Found Reference Audio file: {args.reference}")
    else:
        logging.info("No reference audio file provided.")

    # Handle optional input text
    if args.input_text:
        logging.info(f"Input text provided (first 100 chars): '{args.input_text[:100]}...'")
    else:
        logging.warning("No input text provided via --input_text. Pronunciation/Prosody evaluation may be less specific.")

    # Validate Mellow directory and structure needed for import
    args.mellow_dir = Path(args.mellow_dir).resolve()
    if not args.mellow_dir.is_dir():
        logging.error(f"Mellow directory not found: {args.mellow_dir}")
        valid = False
    else:
        logging.info(f"Found Mellow directory: {args.mellow_dir}")
        inner_mellow_path = args.mellow_dir / "mellow"
        wrapper_path = inner_mellow_path / "wrapper.py"
        init_path = inner_mellow_path / "__init__.py"

        # Crucial checks for import to work via sys.path
        if not inner_mellow_path.is_dir():
             logging.error(f"CRITICAL: Expected 'mellow' subdirectory NOT found inside {args.mellow_dir}. Cannot import wrapper.")
             valid = False
        elif not init_path.exists():
             # Technically import might still work sometimes without it, but good practice
             logging.warning(f"Missing '__init__.py' in {inner_mellow_path}. Ensure this directory is treated as a package.")
        elif not wrapper_path.exists():
             logging.error(f"CRITICAL: Expected 'wrapper.py' NOT found inside {inner_mellow_path}. Cannot import wrapper.")
             valid = False

        # Check for checkpoints directory (needed by wrapper internally)
        if not (args.mellow_dir / "checkpoints").is_dir():
             logging.warning(f"Expected 'checkpoints' subdirectory not found in {args.mellow_dir}. Check Mellow setup.")

    # Ensure output directory exists
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using output directory: {args.output_dir}")

    return valid

# Attempt to import MellowWrapper *after* potentially modifying sys.path
MellowWrapper = None # Define globally to handle import failure
def initialize_mellow_wrapper(mellow_repo_path, config_name, model_name):
    """Adds Mellow dir to path and attempts to import/initialize MellowWrapper."""
    global MellowWrapper # Allow modification of the global variable

    logging.info(f"Attempting to initialize MellowWrapper (config='{config_name}', model='{model_name}')")
    logging.info(f"Using Mellow code from: {mellow_repo_path}")

    # Convert repo path to string for sys.path
    mellow_repo_dir_str = str(mellow_repo_path)

    # Add the Mellow repo root directory to sys.path
    original_sys_path = list(sys.path)
    if mellow_repo_dir_str not in sys.path:
        sys.path.insert(0, mellow_repo_dir_str)
        logging.info(f"Added '{mellow_repo_dir_str}' to sys.path for import.")

    # Log the current path for debugging
    logging.debug(f"Current sys.path: {sys.path}")

    wrapper = None
    try:
        # Import using the structure relative to the added path: mellow_dir/mellow/wrapper.py
        from mellow.wrapper import MellowWrapper as ImportedWrapper
        MellowWrapper = ImportedWrapper # Assign to global variable if import succeeds
        logging.info("Successfully imported MellowWrapper from mellow.wrapper")

        start_time = time.time()
        wrapper = MellowWrapper(
            config=config_name,
            model=model_name,
            device=DEVICE,
            use_cuda=USE_CUDA,
        )
        load_time = time.time() - start_time
        logging.info(f"MellowWrapper initialized successfully in {load_time:.2f} seconds.")
        if wrapper: # Add load_time attribute if initialization succeeded
            wrapper.load_time = load_time

    except ImportError as e:
         logging.error(f"ImportError: Could not import 'from mellow.wrapper import MellowWrapper' even after adding '{mellow_repo_dir_str}' to sys.path. Check the directory structure ({mellow_repo_dir_str}/mellow/wrapper.py) and __init__.py files. Error: {e}", exc_info=True)
         wrapper = None
         MellowWrapper = None # Ensure global variable is None on import failure
    except FileNotFoundError as e:
         logging.error(f"FileNotFoundError during MellowWrapper initialization (likely missing config/checkpoint). Check Mellow setup in '{mellow_repo_dir_str}'. Error: {e}", exc_info=True)
         wrapper = None
    except Exception as e:
        # Catch potential errors during MellowWrapper.__init__ (like the previous torch._custom_ops error)
        logging.error(f"Failed to initialize MellowWrapper instance: {e}", exc_info=True)
        wrapper = None
    finally:
        # Clean up sys.path modification
        if mellow_repo_dir_str in sys.path and mellow_repo_dir_str not in original_sys_path:
            sys.path.remove(mellow_repo_dir_str)
            logging.info(f"Removed '{mellow_repo_dir_str}' from sys.path.")

    return wrapper

# --- Main Pipeline Logic ---
def main():
    """Main function to orchestrate the pipeline steps."""
    pipeline_start_time = time.time()
    logging.info(f"--- Starting Audio Evaluation Pipeline ---")

    # --- Step 1: Parse Arguments and Validate Inputs ---
    args = parse_arguments()
    logging.info(f"Running with arguments: {vars(args)}")
    # Validation includes checking Mellow directory structure
    if not validate_inputs(args):
         logging.error("Input validation failed. Exiting.")
         sys.exit(1)

    # --- Step 2: Initialize Mellow Wrapper (includes import attempt) ---
    logging.info("--- Step 2: Initializing Mellow Wrapper ---")
    mellow_wrapper = initialize_mellow_wrapper(
        mellow_repo_path=args.mellow_dir, # Pass the Path object
        config_name=args.mellow_config,
        model_name=args.mellow_model
    )

    # Exit if initialization failed (either import or instantiation)
    if mellow_wrapper is None:
        logging.error("Failed to initialize MellowWrapper. Exiting.")
        sys.exit(1)

    # --- Step 3: Audio Handling / Prepare Data ---
    logging.info("--- Step 3: Preparing Audio Data ---")
    # Convert Path objects back to strings for the wrapper if needed
    audio_path_strs = {
        "sample1": str(args.sample1),
        "sample2": str(args.sample2),
        "reference": str(args.reference) if args.reference else None,
    }
    logging.info("Using audio file paths as strings for Mellow.")

    # --- Step 4: Define Evaluation Prompts (SIMPLIFIED - NO INLINE TEXT) ---
    logging.info("--- Step 4: Defining Evaluation Prompts (Simplified - No Inline Text) ---")

    # Prompts focus on direct audio comparison. --input_text is passed separately.
    prompts = {
        "overall_preference": "Listen to Audio Sample 1 and Audio Sample 2. Considering all aspects (naturalness, clarity, pronunciation, tone, prosody), which sample do you prefer overall? Provide a detailed explanation for your preference, citing specific examples or qualities observed directly in the audio.",

        "naturalness_detailed": f"Compare Audio Sample 1 and Audio Sample 2 in terms of **naturalness**. Which sounds more like a real human speaking spontaneously? Describe in detail the specific elements contributing to naturalness (or lack thereof) in each sample, such as speech flow, rhythm, pacing, pauses, and overall vocal tone.",

        "emotion_tone": f"Compare Audio Sample 1 and Audio Sample 2 based on **emotion and tone**. Which sample more effectively conveys a discernible emotion? Describe the perceived emotional tone of each voice (e.g., neutral, happy, sad, etc.). Is the tone consistent? Explain using observations about pitch, energy, and inflection.",

        "pronunciation_clarity_comparison": f"Compare the **pronunciation clarity** of Audio Sample 1 and Audio Sample 2. Which sample pronounces words more distinctly and is easier to understand overall? Identify specific moments or types of sounds where one sample is noticeably clearer than the other.",

        "pronunciation_error_analysis": f"Analyze Audio Sample 1 and Audio Sample 2 specifically for apparent **pronunciation errors or articulation issues**. Identify any instances of obviously mispronounced words, slurred sounds, or unnatural articulation in *either* sample. Which sample sounds like it has more pronunciation problems?",

        "prosody_emphasis_intonation": f"Evaluate the **prosody (rhythm, emphasis, and intonation)** of Audio Sample 1 and Audio Sample 2. Does the rhythm and flow of speech sound natural? Does the intonation (rise and fall of pitch) sound appropriate for conversation? Which sample demonstrates more human-like prosody?",

        "noise_artifacts_detailed": "Analyze Audio Sample 1 and Audio Sample 2 for **background noise and synthesis artifacts**. Describe any audible noise (hiss, hum, clicks, etc.) and its level in each sample. Describe any unnatural sounds from the synthesis process (robotic tone, metallic quality, glitches, etc.). Which sample is cleaner?",
    }

    if args.reference:
        prompts["speaker_similarity_detailed"] = f"Compare the overall voice quality of Audio Sample 1 and Audio Sample 2 to the voice characteristics of the Reference Audio (located at {audio_path_strs['reference']}). Considering aspects like pitch range, timbre, speaking rate, and unique vocal mannerisms, which sample (1 or 2) sounds more similar to the reference speaker? Explain your judgment with detailed observations."
        logging.info("Added 'speaker_similarity_detailed' prompt.")
    else:
        logging.info("Skipping 'speaker_similarity_detailed' prompt.")

    logging.info(f"Defined {len(prompts)} simplified evaluation prompts.")

    # --- Step 5: Prepare Batch and Run Inference ---
    logging.info("--- Step 5: Preparing Batch and Running Inference ---")
    evaluation_results = {}
    examples_batch = []
    inference_time = None
    prompt_keys_ordered = list(prompts.keys())

    for key in prompt_keys_ordered:
        prompt_text = prompts[key]
        # Use string paths for the batch
        examples_batch.append([audio_path_strs["sample1"], audio_path_strs["sample2"], prompt_text])
        logging.info(f"Added prompt '{key}' to batch.")

    if not examples_batch:
        logging.warning("No prompts defined. Skipping inference.")
    else:
        logging.info(f"Running Mellow generate for {len(examples_batch)} prompts...")
        inference_start_time = time.time()
        try:
            # Adjust max_len if needed, ensure it's at least 350 for detailed answers
            effective_max_len = max(args.max_new_tokens, 350)
            logging.info(f"Using max_len={effective_max_len}, top_p={args.top_p}, temperature={args.temperature} for generation.")

            responses = mellow_wrapper.generate(
                examples=examples_batch,
                max_len=effective_max_len,
                top_p=args.top_p,
                temperature=args.temperature
            )
            inference_time = time.time() - inference_start_time
            logging.info(f"Mellow inference completed in {inference_time:.2f} seconds.")

            if len(responses) == len(prompt_keys_ordered):
                for i, key in enumerate(prompt_keys_ordered):
                    logging.debug(f"Full response for '{key}': {responses[i]}")
                    evaluation_results[key] = responses[i]
                    logging.info(f"Response for '{key}': {responses[i][:200]}...")
            else:
                logging.error(f"Mismatch between prompts ({len(prompt_keys_ordered)}) and responses ({len(responses)}).")
                evaluation_results["__raw_response_error"] = responses

        except Exception as e:
            if 'inference_start_time' in locals() and inference_start_time:
                 inference_time = time.time() - inference_start_time
                 logging.error(f"Error during Mellow generation after {inference_time:.2f} seconds: {e}", exc_info=True)
            else:
                 logging.error(f"Error during Mellow generation: {e}", exc_info=True)
            for key in prompt_keys_ordered:
                 evaluation_results[key] = f"Error during Mellow generation: {e}"

    # --- Step 6: Output Parsing & Reporting ---
    logging.info("--- Step 6: Saving Results ---")
    s1_name = args.sample1.stem
    s2_name = args.sample2.stem
    ref_part = f"_ref_{args.reference.stem}" if args.reference else "_no_ref"
    # Use Path object for output dir and filename construction
    output_filename = f"eval_{s1_name}_vs_{s2_name}{ref_part}_{args.mellow_model}.json"
    output_file_path = args.output_dir / output_filename

    report_data = {
        "input_files": {
            # Store paths as strings in the final report
            "sample1": str(args.sample1),
            "sample2": str(args.sample2),
            "reference": str(args.reference) if args.reference else None,
        },
        "input_text_provided": args.input_text, # Record the input text used (or None)
        "mellow_config": {
            "config": args.mellow_config,
            "model": args.mellow_model,
            "max_new_tokens": args.max_new_tokens, # Report user setting
            "top_p": args.top_p,
            "temperature": args.temperature,
        },
        "evaluation_results": evaluation_results,
         "timing_seconds": {
             "total_pipeline": time.time() - pipeline_start_time,
             "mellow_load": mellow_wrapper.load_time if hasattr(mellow_wrapper, 'load_time') else None,
             "mellow_inference": inference_time
         }
    }

    try:
        with open(output_file_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        logging.info(f"Evaluation results saved to: {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file_path}: {e}")

    pipeline_end_time = time.time()
    logging.info(f"--- Audio Evaluation Pipeline finished in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")

# --- Entry Point ---
if __name__ == "__main__":
    main()