#!/usr/bin/env python3
"""
Train All Patches Script
========================

Automates the creation of the entire Expert Matrix.

Process:
1. Trains the Temporal Update Patch (2019+)
2. Analyzes the SituatedQA dataset to find the most frequent non-US countries
3. Automatically trains a Geo Patch for each of these top countries

Usage:
    python train_all_patches.py --max_geo_patches 5

Author: Leon Wagner
"""

import argparse
import subprocess
import sys
import logging
from collections import Counter
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_all")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import SituatedQALoader, SituatedQAConfig, is_us_location

def get_top_countries(limit: int = 10) -> list[tuple[str, int]]:
    """Scan dataset to find top non-US countries."""
    logger.info("Scanning dataset for frequent countries...")
    
    config = SituatedQAConfig(streaming=True)
    loader = SituatedQALoader(config)
    geo_stream = loader._load_geo_split()
    
    # Counter for locations
    loc_counts = Counter()
    
    # We need to scan a reasonable amount of data to find the distribution
    # Since it's streaming, we'll scan the first N examples (e.g. 5000)
    # or until exhaustion.
    SCAN_LIMIT = 5000
    
    for i, example in enumerate(geo_stream):
        if i >= SCAN_LIMIT:
            break
            
        loc = example.get("location")
        if not loc:
            continue
            
        # Normalize
        loc_clean = loc.strip()
        
        # Skip US locations
        if is_us_location(loc_clean):
            continue
            
        # Add to counter (store original string for display, lower for counting logic if needed)
        # Here we trust the dataset casing mostly, but could normalize to title case
        loc_counts[loc_clean.lower()] += 1

    # Get most common
    # We map back to a nice Title Case for the country argument usually, 
    # but the dataset has mixed casing. Let's pick the string as is 
    # but we might want to manually map "india" -> "India" for cleaner filenames.
    
    top_locs = loc_counts.most_common(limit)
    
    # Basic casing cleanup
    cleaned_top = []
    for loc, count in top_locs:
        # Simple title casing: "india" -> "India", "united kingdom" -> "United Kingdom"
        clean_name = loc.title() 
        cleaned_top.append((clean_name, count))
        
    return cleaned_top

def run_command(cmd: list[str], description: str, output_dir: str):
    """Run a command in a subprocess, skipping if checkpoint exists."""
    # Check if adapter already exists
    adapter_file = Path(output_dir) / "adapter_model.safetensors"
    if adapter_file.exists():
        logger.info(f"⏩ Skipping {description} (checkpoint exists: {adapter_file})")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING: {description}")
    logger.info(f"COMMAND: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"✓ Finished: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {description} (Exit Code: {e.returncode})")
        
def calculate_dynamic_steps(n_examples: int, target_epochs: int = 10, batch_size: int = 16, min_steps: int = 50, max_steps: int = 1000) -> int:
    """
    Calculate optimal training steps based on dataset size.
    
    Formula: steps = (n_examples * epochs) / effective_batch_size
    """
    raw_steps = (n_examples * target_epochs) // batch_size
    # Clamp between min and max
    steps = max(min_steps, min(raw_steps, max_steps))
    return steps

def main():
    parser = argparse.ArgumentParser(description="Train all available expert patches")
    parser.add_argument("--max_geo_patches", type=int, default=10, help="Maximum number of top countries to consider")
    parser.add_argument("--min_examples", type=int, default=50, help="Minimum examples required to train a patch")
    parser.add_argument("--skip_temporal", action="store_true", help="Skip training the temporal patch")
    parser.add_argument("--force_retrain", action="store_true", help="Overwrite existing checkpoints")
    
    # Batch settings for step calculation (must match train_patch.py defaults: bs=4, grad_acc=4 -> eff=16)
    parser.add_argument("--effective_batch_size", type=int, default=16, help="Effective batch size (bs * grad_acc)")
    parser.add_argument("--epochs", type=int, default=10, help="Target epochs per patch")
    
    args = parser.parse_args()

    # 1. Train Temporal Patch
    if not args.skip_temporal:
        output_dir = "checkpoints/patch_temp_2019_plus"
        check_dir = output_dir if not args.force_retrain else "NON_EXISTENT_DIR"
        
        # Temporal is usually large, so we stick to a safe default or could verify count too
        # For now, let's keep a solid 500-1000 range for the main temporal update
        temporal_steps = 1000 
        
        cmd = [
            sys.executable, "train_patch.py",
            "--type", "temporal",
            "--cutoff_year", "2019",
            "--max_steps", str(temporal_steps),
            "--output_dir", output_dir
        ]
        run_command(cmd, f"Temporal Patch (2019+, fixed {temporal_steps} steps)", check_dir)
    
    # 2. Identify Top Countries
    top_countries = get_top_countries(limit=args.max_geo_patches)
    
    logger.info(f"\nAnalyzing top {len(top_countries)} non-US locations:")
    
    # 3. Train Geo Patches (Dynamic)
    trained_countries = []
    
    for country, count in top_countries:
        if count < args.min_examples:
            logger.warning(f"⚠️ Skipping {country}: Only {count} examples (threshold: {args.min_examples})")
            continue
            
        # Calculate dynamic steps
        steps = calculate_dynamic_steps(
            count, 
            target_epochs=args.epochs, 
            batch_size=args.effective_batch_size
        )
        
        # Construct specific output dir
        safe_name = country.lower().replace(" ", "_")
        output_dir = f"checkpoints/patch_geo_{safe_name}"
        check_dir = output_dir if not args.force_retrain else "NON_EXISTENT_DIR"
        
        cmd = [
            sys.executable, "train_patch.py",
            "--type", "geo",
            "--country", country,
            "--max_steps", str(steps),
            "--output_dir", output_dir
        ]
        
        description = f"Geo Patch: {country} (n={count} -> {steps} steps @ {args.epochs} epochs)"
        run_command(cmd, description, check_dir)
        trained_countries.append(country)

    # 4. Train "Rest of World" Patch (Generic)
    # This catches all countries that were skipped or not in top N
    # We pass the list of trained countries to exclude them
    if trained_countries:
        exclude_list = ",".join(trained_countries)
        logger.info(f"\nTraining 'Rest of World' patch (excluding {len(trained_countries)} trained countries)...")
        
        # Use a reasonable default for generic patch (e.g. 500 steps)
        # or we could calculate dynamically if we knew the count, but that's expensive to scan again
        generic_steps = 500
        output_dir = "checkpoints/patch_geo_others"
        check_dir = output_dir if not args.force_retrain else "NON_EXISTENT_DIR"
        
        cmd = [
            sys.executable, "train_patch.py",
            "--type", "geo_generic",
            "--exclude_countries", exclude_list,
            "--max_steps", str(generic_steps),
            "--output_dir", output_dir
        ]
        run_command(cmd, "Generic Geo Patch (Rest of World)", check_dir)

    logger.info("\n" + "="*60)
    logger.info("MATRIX TRAINING COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()

