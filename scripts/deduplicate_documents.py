#!/usr/bin/env python3
"""
Deduplicate Documents for OpenWebUI Upload.

Scans a directory for duplicate files (by content hash) and copies
only unique files to an output directory, preserving folder structure.

Usage:
    python scripts/deduplicate_documents.py
    python scripts/deduplicate_documents.py --input src/data/documents/DE --output ./unique_documents
    python scripts/deduplicate_documents.py --dry-run  # Preview without copying
"""

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def hash_file(path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_duplicates(input_dir: Path, extensions: set[str] | None = None) -> dict:
    """
    Find all files and group by content hash.

    Returns:
        dict: {hash: [list of file paths with same content]}
    """
    hash_to_files = defaultdict(list)

    # Find all files
    all_files = []
    for path in input_dir.rglob("*"):
        if path.is_file():
            if extensions is None or path.suffix.lower() in extensions:
                all_files.append(path)

    print(f"Scanning {len(all_files)} files...")

    for i, path in enumerate(all_files):
        try:
            file_hash = hash_file(path)
            hash_to_files[file_hash].append(path)
        except Exception as e:
            print(f"  Warning: Could not hash {path.name}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_files)} files...")

    return hash_to_files


def deduplicate(
    input_dir: Path,
    output_dir: Path,
    extensions: set[str] | None = None,
    dry_run: bool = False,
    preserve_structure: bool = True,
) -> tuple[int, int, int]:
    """
    Copy unique files to output directory.

    Args:
        input_dir: Source directory
        output_dir: Destination for unique files
        extensions: File extensions to include (None = all)
        dry_run: If True, don't copy, just report
        preserve_structure: Keep subdirectory structure

    Returns:
        tuple: (total_files, unique_files, duplicate_files)
    """
    hash_to_files = find_duplicates(input_dir, extensions)

    total_files = sum(len(files) for files in hash_to_files.values())
    unique_files = len(hash_to_files)
    duplicate_files = total_files - unique_files

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total files scanned:    {total_files}")
    print(f"Unique files:           {unique_files}")
    print(f"Duplicate files:        {duplicate_files}")
    print()

    # Report duplicates
    if duplicate_files > 0:
        print("Duplicate groups found:")
        dup_count = 0
        for file_hash, files in hash_to_files.items():
            if len(files) > 1:
                dup_count += 1
                if dup_count <= 10:  # Show first 10 groups
                    print(f"\n  Group {dup_count} ({len(files)} copies):")
                    for f in files[:5]:  # Show first 5 of each group
                        print(f"    - {f.relative_to(input_dir)}")
                    if len(files) > 5:
                        print(f"    ... and {len(files) - 5} more")
        if dup_count > 10:
            print(f"\n  ... and {dup_count - 10} more duplicate groups")
        print()

    if dry_run:
        print("DRY RUN - no files copied")
        return total_files, unique_files, duplicate_files

    # Copy unique files
    print(f"Copying {unique_files} unique files to {output_dir}...")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    copied = 0
    for file_hash, files in hash_to_files.items():
        # Take the first file from each group (arbitrary but consistent)
        src_path = files[0]

        if preserve_structure:
            # Keep relative path structure
            rel_path = src_path.relative_to(input_dir)
            dst_path = output_dir / rel_path
        else:
            # Flat structure - just filename
            dst_path = output_dir / src_path.name
            # Handle name collisions
            if dst_path.exists():
                stem = dst_path.stem
                suffix = dst_path.suffix
                counter = 1
                while dst_path.exists():
                    dst_path = output_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

        # Create parent directories if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dst_path)
        copied += 1

        if copied % 100 == 0:
            print(f"  Copied {copied}/{unique_files} files...")

    print(f"  Done! Copied {copied} files.")

    return total_files, unique_files, duplicate_files


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate files and copy unique ones to output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: deduplicate src/data/documents/DE → ./unique_documents
  python scripts/deduplicate_documents.py

  # Custom directories
  python scripts/deduplicate_documents.py --input ./my_docs --output ./clean_docs

  # Preview without copying
  python scripts/deduplicate_documents.py --dry-run

  # Only markdown files
  python scripts/deduplicate_documents.py --extensions .md

  # Flat output (no subdirectories)
  python scripts/deduplicate_documents.py --flat
"""
    )
    parser.add_argument(
        "--input", "-i",
        default="src/data/documents/DE",
        help="Input directory to scan (default: src/data/documents/DE)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./unique_documents",
        help="Output directory for unique files (default: ./unique_documents)"
    )
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        help="File extensions to include (e.g., .md .pdf). Default: all files"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without copying files"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Don't preserve directory structure (flat output)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if output_dir.exists() and not args.dry_run:
        print(f"Warning: Output directory already exists: {output_dir}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(output_dir)

    extensions = None
    if args.extensions:
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}
        print(f"Filtering for extensions: {extensions}")

    print(f"Input:  {input_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print()

    total, unique, duplicates = deduplicate(
        input_dir=input_dir,
        output_dir=output_dir,
        extensions=extensions,
        dry_run=args.dry_run,
        preserve_structure=not args.flat,
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:      {total}")
    print(f"Unique files:     {unique}")
    print(f"Duplicates:       {duplicates} ({duplicates/total*100:.1f}%)" if total > 0 else "Duplicates: 0")

    if not args.dry_run and unique > 0:
        print()
        print(f"Unique documents saved to: {output_dir.absolute()}")
        print()
        print("Next steps:")
        print("  1. Stop OpenWebUI:  ./scripts/setup_openwebui.sh --stop")
        print("  2. Clear vector DB: rm -rf ~/.openwebui/vector_db/")
        print("  3. Start OpenWebUI: ./scripts/setup_openwebui.sh")
        print(f"  4. Upload files from: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
