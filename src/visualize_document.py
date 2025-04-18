import os
import argparse
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import load_json


def load_processed_document(file_path: str) -> Dict[str, Any]:
    """Load a processed document from JSON."""
    return load_json(file_path)


def visualize_chunks(document: Dict[str, Any]) -> None:
    """Visualize document chunks and sections."""
    # Extract data
    title = document.get("metadata", {}).get("title", "Unknown Document")
    chunks = document.get("chunks", [])

    if not chunks:
        print(f"No chunks found in document: {title}")
        return

    # Count sections and chunks per section
    section_counts = {}
    for chunk in chunks:
        section = chunk.get("section", "Unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    # Plot section distribution
    plt.figure(figsize=(12, 8))

    # Sort sections by count
    sorted_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
    section_names = [s[0][:40] + "..." if len(s[0]) > 40 else s[0] for s in sorted_sections]
    section_counts = [s[1] for s in sorted_sections]

    # Get a list of distinct colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(section_names) > len(colors):
        colors = colors * (len(section_names) // len(colors) + 1)

    # Create bar chart
    plt.bar(section_names, section_counts, color=colors[:len(section_names)])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Sections")
    plt.ylabel("Number of Chunks")
    plt.title(f"Document Structure: {title}")
    plt.tight_layout()

    # Save the figure
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]
    plt.savefig(os.path.join(output_dir, f"{safe_title}_structure.png"))
    plt.close()

    # Print structure summary
    print(f"Document: {title}")
    print(f"Total chunks: {len(chunks)}")
    print("Section breakdown:")
    for section, count in sorted_sections:
        print(f"  - {section}: {count} chunks")


def main():
    parser = argparse.ArgumentParser(description="Visualize processed document structure")
    parser.add_argument("--document", type=str, help="Path to processed document JSON file")
    parser.add_argument("--all", action="store_true", help="Process all documents in processed directory")

    args = parser.parse_args()

    if args.document:
        document = load_processed_document(args.document)
        visualize_chunks(document)
    elif args.all:
        processed_dir = "data/processed"
        if not os.path.exists(processed_dir):
            print(f"Directory not found: {processed_dir}")
            return

        json_files = [f for f in os.listdir(processed_dir) if f.endswith(".json") and f != "processing_summary.json"]

        for json_file in json_files:
            file_path = os.path.join(processed_dir, json_file)
            document = load_processed_document(file_path)
            visualize_chunks(document)
            print("-" * 50)
    else:
        print("Please provide either --document or --all argument.")


if __name__ == "__main__":
    main()
