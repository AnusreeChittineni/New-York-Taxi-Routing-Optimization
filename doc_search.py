import os
import sys
import re
import argparse

DOC_FILES = [
    "README.md",
    "gnn/gnn.md",
    "QUICK_REFERENCE.md"
]

def load_docs():
    docs = {}
    for file_path in DOC_FILES:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                docs[file_path] = f.readlines()
    return docs

def search_docs(query, context_lines=2):
    docs = load_docs()
    results = []
    
    print(f"Searching for '{query}' in documentation...\n")
    
    for filename, lines in docs.items():
        for i, line in enumerate(lines):
            if query.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "".join(lines[start:end])
                results.append((filename, i + 1, snippet))
    
    if not results:
        print("No matches found.")
        return

    for filename, line_num, snippet in results:
        print(f"📄 \033[1m{filename}\033[0m (Line {line_num})")
        print("-" * 40)
        print(snippet.strip())
        print("-" * 40)
        print()

def main():
    parser = argparse.ArgumentParser(description="Search project documentation.")
    parser.add_argument("query", nargs='?', help="Search query string")
    parser.add_argument("--list", action="store_true", help="List all doc files")
    
    args = parser.parse_args()
    
    if args.list:
        print("Documentation Files:")
        for f in DOC_FILES:
            print(f"- {f}")
        return

    if not args.query:
        print("Usage: python doc_search.py <query>")
        print("Example: python doc_search.py 'training loop'")
        return

    search_docs(args.query)

if __name__ == "__main__":
    main()
