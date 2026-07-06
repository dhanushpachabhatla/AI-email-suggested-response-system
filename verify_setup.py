#!/usr/bin/env python
"""
Verification script for project setup.
Validates that all components are installed and configured correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.utils import setup_logging, get_logger


def main():
    """Run setup verification checks."""
    print("=" * 60)
    print("AI Email Response System - Setup Verification")
    print("=" * 60)
    print()
    
    # 1. Check Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # 2. Load configuration
    try:
        config = load_config()
        print(f"✓ Configuration loaded from: {config.config_path}")
        print(f"  - Primary LLM: {config.get('llm.primary')}")
        print(f"  - LM Studio URL: {config.get('llm.lm_studio_url')}")
        print(f"  - Embedding model: {config.get('embeddings.model')}")
        print(f"  - Dataset path: {config.get('dataset.path')}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    # 3. Test logging setup
    try:
        logger = setup_logging(
            name="setup_verification",
            level=config.get("logging.level", "INFO"),
            log_format=config.get("logging.format", "json"),
            log_dir=config.get("logging.log_dir", "logs"),
            log_file="verification.log",
            console_output=False,
        )
        logger.info("Setup verification successful", extra={"status": "ok"})
        print(f"✓ Logging configured")
        print(f"  - Level: {config.get('logging.level')}")
        print(f"  - Format: {config.get('logging.format')}")
        print(f"  - Log directory: {config.get('logging.log_dir')}")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
        return False
    
    # 4. Check required directories
    required_dirs = ["src", "data", "tests", "logs", "results", "venv"]
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_name}/")
        else:
            missing_dirs.append(dir_name)
            print(f"✗ Directory missing: {dir_name}/")
    
    # 5. Check required files
    required_files = [
        "requirements.txt",
        "config.yaml",
        ".env.example",
        ".gitignore",
        "src/__init__.py",
        "src/config.py",
        "src/utils.py",
        "tests/__init__.py",
        "tests/test_config.py",
        "tests/test_utils.py",
    ]
    missing_files = []
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✓ File exists: {file_name}")
        else:
            missing_files.append(file_name)
            print(f"✗ File missing: {file_name}")
    
    # 6. Verify key dependencies
    print()
    print("Checking key dependencies...")
    dependencies = [
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("transformers", "Transformers"),
        ("requests", "Requests"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("pytest", "Pytest"),
    ]
    
    missing_deps = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            missing_deps.append(name)
            print(f"✗ {name} NOT installed")
    
    # Summary
    print()
    print("=" * 60)
    if not missing_dirs and not missing_files and not missing_deps:
        print("✓ ALL CHECKS PASSED - Setup is complete!")
        print()
        print("Next steps:")
        print("  1. Start LM Studio and load a model (e.g., Mistral)")
        print("  2. Ensure LM Studio server is running at http://127.0.0.1:1234")
        print("  3. Run the dataset generation to create email pairs")
        print("  4. Execute the main pipeline")
        return True
    else:
        print("✗ SETUP INCOMPLETE - Some components are missing")
        if missing_dirs:
            print(f"  Missing directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"  Missing files: {', '.join(missing_files)}")
        if missing_deps:
            print(f"  Missing dependencies: {', '.join(missing_deps)}")
        return False
    print("=" * 60)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
