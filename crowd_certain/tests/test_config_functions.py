"""
Test script for the fixed find_config_file and revert_to_default_config functions.

This script tests:
1. Finding a config file in various locations
2. Finding a default config file in the fixed location
3. Reverting to the default config
"""

import os
import shutil
from pathlib import Path

from crowd_certain.utilities.settings import find_config_file, revert_to_default_config

def test_find_config_file():
    """Test the find_config_file function."""
    print("\n=== Testing find_config_file ===")

    # Test finding the regular config file
    config_path = find_config_file(debug=True)
    print(f"Regular config path: {config_path}")
    print(f"Exists: {config_path.exists()}")

    # Test finding the default config file
    default_config_path = find_config_file(find_default_config=True, debug=True)
    print(f"Default config path: {default_config_path}")
    print(f"Exists: {default_config_path.exists()}")

    # Test with a specific path
    specific_path = Path.cwd() / "test_specific_dir"
    specific_path.mkdir(exist_ok=True)
    specific_config_path = find_config_file(config_path=specific_path, debug=True)
    print(f"Specific config path: {specific_config_path}")
    print(f"Exists: {specific_config_path.exists()}")

    # Clean up
    if specific_path.exists():
        try:
            specific_path.rmdir()
            print(f"Removed test directory: {specific_path}")
        except:
            print(f"Could not remove test directory: {specific_path}")

    return config_path, default_config_path

def test_revert_to_default_config():
    """Test the revert_to_default_config function."""
    print("\n=== Testing revert_to_default_config ===")

    # Create a test directory
    test_dir = Path.cwd() / "test_config_dir"
    test_dir.mkdir(exist_ok=True)

    # Test reverting to default config
    success, config_path = revert_to_default_config(config_path=test_dir, debug=True)
    print(f"Revert success: {success}")
    print(f"Config path: {config_path}")
    print(f"Exists: {config_path.exists()}")

    # Clean up
    if config_path.exists():
        try:
            os.remove(config_path)
            print(f"Removed test config file: {config_path}")
        except:
            print(f"Could not remove test config file: {config_path}")

    if test_dir.exists():
        try:
            test_dir.rmdir()
            print(f"Removed test directory: {test_dir}")
        except:
            print(f"Could not remove test directory: {test_dir}")

    return success, config_path

def test_with_backup():
    """Test with backing up and restoring the original config."""
    print("\n=== Testing with backup and restore ===")

    # Find the current config file
    config_path = find_config_file(debug=True)

    # Check if it exists
    if not config_path.exists():
        print(f"Config file does not exist: {config_path}")
        return False

    # Create a backup
    backup_path = config_path.with_suffix('.backup')
    try:
        if config_path.exists():
            shutil.copy2(config_path, backup_path)
            print(f"Created backup: {backup_path}")

        # Test reverting to default
        success, new_config_path = revert_to_default_config(config_path=config_path, debug=True)
        print(f"Revert success: {success}")
        print(f"New config path: {new_config_path}")
        print(f"Exists: {new_config_path.exists()}")

        # Restore the backup
        if backup_path.exists():
            shutil.copy2(backup_path, config_path)
            print(f"Restored backup: {config_path}")
            os.remove(backup_path)
            print(f"Removed backup: {backup_path}")

        return success
    except Exception as e:
        print(f"Error in test_with_backup: {str(e)}")

        # Try to restore the backup
        if backup_path.exists():
            try:
                shutil.copy2(backup_path, config_path)
                print(f"Restored backup: {config_path}")
                os.remove(backup_path)
                print(f"Removed backup: {backup_path}")
            except Exception as e2:
                print(f"Error restoring backup: {str(e2)}")

        return False

def main():
    """Main function."""
    print("=== Testing Configuration Functions ===")

    # Test find_config_file
    config_path, default_config_path = test_find_config_file()

    # Test revert_to_default_config
    test_revert_to_default_config()

    # Test with backup and restore
    test_with_backup()

    print("\n=== Configuration Function Tests Complete ===")

if __name__ == "__main__":
    main()
