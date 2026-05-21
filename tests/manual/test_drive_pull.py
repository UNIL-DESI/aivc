import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path("src").resolve()))

from aivc.sync.drive import NativeDriveSyncManager

def main():
    storage_root = Path.home() / ".aivc" / "storage"
    print(f"Initializing NativeDriveSyncManager with root: {storage_root}")
    
    manager = NativeDriveSyncManager(storage_root)
    if not manager.enabled:
        print("Sync is disabled in configuration.")
        return
        
    print(f"Local machine ID: {manager.machine_id}")
    
    print("\nListing remote machines seen on Google Drive:")
    try:
        machines = manager.list_remote_machines()
        print(f"Machines: {machines}")
    except Exception as e:
        print(f"Error listing remote machines: {e}")
        return
        
    print("\nForcing pull of memories from others...")
    try:
        pulled = manager.pull_memories_from_others()
        print(f"Successfully pulled {pulled} new memories from Google Drive!")
    except Exception as e:
        print(f"Error pulling memories: {e}")

if __name__ == "__main__":
    main()
