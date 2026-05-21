import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from aivc.sync.drive import NativeDriveSyncManager

def explore_node(service, folder_id, indent=""):
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query, spaces="drive", fields="files(id, name, mimeType)"
    ).execute()
    
    files = results.get("files", [])
    for f in files:
        if f["mimeType"] == "application/vnd.google-apps.folder":
            print(f"{indent}[DIR] {f['name']} (ID: {f['id']})")
            explore_node(service, f["id"], indent + "  ")
        else:
            print(f"{indent}[FILE] {f['name']} (ID: {f['id']})")

def main():
    storage_root = Path.home() / ".aivc" / "storage"
    manager = NativeDriveSyncManager(storage_root)
    if not manager.enabled:
        print("Sync is disabled.")
        return
        
    service = manager._get_service()
    root_id = manager._get_root_folder_id()
    print(f"Root Folder AIVC_Sync ID: {root_id}")
    
    print("\n--- Recursive Tree on Google Drive ---")
    explore_node(service, root_id)

if __name__ == "__main__":
    main()
