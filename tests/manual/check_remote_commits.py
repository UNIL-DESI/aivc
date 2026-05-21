import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from aivc.sync.drive import NativeDriveSyncManager

def main():
    storage_root = Path.home() / ".aivc" / "storage"
    manager = NativeDriveSyncManager(storage_root)
    if not manager.enabled:
        print("Sync is disabled.")
        return
        
    service = manager._get_service()
    root_id = manager._get_root_folder_id()
    print(f"Local machine_id: {manager.machine_id}")
    
    # List machine folders under root
    query = f"'{root_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])
    
    for mf in folders:
        print(f"\nMachine Folder: {mf['name']} (ID: {mf['id']})")
        if mf["name"] == manager.machine_id:
            print("  -> (This is the local machine)")
            
        # Find 'commits' folder inside this machine folder
        c_query = f"name = 'commits' and '{mf['id']}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        c_results = service.files().list(q=c_query, fields="files(id, name)").execute()
        c_folders = c_results.get("files", [])
        
        if not c_folders:
            print("  -> No 'commits' folder found.")
            continue
            
        c_folder = c_folders[0]
        print(f"  -> Found 'commits' folder (ID: {c_folder['id']})")
        
        # Count files inside 'commits'
        f_query = f"'{c_folder['id']}' in parents and trashed = false"
        f_results = service.files().list(q=f_query, fields="files(id, name)").execute()
        files = f_results.get("files", [])
        print(f"  -> Contains {len(files)} commit files.")
        if files:
            print(f"  -> Sample files: {[f['name'] for f in files[:5]]}")

if __name__ == "__main__":
    main()
