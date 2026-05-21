import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from aivc.sync.drive import NativeDriveSyncManager

def debug_pull():
    storage_root = Path.home() / ".aivc" / "storage"
    manager = NativeDriveSyncManager(storage_root)
    if not manager.enabled:
        print("Sync is disabled.")
        return
        
    service = manager._get_service()
    root_id = manager._get_root_folder_id()
    print(f"Local machine ID: {manager.machine_id}")
    
    # List machine folders in AIVC_Sync
    query = f"'{root_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    machine_folders = results.get("files", [])
    print(f"Machine folders found: {[m['name'] for m in machine_folders]}")

    local_memories_dir = manager.storage_root / "commits"
    local_memories_dir.mkdir(parents=True, exist_ok=True)
    existing_memories = {f.name for f in local_memories_dir.iterdir() if f.suffix == ".json"}
    print(f"Local existing memories count: {len(existing_memories)}")
    
    pulled_count = 0

    for mf in machine_folders:
        if mf["name"] == manager.machine_id or mf["name"] == "blobs":
            print(f"Skipping folder: {mf['name']}")
            continue
            
        print(f"\nProcessing remote machine folder: {mf['name']}")
        
        # Find commits/ subfolder
        memories_query = (
            f"name = 'commits' and '{mf['id']}' in parents "
            f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        memories_result = service.files().list(q=memories_query, fields="files(id)").execute()
        memories_folders = memories_result.get("files", [])
        if not memories_folders:
            print("  -> No 'commits' folder found.")
            continue

        memories_folder_id = memories_folders[0]["id"]
        print(f"  -> Found 'commits' folder ID: {memories_folder_id}")

        # List memory files
        files_query = f"'{memories_folder_id}' in parents and trashed = false"
        
        page_token = None
        while True:
            files_result = service.files().list(
                q=files_query, spaces="drive", fields="nextPageToken, files(id, name)",
                pageSize=1000, pageToken=page_token
            ).execute()

            remote_files = files_result.get("files", [])
            print(f"  -> Found {len(remote_files)} files in remote 'commits' folder.")
            
            for remote_file in remote_files:
                if remote_file["name"] in existing_memories:
                    # Log first few skipped to keep clean
                    continue
                
                print(f"  -> Downloading missing file: {remote_file['name']}")
                try:
                    manager._download_file(remote_file["id"], local_memories_dir / remote_file["name"])
                    pulled_count += 1
                    existing_memories.add(remote_file["name"])
                except Exception as e:
                    print(f"  -> Error downloading {remote_file['name']}: {e}")

            page_token = files_result.get("nextPageToken")
            if not page_token:
                break
                
    print(f"\nCompleted! Pulled {pulled_count} memories.")

if __name__ == "__main__":
    debug_pull()
