"""
BackgroundSyncer: Daemon thread to pull commits periodically (or at startup).
"""

import threading
import logging
from pathlib import Path
from aivc.sync.drive import NativeDriveSyncManager

logger = logging.getLogger(__name__)

class BackgroundSyncer:
    """Daemon responsible for pulling distant commits at startup and potentially periodically."""
    
    def __init__(self, storage_root: Path, on_pull_callback=None):
        self.manager = NativeDriveSyncManager(storage_root)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread = None
        self._on_pull_callback = on_pull_callback

    def trigger_sync(self):
        """Wake up background syncer loop immediately."""
        self._wake_event.set()

    def start(self):
        """Start the background pull thread."""
        if not self.manager.enabled:
            return
            
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Sync pulls and pushes periodically."""
        while not self._stop_event.is_set():
            self._wake_event.clear()
            try:
                # 1. Pull from others
                pulled = self.manager.pull_memories_from_others()
                if pulled and pulled > 0:
                    logger.info("[AIVC Sync] Auto-pulled %d distant memories from Drive.", pulled)
                    if self._on_pull_callback:
                        self._on_pull_callback()

                # 2. Push missing local memories
                stats = self.manager.push_missing()
                pushed = stats.get("memories_pushed", 0)
                if pushed > 0:
                    logger.info("[AIVC Sync] Auto-pushed %d local memories to Drive.", pushed)

            except Exception as e:
                logger.exception("Background sync failed: %s", e)
                
            if self._stop_event.is_set():
                break

            self._wake_event.wait(timeout=60)

    def stop(self):
        """Stop the syncer."""
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=2)
