"""
Atomic Filesystem Operations for Stable Diffusion WebUI
Resilient model loading with transactional safety, checksum validation, and automatic recovery.
"""

import os
import sys
import json
import time
import hashlib
import shutil
import tempfile
import threading
import contextlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
QUARANTINE_DIR = "quarantine"
TRANSACTION_LOG = "atomic_fs_transactions.json"
CHUNK_SIZE = 8192
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DOWNLOAD_TIMEOUT = 30  # seconds

class OperationType(Enum):
    """Types of atomic operations."""
    WRITE = "write"
    RENAME = "rename"
    DELETE = "delete"
    DOWNLOAD = "download"
    COPY = "copy"

class FileStatus(Enum):
    """Status of a file in the system."""
    HEALTHY = "healthy"
    CORRUPTED = "corrupted"
    QUARANTINED = "quarantined"
    MISSING = "missing"
    DOWNLOADING = "downloading"

@dataclass
class TransactionRecord:
    """Record of an atomic operation for recovery."""
    operation_id: str
    operation_type: OperationType
    source_path: Optional[str]
    target_path: Optional[str]
    temp_path: Optional[str]
    checksum: Optional[str]
    timestamp: float
    status: str = "pending"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['operation_type'] = self.operation_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionRecord':
        """Create from dictionary."""
        data['operation_type'] = OperationType(data['operation_type'])
        return cls(**data)

class AtomicFilesystem:
    """
    Resilient filesystem operations with transactional safety.
    Provides atomic writes, checksum validation, quarantine system, and download resume.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize AtomicFilesystem.
        
        Args:
            base_dir: Base directory for operations. Defaults to current working directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.quarantine_dir = self.base_dir / QUARANTINE_DIR
        self.transaction_log = self.base_dir / TRANSACTION_LOG
        self._lock = threading.RLock()
        self._active_transactions: Dict[str, TransactionRecord] = {}
        
        # Ensure directories exist
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing transactions
        self._load_transactions()
    
    def _load_transactions(self) -> None:
        """Load transaction records from log file."""
        if not self.transaction_log.exists():
            return
        
        try:
            with open(self.transaction_log, 'r') as f:
                data = json.load(f)
                for record_data in data.get('transactions', []):
                    record = TransactionRecord.from_dict(record_data)
                    self._active_transactions[record.operation_id] = record
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load transaction log: {e}")
            # Backup corrupted log
            backup_path = self.transaction_log.with_suffix('.bak')
            try:
                shutil.copy2(self.transaction_log, backup_path)
                logger.info(f"Backed up corrupted transaction log to {backup_path}")
            except Exception:
                pass
    
    def _save_transactions(self) -> None:
        """Save transaction records to log file."""
        try:
            data = {
                'transactions': [record.to_dict() for record in self._active_transactions.values()],
                'timestamp': time.time()
            }
            
            # Atomic write for transaction log
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.base_dir,
                prefix='.transaction_',
                suffix='.json'
            )
            
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            if sys.platform == 'win32':
                # Windows requires special handling
                if self.transaction_log.exists():
                    self.transaction_log.unlink()
            
            Path(temp_path).replace(self.transaction_log)
            
        except Exception as e:
            logger.error(f"Failed to save transaction log: {e}")
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        return f"{int(time.time() * 1000)}_{os.getpid()}_{threading.get_ident()}"
    
    @staticmethod
    def calculate_checksum(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
        """
        Calculate checksum of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm (default: sha256)
            
        Returns:
            Hex digest of the checksum
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(CHUNK_SIZE):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _validate_file(self, file_path: Union[str, Path], expected_checksum: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a file's integrity.
        
        Args:
            file_path: Path to the file
            expected_checksum: Optional expected checksum
            
        Returns:
            Tuple of (is_valid, actual_checksum)
        """
        path = Path(file_path)
        
        if not path.exists():
            return False, None
        
        if expected_checksum:
            actual_checksum = self.calculate_checksum(path)
            return actual_checksum == expected_checksum, actual_checksum
        
        # If no expected checksum, check if file is readable and not empty
        try:
            with open(path, 'rb') as f:
                f.read(1024)  # Read first 1KB to check readability
            return True, None
        except Exception:
            return False, None
    
    def quarantine_file(self, file_path: Union[str, Path], reason: str = "corrupted") -> Optional[Path]:
        """
        Move a file to quarantine instead of deleting it.
        
        Args:
            file_path: Path to the file to quarantine
            reason: Reason for quarantine
            
        Returns:
            Path to quarantined file or None if failed
        """
        source = Path(file_path)
        
        if not source.exists():
            logger.warning(f"Cannot quarantine non-existent file: {source}")
            return None
        
        # Generate quarantine filename with timestamp
        timestamp = int(time.time())
        quarantine_name = f"{source.stem}_{timestamp}_{reason}{source.suffix}"
        quarantine_path = self.quarantine_dir / quarantine_name
        
        # Handle duplicates
        counter = 1
        while quarantine_path.exists():
            quarantine_name = f"{source.stem}_{timestamp}_{reason}_{counter}{source.suffix}"
            quarantine_path = self.quarantine_dir / quarantine_name
            counter += 1
        
        try:
            shutil.move(str(source), str(quarantine_path))
            logger.info(f"Quarantined {source} to {quarantine_path}")
            return quarantine_path
        except Exception as e:
            logger.error(f"Failed to quarantine {source}: {e}")
            return None
    
    @contextlib.contextmanager
    def atomic_write(self, target_path: Union[str, Path], mode: str = 'wb', 
                    expected_checksum: Optional[str] = None, 
                    validate_after_write: bool = True):
        """
        Context manager for atomic file writes.
        
        Args:
            target_path: Final destination path
            mode: File open mode (default: 'wb')
            expected_checksum: Optional checksum to validate after write
            validate_after_write: Whether to validate file after writing
            
        Yields:
            File object for writing
        """
        target = Path(target_path)
        operation_id = self._generate_operation_id()
        temp_fd = None
        temp_path = None
        
        # Create transaction record
        transaction = TransactionRecord(
            operation_id=operation_id,
            operation_type=OperationType.WRITE,
            source_path=None,
            target_path=str(target),
            temp_path=None,
            checksum=expected_checksum,
            timestamp=time.time()
        )
        
        try:
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            # Create temporary file in same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=target.parent,
                prefix=f'.{target.stem}_',
                suffix=target.suffix
            )
            
            transaction.temp_path = temp_path
            transaction.status = "writing"
            
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            # Yield file object for writing
            with os.fdopen(temp_fd, mode) as f:
                temp_fd = None  # fdopen takes ownership
                yield f
            
            # Validate if requested
            if validate_after_write and expected_checksum:
                is_valid, actual_checksum = self._validate_file(temp_path, expected_checksum)
                if not is_valid:
                    raise ValueError(
                        f"Checksum mismatch for {target_path}. "
                        f"Expected: {expected_checksum}, Got: {actual_checksum}"
                    )
            
            # Atomic rename
            if sys.platform == 'win32':
                # Windows requires special handling
                if target.exists():
                    target.unlink()
            
            Path(temp_path).replace(target)
            
            transaction.status = "completed"
            
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            logger.debug(f"Atomic write completed: {target}")
            
        except Exception as e:
            # Clean up temporary file on error
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
            
            transaction.status = "failed"
            transaction.error = str(e)
            
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            logger.error(f"Atomic write failed for {target}: {e}")
            raise
        
        finally:
            # Clean up transaction record after completion
            with self._lock:
                if operation_id in self._active_transactions:
                    del self._active_transactions[operation_id]
                    self._save_transactions()
    
    def atomic_download(self, url: str, target_path: Union[str, Path],
                       expected_checksum: Optional[str] = None,
                       headers: Optional[Dict[str, str]] = None,
                       progress_callback: Optional[Callable[[int, int], None]] = None,
                       resume: bool = True) -> bool:
        """
        Download a file atomically with resume support and checksum validation.
        
        Args:
            url: URL to download from
            target_path: Destination file path
            expected_checksum: Optional expected SHA256 checksum
            headers: Optional HTTP headers
            progress_callback: Optional callback(downloaded, total)
            resume: Whether to resume interrupted downloads
            
        Returns:
            True if successful, False otherwise
        """
        target = Path(target_path)
        operation_id = self._generate_operation_id()
        temp_path = None
        
        # Create transaction record
        transaction = TransactionRecord(
            operation_id=operation_id,
            operation_type=OperationType.DOWNLOAD,
            source_path=url,
            target_path=str(target),
            temp_path=None,
            checksum=expected_checksum,
            timestamp=time.time()
        )
        
        try:
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            # Check for existing partial download
            partial_path = target.with_suffix(target.suffix + '.part')
            downloaded_bytes = 0
            
            if resume and partial_path.exists():
                downloaded_bytes = partial_path.stat().st_size
                logger.info(f"Resuming download from {downloaded_bytes} bytes")
            
            # Prepare request headers
            request_headers = headers.copy() if headers else {}
            if downloaded_bytes > 0:
                request_headers['Range'] = f'bytes={downloaded_bytes}-'
            
            # Create request
            request = urllib.request.Request(url, headers=request_headers)
            
            # Open connection with timeout
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                # Get total size
                total_size = int(response.headers.get('Content-Length', 0))
                if downloaded_bytes > 0:
                    # For resumed downloads, total is partial + remaining
                    total_size += downloaded_bytes
                
                # Check if server supports range requests
                content_range = response.headers.get('Content-Range')
                if downloaded_bytes > 0 and not content_range:
                    # Server doesn't support resume, start over
                    logger.warning("Server doesn't support resume, starting fresh download")
                    downloaded_bytes = 0
                    if partial_path.exists():
                        partial_path.unlink()
                
                # Create temporary file
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=target.parent,
                    prefix=f'.download_',
                    suffix=target.suffix
                )
                
                transaction.temp_path = temp_path
                transaction.status = "downloading"
                
                with self._lock:
                    self._active_transactions[operation_id] = transaction
                    self._save_transactions()
                
                # Copy existing partial data if resuming
                if downloaded_bytes > 0 and partial_path.exists():
                    with open(partial_path, 'rb') as src, os.fdopen(temp_fd, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                        temp_fd = None  # Already closed by os.fdopen
                else:
                    # Just close the temp file, we'll open it again
                    os.close(temp_fd)
                    temp_fd = None
                
                # Download with progress
                with open(temp_path, 'ab' if downloaded_bytes > 0 else 'wb') as f:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded_bytes, total_size)
                
                # Validate checksum if provided
                if expected_checksum:
                    logger.info(f"Validating checksum for {target}")
                    is_valid, actual_checksum = self._validate_file(temp_path, expected_checksum)
                    
                    if not is_valid:
                        # Quarantine the corrupted download
                        quarantine_path = self.quarantine_file(
                            temp_path,
                            reason=f"checksum_mismatch_expected_{expected_checksum[:8]}"
                        )
                        temp_path = None  # Already moved to quarantine
                        
                        raise ValueError(
                            f"Downloaded file checksum mismatch. "
                            f"Expected: {expected_checksum}, Got: {actual_checksum}. "
                            f"File quarantined at: {quarantine_path}"
                        )
                
                # Atomic rename to final destination
                if sys.platform == 'win32':
                    if target.exists():
                        target.unlink()
                
                Path(temp_path).replace(target)
                temp_path = None  # Successfully moved
                
                # Clean up partial file if exists
                if partial_path.exists():
                    partial_path.unlink()
                
                transaction.status = "completed"
                
                with self._lock:
                    self._active_transactions[operation_id] = transaction
                    self._save_transactions()
                
                logger.info(f"Successfully downloaded {url} to {target}")
                return True
                
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP error downloading {url}: {e.code} {e.reason}"
            logger.error(error_msg)
            transaction.status = "failed"
            transaction.error = error_msg
            
        except urllib.error.URLError as e:
            error_msg = f"URL error downloading {url}: {e.reason}"
            logger.error(error_msg)
            transaction.status = "failed"
            transaction.error = error_msg
            
        except Exception as e:
            error_msg = f"Failed to download {url}: {str(e)}"
            logger.error(error_msg)
            transaction.status = "failed"
            transaction.error = error_msg
            
        finally:
            # Clean up on failure
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
            
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
                
                # Keep failed transactions for debugging
                if transaction.status == "failed":
                    # Clean up after 24 hours
                    def cleanup():
                        time.sleep(86400)  # 24 hours
                        with self._lock:
                            if operation_id in self._active_transactions:
                                del self._active_transactions[operation_id]
                                self._save_transactions()
                    
                    threading.Thread(target=cleanup, daemon=True).start()
                else:
                    if operation_id in self._active_transactions:
                        del self._active_transactions[operation_id]
                        self._save_transactions()
        
        return False
    
    def atomic_rename(self, source: Union[str, Path], target: Union[str, Path]) -> bool:
        """
        Atomically rename a file.
        
        Args:
            source: Source path
            target: Target path
            
        Returns:
            True if successful, False otherwise
        """
        source_path = Path(source)
        target_path = Path(target)
        operation_id = self._generate_operation_id()
        
        transaction = TransactionRecord(
            operation_id=operation_id,
            operation_type=OperationType.RENAME,
            source_path=str(source_path),
            target_path=str(target_path),
            temp_path=None,
            checksum=None,
            timestamp=time.time()
        )
        
        try:
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic rename
            if sys.platform == 'win32':
                if target_path.exists():
                    target_path.unlink()
            
            source_path.replace(target_path)
            
            transaction.status = "completed"
            
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            logger.debug(f"Atomic rename: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to rename {source_path} to {target_path}: {e}"
            logger.error(error_msg)
            
            transaction.status = "failed"
            transaction.error = error_msg
            
            with self._lock:
                self._active_transactions[operation_id] = transaction
                self._save_transactions()
            
            return False
        
        finally:
            with self._lock:
                if operation_id in self._active_transactions:
                    del self._active_transactions[operation_id]
                    self._save_transactions()
    
    def recover_transactions(self) -> Dict[str, Any]:
        """
        Attempt to recover from interrupted transactions.
        
        Returns:
            Dictionary with recovery results
        """
        results = {
            'recovered': [],
            'failed': [],
            'quarantined': []
        }
        
        with self._lock:
            transactions = list(self._active_transactions.values())
        
        for transaction in transactions:
            try:
                if transaction.status == "completed":
                    # Already completed, just clean up
                    continue
                
                if transaction.operation_type == OperationType.WRITE:
                    # Check if temp file exists
                    if transaction.temp_path and Path(transaction.temp_path).exists():
                        # Try to validate and move to target
                        if transaction.checksum:
                            is_valid, _ = self._validate_file(
                                transaction.temp_path,
                                transaction.checksum
                            )
                            
                            if is_valid:
                                # Move to target
                                target_path = Path(transaction.target_path)
                                if sys.platform == 'win32' and target_path.exists():
                                    target_path.unlink()
                                
                                Path(transaction.temp_path).replace(target_path)
                                results['recovered'].append(transaction.operation_id)
                            else:
                                # Quarantine corrupted file
                                quarantine_path = self.quarantine_file(
                                    transaction.temp_path,
                                    reason="recovery_checksum_mismatch"
                                )
                                results['quarantined'].append({
                                    'operation_id': transaction.operation_id,
                                    'quarantine_path': str(quarantine_path)
                                })
                        else:
                            # No checksum, just move
                            target_path = Path(transaction.target_path)
                            if sys.platform == 'win32' and target_path.exists():
                                target_path.unlink()
                            
                            Path(transaction.temp_path).replace(target_path)
                            results['recovered'].append(transaction.operation_id)
                
                elif transaction.operation_type == OperationType.DOWNLOAD:
                    # Check for partial download
                    if transaction.target_path:
                        target_path = Path(transaction.target_path)
                        partial_path = target_path.with_suffix(target_path.suffix + '.part')
                        
                        if partial_path.exists():
                            # Try to validate partial file
                            if transaction.checksum:
                                is_valid, _ = self._validate_file(
                                    partial_path,
                                    transaction.checksum
                                )
                                
                                if is_valid:
                                    # Move to target
                                    if sys.platform == 'win32' and target_path.exists():
                                        target_path.unlink()
                                    
                                    partial_path.replace(target_path)
                                    results['recovered'].append(transaction.operation_id)
                                else:
                                    # Quarantine corrupted partial
                                    quarantine_path = self.quarantine_file(
                                        partial_path,
                                        reason="recovery_partial_corrupted"
                                    )
                                    results['quarantined'].append({
                                        'operation_id': transaction.operation_id,
                                        'quarantine_path': str(quarantine_path)
                                    })
                            else:
                                # No checksum, move to target anyway
                                if sys.platform == 'win32' and target_path.exists():
                                    target_path.unlink()
                                
                                partial_path.replace(target_path)
                                results['recovered'].append(transaction.operation_id)
                
                # Clean up transaction
                with self._lock:
                    if transaction.operation_id in self._active_transactions:
                        del self._active_transactions[transaction.operation_id]
                
            except Exception as e:
                logger.error(f"Failed to recover transaction {transaction.operation_id}: {e}")
                results['failed'].append({
                    'operation_id': transaction.operation_id,
                    'error': str(e)
                })
        
        # Save cleaned up transactions
        self._save_transactions()
        
        return results
    
    def verify_file(self, file_path: Union[str, Path], 
                   expected_checksum: Optional[str] = None) -> FileStatus:
        """
        Verify the status of a file.
        
        Args:
            file_path: Path to the file
            expected_checksum: Optional expected checksum
            
        Returns:
            FileStatus enum value
        """
        path = Path(file_path)
        
        if not path.exists():
            return FileStatus.MISSING
        
        if expected_checksum:
            is_valid, _ = self._validate_file(path, expected_checksum)
            if not is_valid:
                return FileStatus.CORRUPTED
        
        # Check if file is in quarantine
        try:
            quarantine_path = self.quarantine_dir / path.name
            if quarantine_path.exists():
                return FileStatus.QUARANTINED
        except Exception:
            pass
        
        return FileStatus.HEALTHY
    
    def cleanup_quarantine(self, max_age_days: int = 30) -> int:
        """
        Clean up old quarantined files.
        
        Args:
            max_age_days: Maximum age in days for quarantined files
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned_count = 0
        
        for file_path in self.quarantine_dir.iterdir():
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned up quarantined file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up quarantined file {file_path}: {e}")
        
        return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the filesystem operations.
        
        Returns:
            Dictionary with statistics
        """
        quarantine_count = sum(1 for _ in self.quarantine_dir.iterdir() if _.is_file())
        
        return {
            'active_transactions': len(self._active_transactions),
            'quarantined_files': quarantine_count,
            'quarantine_dir': str(self.quarantine_dir),
            'transaction_log': str(self.transaction_log)
        }

# Global instance for convenience
_default_instance = None

def get_default_instance() -> AtomicFilesystem:
    """Get the default AtomicFilesystem instance."""
    global _default_instance
    if _default_instance is None:
        # Use models directory as base
        from modules import paths_internal
        _default_instance = AtomicFilesystem(paths_internal.models_path)
    return _default_instance

# Convenience functions using default instance
def atomic_write(target_path: Union[str, Path], mode: str = 'wb', 
                expected_checksum: Optional[str] = None):
    """Atomic write using default instance."""
    return get_default_instance().atomic_write(target_path, mode, expected_checksum)

def atomic_download(url: str, target_path: Union[str, Path],
                   expected_checksum: Optional[str] = None,
                   headers: Optional[Dict[str, str]] = None,
                   progress_callback: Optional[Callable[[int, int], None]] = None,
                   resume: bool = True) -> bool:
    """Atomic download using default instance."""
    return get_default_instance().atomic_download(
        url, target_path, expected_checksum, headers, progress_callback, resume
    )

def calculate_checksum(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Calculate checksum using default instance."""
    return AtomicFilesystem.calculate_checksum(file_path, algorithm)

def quarantine_file(file_path: Union[str, Path], reason: str = "corrupted") -> Optional[Path]:
    """Quarantine file using default instance."""
    return get_default_instance().quarantine_file(file_path, reason)

def verify_file(file_path: Union[str, Path], 
               expected_checksum: Optional[str] = None) -> FileStatus:
    """Verify file using default instance."""
    return get_default_instance().verify_file(file_path, expected_checksum)

# Integration with existing LDSR loader
class LDSRModelLoader:
    """
    Wrapper for LDSR model loading with atomic operations.
    Integrates with existing LDSR code while providing safety.
    """
    
    def __init__(self):
        self.afs = get_default_instance()
    
    def safe_load_ldsr_model(self, model_path: Union[str, Path], 
                           yaml_path: Union[str, Path],
                           expected_checksum: Optional[str] = None) -> bool:
        """
        Safely load LDSR model with atomic operations.
        
        Args:
            model_path: Path to model file
            yaml_path: Path to YAML config
            expected_checksum: Optional model checksum
            
        Returns:
            True if successful
        """
        model_path = Path(model_path)
        yaml_path = Path(yaml_path)
        
        # Verify model file
        model_status = self.afs.verify_file(model_path, expected_checksum)
        
        if model_status == FileStatus.MISSING:
            logger.error(f"LDSR model not found: {model_path}")
            return False
        
        if model_status == FileStatus.CORRUPTED:
            logger.warning(f"LDSR model corrupted, quarantining: {model_path}")
            self.afs.quarantine_file(model_path, reason="ldsr_model_corrupted")
            return False
        
        if model_status == FileStatus.QUARANTINED:
            logger.error(f"LDSR model is quarantined: {model_path}")
            return False
        
        # Verify YAML file exists
        if not yaml_path.exists():
            logger.error(f"LDSR YAML config not found: {yaml_path}")
            return False
        
        # Safe YAML file operations (instead of dangerous rename/delete)
        # Create backup of YAML before any operations
        backup_path = yaml_path.with_suffix('.yaml.bak')
        try:
            if yaml_path.exists() and not backup_path.exists():
                shutil.copy2(yaml_path, backup_path)
                logger.debug(f"Created YAML backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create YAML backup: {e}")
        
        return True
    
    def download_ldsr_model(self, url: str, model_path: Union[str, Path],
                          yaml_url: Optional[str] = None,
                          yaml_path: Optional[Union[str, Path]] = None,
                          expected_checksum: Optional[str] = None,
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """
        Download LDSR model and config atomically.
        
        Args:
            url: Model download URL
            model_path: Destination for model file
            yaml_url: Optional YAML config URL
            yaml_path: Optional destination for YAML config
            expected_checksum: Optional model checksum
            progress_callback: Optional progress callback
            
        Returns:
            True if successful
        """
        # Download model
        if not self.afs.atomic_download(url, model_path, expected_checksum, 
                                       progress_callback=progress_callback):
            return False
        
        # Download YAML if provided
        if yaml_url and yaml_path:
            if not self.afs.atomic_download(yaml_url, yaml_path):
                # Quarantine the model if YAML download fails
                self.afs.quarantine_file(model_path, reason="yaml_download_failed")
                return False
        
        return True

# Module initialization
logger.info("Atomic filesystem module loaded")