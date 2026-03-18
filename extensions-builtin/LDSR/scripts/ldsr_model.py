import os
import hashlib
import requests
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from modules.modelloader import load_file_from_url
from modules.upscaler import Upscaler, UpscalerData
from ldsr_model_arch import LDSR
from modules import shared, script_callbacks, errors
import sd_hijack_autoencoder  # noqa: F401
import sd_hijack_ddpm_v1  # noqa: F401


def compute_sha256(file_path):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def quarantine_file(file_path, quarantine_dir=None):
    """Move file to quarantine directory with timestamp instead of deleting."""
    if not os.path.exists(file_path):
        return
    
    if quarantine_dir is None:
        quarantine_dir = os.path.join(os.path.dirname(file_path), "quarantine")
    
    os.makedirs(quarantine_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_path = os.path.join(quarantine_dir, f"{base_name}.{timestamp}")
    
    try:
        shutil.move(file_path, quarantine_path)
        print(f"Quarantined {file_path} to {quarantine_path}")
        return quarantine_path
    except Exception as e:
        print(f"Failed to quarantine {file_path}: {e}")
        return None


def atomic_rename(src, dst):
    """Atomic rename with rollback capability."""
    if not os.path.exists(src):
        return False
    
    if os.path.exists(dst):
        quarantine_file(dst)
    
    try:
        os.rename(src, dst)
        return True
    except Exception as e:
        print(f"Failed to rename {src} to {dst}: {e}")
        return False


def download_with_resume(url, target_path, expected_hash=None):
    """Download file with resume support and atomic operations."""
    temp_path = target_path + ".tmp"
    sidecar_path = target_path + ".sha256"
    
    # Check if target exists and is valid
    if os.path.exists(target_path):
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r") as f:
                    stored_hash = f.read().strip()
                actual_hash = compute_sha256(target_path)
                if actual_hash == stored_hash:
                    return target_path
                else:
                    print(f"Hash mismatch for {target_path}. Quarantining.")
                    quarantine_file(target_path)
                    if os.path.exists(sidecar_path):
                        os.remove(sidecar_path)
            except Exception as e:
                print(f"Error checking hash for {target_path}: {e}")
                quarantine_file(target_path)
        else:
            # No sidecar file, quarantine and re-download
            quarantine_file(target_path)
    
    # Check for partial download
    if os.path.exists(temp_path):
        try:
            head_response = requests.head(url, timeout=10)
            total_size = int(head_response.headers.get('content-length', 0))
            current_size = os.path.getsize(temp_path)
            
            if head_response.headers.get('accept-ranges') == 'bytes' and current_size < total_size:
                headers = {'Range': f'bytes={current_size}-'}
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                with open(temp_path, 'ab') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                # Can't resume or file is complete, start fresh
                os.remove(temp_path)
        except Exception as e:
            print(f"Error resuming download: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Download from scratch if no temp file
    if not os.path.exists(temp_path):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    # Verify download
    try:
        head_response = requests.head(url, timeout=10)
        total_size = int(head_response.headers.get('content-length', 0))
        downloaded_size = os.path.getsize(temp_path)
        
        if total_size > 0 and downloaded_size != total_size:
            print(f"Download incomplete for {url}. Expected {total_size}, got {downloaded_size}")
            quarantine_file(temp_path)
            raise Exception(f"Download incomplete for {url}")
        
        # Compute and store hash
        file_hash = compute_sha256(temp_path)
        with open(sidecar_path, "w") as f:
            f.write(file_hash)
        
        # Verify against expected hash if provided
        if expected_hash and file_hash != expected_hash:
            print(f"Hash verification failed for {url}")
            quarantine_file(temp_path)
            if os.path.exists(sidecar_path):
                os.remove(sidecar_path)
            raise Exception(f"Hash verification failed for {url}")
        
        # Atomic move to final location
        if os.path.exists(target_path):
            quarantine_file(target_path)
        
        os.rename(temp_path, target_path)
        return target_path
        
    except Exception as e:
        if os.path.exists(temp_path):
            quarantine_file(temp_path)
        raise


class UpscalerLDSR(Upscaler):
    def __init__(self, user_path):
        self.name = "LDSR"
        self.user_path = user_path
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        super().__init__()
        scaler_data = UpscalerData("LDSR", None, self)
        self.scalers = [scaler_data]

    def load_model(self, path: str):
        # Define paths
        yaml_path = os.path.join(self.model_path, "project.yaml")
        old_model_path = os.path.join(self.model_path, "model.pth")
        new_model_path = os.path.join(self.model_path, "model.ckpt")

        # Find existing local models
        local_model_paths = self.find_models(ext_filter=[".ckpt", ".safetensors"])
        local_ckpt_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("model.ckpt")]), None)
        local_safetensors_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("model.safetensors")]), None)
        local_yaml_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("project.yaml")]), None)

        # Handle invalid YAML file with quarantine instead of deletion
        if os.path.exists(yaml_path):
            try:
                statinfo = os.stat(yaml_path)
                if statinfo.st_size >= 10485760:
                    print("Quarantining invalid LDSR YAML file.")
                    quarantine_file(yaml_path)
            except Exception as e:
                print(f"Error checking YAML file: {e}")

        # Atomic rename of model.pth to model.ckpt
        if os.path.exists(old_model_path) and not os.path.exists(new_model_path):
            print("Renaming model from model.pth to model.ckpt")
            atomic_rename(old_model_path, new_model_path)

        # Determine model source
        if local_safetensors_path is not None and os.path.exists(local_safetensors_path):
            model = local_safetensors_path
        elif local_ckpt_path is not None and os.path.exists(local_ckpt_path):
            model = local_ckpt_path
        else:
            # Download with resilient operations
            model_target_path = os.path.join(self.model_download_path, "model.ckpt")
            try:
                model = download_with_resume(self.model_url, model_target_path)
            except Exception as e:
                print(f"Failed to download LDSR model: {e}")
                # Fallback to original loader
                model = load_file_from_url(self.model_url, model_dir=self.model_download_path, file_name="model.ckpt")

        # Determine YAML source
        if local_yaml_path is not None and os.path.exists(local_yaml_path):
            yaml = local_yaml_path
        else:
            # Download with resilient operations
            yaml_target_path = os.path.join(self.model_download_path, "project.yaml")
            try:
                yaml = download_with_resume(self.yaml_url, yaml_target_path)
            except Exception as e:
                print(f"Failed to download LDSR YAML: {e}")
                # Fallback to original loader
                yaml = load_file_from_url(self.yaml_url, model_dir=self.model_download_path, file_name="project.yaml")

        return LDSR(model, yaml)

    def do_upscale(self, img, path):
        try:
            ldsr = self.load_model(path)
        except Exception:
            errors.report(f"Failed loading LDSR model {path}", exc_info=True)
            return img
        ddim_steps = shared.opts.ldsr_steps
        return ldsr.super_resolution(img, ddim_steps, self.scale)


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("ldsr_steps", shared.OptionInfo(100, "LDSR processing steps. Lower = faster", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("ldsr_cached", shared.OptionInfo(False, "Cache LDSR model in memory", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")))


script_callbacks.on_ui_settings(on_ui_settings)