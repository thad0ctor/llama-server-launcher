"""
About Tab Module for llama-server-launcher

This module provides an About tab with version information, GitHub link, and donation link.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from pathlib import Path
import sys
import os
import shlex
import subprocess
import requests
import threading
from datetime import datetime
import shutil


def build_update_script(current_dir, backup_path, current_version, remote_version,
                        github_url, exclusions):
    """Build the bash update script text.

    Pure helper extracted from :meth:`AboutTab._generate_update_script` so it
    can be tested with adversarial path inputs (paths containing ``$``, backtick,
    ``;``, quotes, etc.) without standing up a Tk GUI.

    All interpolated paths are passed through :func:`shlex.quote` before being
    embedded in the bash script body. ``github_url`` is also quoted even though
    it's currently a literal — defense in depth in case it ever becomes derived
    from remote data. Version strings are embedded in plain ``echo`` statements
    inside double-quoted strings, so they're also quoted for safety.

    Parameters
    ----------
    current_dir, backup_path:
        :class:`pathlib.Path` (or string) paths. Used in multiple places in the
        script. Each use site re-quotes the value — cheaper than carrying two
        versions around and prevents us accidentally using the un-quoted form.
    current_version, remote_version:
        Version strings displayed in ``echo`` headers. Quoted via shlex.
    github_url:
        Clone URL. Quoted via shlex.
    exclusions:
        Pre-built find-exclusion string produced by ``_get_backup_exclusions``.
        Already contains shell syntax, so it is NOT re-quoted.
    """
    q_current_dir = shlex.quote(str(current_dir))
    q_backup_path = shlex.quote(str(backup_path))
    q_current_version = shlex.quote(str(current_version))
    q_remote_version = shlex.quote(str(remote_version) if remote_version is not None else "")
    q_github_url = shlex.quote(str(github_url))

    script = f"""#!/bin/bash
set -e

echo "=== Llama Server Launcher Auto-Update ==="
echo "Current Version: "{q_current_version}
echo "Target Version: "{q_remote_version}
echo ""

# Create backup directory
echo "Creating backup directory..."
mkdir -p {q_backup_path}

# Backup files (excluding JSON files, .gitignore patterns, and git data)
echo "Backing up important files (excluding temporary/cache files)..."

# Create exclusion arguments for find
EXCLUDE_ARGS="-name backup -prune -o -name .git -prune -o -name update_script.sh -prune{exclusions}"

# Backup files
find {q_current_dir} -maxdepth 1 -type f $EXCLUDE_ARGS -name "*.py" -print -exec cp {{}} {q_backup_path}/ \\; -o \\
$EXCLUDE_ARGS -name "*.md" -print -exec cp {{}} {q_backup_path}/ \\; -o \\
$EXCLUDE_ARGS -name ".git*" -print -exec cp {{}} {q_backup_path}/ \\; 2>/dev/null || true

# Backup important directories (excluding .git, __pycache__, etc.)
for dir in {q_current_dir}/*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        case "$dirname" in
            .git|backup|images|__pycache__|*.egg-info|.pytest_cache|.mypy_cache)
                echo "Skipping $dirname (cache/git/static data)"
                ;;
            *)
                if [ ! -f {q_current_dir}/.gitignore ] || ! grep -q "^$dirname$" {q_current_dir}/.gitignore 2>/dev/null; then
                    echo "Backing up directory: $dirname"
                    cp -r "$dir" {q_backup_path}/
                fi
                ;;
        esac
    fi
done

echo "Backup completed in: "{q_backup_path}
echo ""

# Remove old files (keep JSON files and backup directory)
echo "Removing old files for clean installation..."

# Remove Python files and other source files
find {q_current_dir} -maxdepth 1 -type f -name "*.py" ! -name "update_script.sh" -delete 2>/dev/null || true
find {q_current_dir} -maxdepth 1 -type f -name "*.md" -delete 2>/dev/null || true
find {q_current_dir}/config -maxdepth 1 -type f -name "version" -delete 2>/dev/null || true
find {q_current_dir} -maxdepth 1 -type f -name ".git*" -delete 2>/dev/null || true

# Remove directories (except JSON config dirs, backup, and .git)
for dir in {q_current_dir}/*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        case "$dirname" in
            backup|.git|images)
                echo "Preserving $dirname"
                ;;
            *)
                echo "Removing directory: $dirname"
                rm -rf "$dir"
                ;;
        esac
    fi
done

echo "Old files cleaned up."
echo ""

# Clone new version
echo "Cloning latest version from GitHub..."
cd {q_current_dir}
git clone {q_github_url} temp_clone
cd temp_clone

# Move files from temp clone to current directory
echo "Installing new version..."
# Move all files except .git and images directories
find . -maxdepth 1 ! -name . ! -name .git ! -name images -exec mv {{}} {q_current_dir}/ \\; 2>/dev/null || true
echo "Skipped downloading images folder (using existing)"
cd {q_current_dir}
rm -rf temp_clone

# Restore user-owned gitignored state from backup. The fresh clone never
# contains llama_cpp_launcher_configs.json (it's gitignored), so without this
# step every self-update would silently wipe the user's saved configurations.
# We detect the layout of the freshly-cloned tree and write the restored file
# to the location the new code will look in.
echo "Restoring user configuration from backup..."
USER_CONFIG_SRC=""
if [ -f {q_backup_path}/config/llama_cpp_launcher_configs.json ]; then
    USER_CONFIG_SRC={q_backup_path}/config/llama_cpp_launcher_configs.json
elif [ -f {q_backup_path}/llama_cpp_launcher_configs.json ]; then
    USER_CONFIG_SRC={q_backup_path}/llama_cpp_launcher_configs.json
fi

if [ -n "$USER_CONFIG_SRC" ]; then
    if [ -d {q_current_dir}/config ] && [ -d {q_current_dir}/modules ]; then
        cp "$USER_CONFIG_SRC" {q_current_dir}/config/llama_cpp_launcher_configs.json
        echo "  Restored: config/llama_cpp_launcher_configs.json"
    else
        cp "$USER_CONFIG_SRC" {q_current_dir}/llama_cpp_launcher_configs.json
        echo "  Restored: llama_cpp_launcher_configs.json (repo root; will migrate on next launch)"
    fi
else
    echo "  No user configuration found in backup - clean install, nothing to restore."
fi

echo ""
echo "=== Update Complete ==="
echo "New version installed successfully!"
echo "Backup saved in: "{q_backup_path}
echo "User configurations were preserved and restored."
echo ""
echo "You can now restart the application."
echo ""
read -p "Press Enter to exit..."

# Clean up update script
rm -f {q_current_dir}/update_script.sh
"""
    return script


class AboutTab:
    """About tab for the llama-server-launcher application."""
    
    def __init__(self):
        self.version = self._load_version()
        self.github_url = "https://github.com/thad0ctor/llama-server-launcher"
        self.github_version_url = "https://raw.githubusercontent.com/thad0ctor/llama-server-launcher/main/config/version"
        self.donate_url = "https://www.paypal.me/thad0ctor"
        self.version_status = "Checking..."
        self.remote_version = None
        self.version_label = None
        self.update_button = None
        
    def _load_version(self):
        """Load version from the version file."""
        try:
            # Get the repo root directory (this module lives in modules/)
            script_dir = Path(__file__).parent.parent
            version_file = script_dir / "config" / "version"
            
            if version_file.exists():
                with open(version_file, 'r', encoding='utf-8') as f:
                    version = f.read().strip()
                    return version if version else "Unknown"
            else:
                return "Version file not found"
        except Exception as e:
            print(f"Error loading version: {e}", file=sys.stderr)
            return "Unknown"
    
    def _parse_version(self, version_str):
        """Parse version string in format YYYY-MM-DD-REV to comparable tuple."""
        try:
            if version_str in ["Unknown", "Version file not found"]:
                return (0, 0, 0, 0)
            
            parts = version_str.strip().split('-')
            if len(parts) != 4:
                return (0, 0, 0, 0)
            
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            rev = int(parts[3])
            
            return (year, month, day, rev)
        except (ValueError, IndexError):
            return (0, 0, 0, 0)
    
    def _is_version_newer(self, current_version, remote_version):
        """Compare two versions to determine if remote is newer."""
        current_tuple = self._parse_version(current_version)
        remote_tuple = self._parse_version(remote_version)
        return remote_tuple > current_tuple
    
    def _check_version_online(self):
        """Check version against GitHub repository."""
        try:
            response = requests.get(self.github_version_url, timeout=10)
            if response.status_code == 200:
                self.remote_version = response.text.strip()
                
                if self._is_version_newer(self.version, self.remote_version):
                    self.version_status = "Update Available"
                    self._update_version_display()
                    self._show_update_button()
                else:
                    self.version_status = "Current"
                    self._update_version_display()
            else:
                self.version_status = "Check Failed"
                self._update_version_display()
        except requests.RequestException as e:
            print(f"Error checking version: {e}", file=sys.stderr)
            self.version_status = "Check Failed"
            self._update_version_display()
    
    def _update_version_display(self):
        """Update the version display with status."""
        if self.version_label:
            version_text = f"Version: {self.version} ({self.version_status})"
            if self.remote_version and self.version_status == "Update Available":
                version_text += f" - Latest: {self.remote_version}"
            self.version_label.config(text=version_text)
    
    def _show_update_button(self):
        """Show the update button when an update is available."""
        if self.update_button:
            self.update_button.pack(pady=(10, 0))
    
    def _perform_update(self):
        """Perform the auto-update process."""
        result = messagebox.askyesno(
            "Auto Update", 
            f"Update from {self.version} to {self.remote_version}?\n\n"
            "This will:\n"
            "• Create a backup of current files (excluding JSON files)\n"
            "• Clone the latest version from GitHub\n"
            "• Open a new terminal window\n\n"
            "Continue with update?",
            icon='question'
        )
        
        if result:
            self._start_update_process()
    
    def _start_update_process(self):
        """Start the update process in a new terminal."""
        try:
            # Get the repo root directory (this module lives in modules/)
            current_dir = Path(__file__).parent.parent
            
            # Create update script
            script_content = self._generate_update_script(current_dir)
            script_path = current_dir / "update_script.sh"
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Open new terminal and run the update script
            if sys.platform.startswith('linux'):
                # Try different terminal emulators
                terminals = ['gnome-terminal', 'konsole', 'xterm', 'xfce4-terminal']
                for terminal in terminals:
                    try:
                        if terminal == 'gnome-terminal':
                            subprocess.Popen([terminal, '--', 'bash', str(script_path)], 
                                           cwd=current_dir)
                        elif terminal == 'konsole':
                            subprocess.Popen([terminal, '-e', 'bash', str(script_path)], 
                                           cwd=current_dir)
                        else:
                            subprocess.Popen([terminal, '-e', f'bash {script_path}'], 
                                           cwd=current_dir)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    # Fallback to xterm
                    subprocess.Popen(['xterm', '-e', f'bash {script_path}'], 
                                   cwd=current_dir)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', '-a', 'Terminal', str(script_path)], 
                               cwd=current_dir)
            elif sys.platform.startswith('win'):  # Windows
                subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', str(script_path)], 
                               cwd=current_dir, shell=True)
            
            messagebox.showinfo("Update Started", 
                              "Update process started in new terminal window.\n"
                              "Please follow the instructions in the terminal.")
            
        except Exception as e:
            messagebox.showerror("Update Error", f"Failed to start update process:\n{str(e)}")
    
    def _generate_update_script(self, current_dir):
        """Generate the update script content.

        Delegates to :func:`build_update_script`, a pure helper that can be
        tested independently of this class. All paths are passed through
        :func:`shlex.quote` when interpolated into the bash script body so a
        path containing shell metacharacters (``$``, backtick, ``;``, etc.)
        can't smuggle in extra commands at update time.
        """
        backup_dir = current_dir / "backup"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"

        # Generate exclusion patterns from .gitignore
        exclusions = self._get_backup_exclusions(current_dir)

        return build_update_script(
            current_dir=current_dir,
            backup_path=backup_path,
            current_version=self.version,
            remote_version=self.remote_version,
            github_url=self.github_url,
            exclusions=exclusions,
        )
    
    def _get_backup_exclusions(self, current_dir):
        """Generate find exclusion arguments based on .gitignore patterns."""
        exclusions = []
        gitignore_path = current_dir / ".gitignore"
        
        # Default exclusions (common patterns)
        default_exclusions = [
            "__pycache__", "*.pyc", "*.pyo", "*.pyd",
            ".pytest_cache", ".mypy_cache", ".coverage",
            "*.egg-info", ".tox", ".venv", "venv",
            ".DS_Store", "Thumbs.db", "*.tmp", "*.log"
        ]
        
        # Add exclusions from .gitignore if it exists
        try:
            if gitignore_path.exists():
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            default_exclusions.append(line)
        except Exception as e:
            print(f"Warning: Could not read .gitignore: {e}", file=sys.stderr)
        
        # Convert to find exclusion arguments
        for pattern in default_exclusions:
            if pattern:
                exclusions.append(f" -o -name '{pattern}' -prune")
        
        return "".join(exclusions)
    
    def _open_url(self, url):
        """Open URL in the default web browser."""
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening URL {url}: {e}", file=sys.stderr)
    
    def setup_about_tab(self, parent):
        """Set up the About tab UI."""
        # Create main frame with padding
        main_frame = ttk.Frame(parent, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Configure grid weights for centering
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=0, column=0, sticky="")
        
        row = 0
        
        # Title
        title_label = ttk.Label(content_frame, text="Llama.cpp Server Launcher", 
                               font=("TkDefaultFont", 16, "bold"))
        title_label.grid(row=row, column=0, pady=(0, 20))
        row += 1
        
        # Version information
        version_frame = ttk.LabelFrame(content_frame, text="Version Information", padding=15)
        version_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        
        # Version label that will be updated
        self.version_label = ttk.Label(version_frame, 
                                     text=f"Version: {self.version} ({self.version_status})", 
                                     font=("TkDefaultFont", 11))
        self.version_label.pack(anchor="w")
        
        # Update button (initially hidden)
        self.update_button = ttk.Button(version_frame, text="🔄 Update Available - Click to Update", 
                                      command=self._perform_update,
                                      style="Accent.TButton")  # Use accent style if available
        # Don't pack initially - will be shown when update is available
        
        # Start version check in background
        threading.Thread(target=self._check_version_online, daemon=True).start()
        
        # Project information
        project_frame = ttk.LabelFrame(content_frame, text="Project Information", padding=15)
        project_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        
        # Description
        description = ("A user-friendly GUI to easily configure and launch the llama.cpp server, "
                      "manage model configurations, set environment variables, and generate launch scripts.")
        desc_label = ttk.Label(project_frame, text=description, wraplength=400, justify="left")
        desc_label.pack(anchor="w", pady=(0, 10))
        
        # GitHub link
        github_frame = ttk.Frame(project_frame)
        github_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Label(github_frame, text="GitHub Repository:").pack(side="left")
        github_button = ttk.Button(github_frame, text="Visit GitHub", 
                                  command=lambda: self._open_url(self.github_url))
        github_button.pack(side="right")
        
        # Support section
        support_frame = ttk.LabelFrame(content_frame, text="Support the Project", padding=15)
        support_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        
        support_text = ("If you find this tool useful, consider supporting its development!")
        support_label = ttk.Label(support_frame, text=support_text, wraplength=400, justify="left")
        support_label.pack(anchor="w", pady=(0, 10))
        
        # Donate button
        donate_frame = ttk.Frame(support_frame)
        donate_frame.pack(fill="x")
        
        ttk.Label(donate_frame, text="Donate via PayPal:").pack(side="left")
        donate_button = ttk.Button(donate_frame, text="💝 Donate", 
                                  command=lambda: self._open_url(self.donate_url))
        donate_button.pack(side="right")
        
        # Credits section
        credits_frame = ttk.LabelFrame(content_frame, text="Credits", padding=15)
        credits_frame.grid(row=row, column=0, sticky="ew")
        
        credits_text = ("Built with Python and Tkinter\n"
                       "Designed for use with llama.cpp\n"
                       "Created by thad0ctor")
        credits_label = ttk.Label(credits_frame, text=credits_text, justify="left")
        credits_label.pack(anchor="w")
        
        # Configure column weights for proper sizing
        content_frame.columnconfigure(0, weight=1)
        version_frame.columnconfigure(0, weight=1)
        project_frame.columnconfigure(0, weight=1)
        support_frame.columnconfigure(0, weight=1)
        credits_frame.columnconfigure(0, weight=1)


# Factory function for creating the about tab
def create_about_tab():
    """Factory function to create an AboutTab instance."""
    return AboutTab() 
