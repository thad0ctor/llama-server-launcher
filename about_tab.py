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
import subprocess
import requests
import threading
from datetime import datetime
import shutil


class AboutTab:
    """About tab for the llama-server-launcher application."""
    
    def __init__(self):
        self.version = self._load_version()
        self.github_url = "https://github.com/thad0ctor/llama-server-launcher"
        self.github_version_url = "https://raw.githubusercontent.com/thad0ctor/llama-server-launcher/main/version"
        self.donate_url = "https://www.paypal.me/thad0ctor"
        self.version_status = "Checking..."
        self.remote_version = None
        self.version_label = None
        self.update_button = None
        
    def _load_version(self):
        """Load version from the version file."""
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            version_file = script_dir / "version"
            
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
            # Get the current directory
            current_dir = Path(__file__).parent
            
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
        """Generate the update script content."""
        backup_dir = current_dir / "backup"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        
        script = f"""#!/bin/bash
set -e

echo "=== Llama Server Launcher Auto-Update ==="
echo "Current Version: {self.version}"
echo "Target Version: {self.remote_version}"
echo ""

# Create backup directory
echo "Creating backup directory..."
mkdir -p "{backup_path}"

# Copy files to backup (excluding JSON files)
echo "Backing up current files (excluding JSON files)..."
find "{current_dir}" -maxdepth 1 -type f ! -name "*.json" ! -name "update_script.sh" -exec cp {{}} "{backup_path}/" \\;

# Copy directories to backup (excluding .git if present)
for dir in "{current_dir}"/*; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != ".git" ] && [ "$(basename "$dir")" != "backup" ]; then
        echo "Backing up directory: $(basename "$dir")"
        cp -r "$dir" "{backup_path}/"
    fi
done

echo "Backup completed in: {backup_path}"
echo ""

# Remove old files (keep JSON files and backup directory)
echo "Removing old files..."
find "{current_dir}" -maxdepth 1 -type f ! -name "*.json" ! -name "update_script.sh" -delete
for dir in "{current_dir}"/*; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != "backup" ]; then
        echo "Removing directory: $(basename "$dir")"
        rm -rf "$dir"
    fi
done

echo "Old files removed."
echo ""

# Clone new version
echo "Cloning latest version from GitHub..."
cd "{current_dir}"
git clone {self.github_url} temp_clone
cd temp_clone

# Move files from temp clone to current directory
echo "Installing new version..."
mv * "{current_dir}/" 2>/dev/null || true
mv .* "{current_dir}/" 2>/dev/null || true
cd "{current_dir}"
rm -rf temp_clone

echo ""
echo "=== Update Complete ==="
echo "New version installed successfully!"
echo "Backup saved in: {backup_path}"
echo ""
echo "You can now restart the application."
echo ""
read -p "Press Enter to exit..."

# Clean up update script
rm -f "{current_dir}/update_script.sh"
"""
        return script
    
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