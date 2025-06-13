"""
About Tab Module for llama-server-launcher

This module provides an About tab with version information, GitHub link, and donation link.
"""

import tkinter as tk
from tkinter import ttk
import webbrowser
from pathlib import Path
import sys
import os


class AboutTab:
    """About tab for the llama-server-launcher application."""
    
    def __init__(self):
        self.version = self._load_version()
        self.github_url = "https://github.com/thad0ctor/llama-server-launcher"
        self.donate_url = "https://www.paypal.me/thad0ctor"
    
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
        
        ttk.Label(version_frame, text=f"Version: {self.version}", 
                 font=("TkDefaultFont", 11)).pack(anchor="w")
        
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