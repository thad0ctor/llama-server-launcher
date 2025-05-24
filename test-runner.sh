#!/bin/bash
gnome-terminal -- bash -c "python3 \"$1\"; echo \"\"; read -p \"Press Enter to close...\""
