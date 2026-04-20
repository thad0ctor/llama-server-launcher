"""Settings tab: UI theme + font controls, persisted via app_settings."""

import sys
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont

import ui_theme


class SettingsTab:
    """Settings tab content. Reads/writes values on the launcher's app_settings dict."""

    THEME_MODE_LABELS = [
        ("Auto (follow OS/default)", "auto"),
        ("Light",                    "light"),
        ("Dark",                     "dark"),
        ("Specific theme…",          "specific"),
    ]

    # Mutually-exclusive font size presets. "0" = system default, "custom" = manual override.
    FONT_SIZE_PRESETS = [
        ("Default",  "0"),
        ("10",       "10"),
        ("12",       "12"),
        ("14",       "14"),
        ("16",       "16"),
        ("20",       "20"),
    ]
    FONT_SIZE_MAX = 32  # exclusive upper bound for custom override

    def __init__(self, launcher):
        self.launcher = launcher
        self.root = launcher.root

        s = launcher.app_settings
        self.theme_mode_var = tk.StringVar(value=s.get("ui_theme_mode", "auto"))
        self.theme_name_var = tk.StringVar(value=s.get("ui_theme_name", ""))
        self.font_family_var = tk.StringVar(value=s.get("ui_font_family", ""))

        # Font size: convert the persisted integer to our two-part state
        # (radio choice + custom Entry). If the stored size matches a preset,
        # select that preset; otherwise select "custom" and seed the entry.
        stored_size = int(s.get("ui_font_size", 0) or 0)
        preset_values = {int(v) for _, v in self.FONT_SIZE_PRESETS}
        if stored_size in preset_values:
            self.font_size_choice_var = tk.StringVar(value=str(stored_size))
            self.font_size_custom_var = tk.StringVar(value="")
        elif stored_size > 0:
            self.font_size_choice_var = tk.StringVar(value="custom")
            self.font_size_custom_var = tk.StringVar(value=str(stored_size))
        else:
            self.font_size_choice_var = tk.StringVar(value="0")
            self.font_size_custom_var = tk.StringVar(value="")

    # ------------------------------------------------------------------ setup
    def setup_settings_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(parent, text="UI Appearance", font=("TkDefaultFont", 12, "bold")) \
            .grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(10, 5))
        row += 1
        ttk.Separator(parent, orient="horizontal") \
            .grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        row += 1

        # --- Theme mode ---
        ttk.Label(parent, text="Theme mode:") \
            .grid(column=0, row=row, sticky="w", padx=10, pady=4)
        mode_frame = ttk.Frame(parent)
        mode_frame.grid(column=1, row=row, columnspan=2, sticky="w", padx=5, pady=4)
        for label, value in self.THEME_MODE_LABELS:
            ttk.Radiobutton(
                mode_frame, text=label, value=value,
                variable=self.theme_mode_var,
                command=self._on_theme_mode_changed,
            ).pack(side="left", padx=(0, 10))
        row += 1

        # --- Specific theme picker ---
        ttk.Label(parent, text="Specific theme:") \
            .grid(column=0, row=row, sticky="w", padx=10, pady=4)
        available = ui_theme.list_available_themes(self.root)
        self.theme_combo = ttk.Combobox(
            parent, textvariable=self.theme_name_var,
            values=available, state="readonly", width=30,
        )
        self.theme_combo.grid(column=1, row=row, sticky="w", padx=5, pady=4)
        ttk.Label(parent, text="(used when mode = Specific)", font=("TkSmallCaptionFont",)) \
            .grid(column=2, row=row, sticky="w", padx=5, pady=4)
        row += 1

        ttk.Separator(parent, orient="horizontal") \
            .grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=10)
        row += 1

        # --- Font family ---
        ttk.Label(parent, text="Font family:") \
            .grid(column=0, row=row, sticky="w", padx=10, pady=4)
        families = ui_theme.list_font_families(self.root)
        self.font_family_combo = ttk.Combobox(
            parent, textvariable=self.font_family_var,
            values=[""] + families, width=30,
        )
        self.font_family_combo.grid(column=1, row=row, sticky="w", padx=5, pady=4)
        ttk.Label(parent, text="(blank = system default)", font=("TkSmallCaptionFont",)) \
            .grid(column=2, row=row, sticky="w", padx=5, pady=4)
        row += 1

        # --- Font size (preset radios + custom override) ---
        ttk.Label(parent, text="Font size:") \
            .grid(column=0, row=row, sticky="nw", padx=10, pady=4)
        size_frame = ttk.Frame(parent)
        size_frame.grid(column=1, row=row, columnspan=2, sticky="w", padx=5, pady=4)

        # Presets
        for label, value in self.FONT_SIZE_PRESETS:
            ttk.Radiobutton(
                size_frame, text=label, value=value,
                variable=self.font_size_choice_var,
                command=self._on_font_size_choice_changed,
            ).pack(side="left", padx=(0, 8))

        # Custom radio + entry
        ttk.Radiobutton(
            size_frame, text="Custom:", value="custom",
            variable=self.font_size_choice_var,
            command=self._on_font_size_choice_changed,
        ).pack(side="left", padx=(0, 4))

        # Validation: restrict typing to digits only; range check happens at apply time.
        vcmd = (self.root.register(self._validate_custom_digit), "%P")
        self.font_size_custom_entry = ttk.Entry(
            size_frame, textvariable=self.font_size_custom_var,
            width=4, validate="key", validatecommand=vcmd,
        )
        self.font_size_custom_entry.pack(side="left")
        # Typing in the entry auto-selects the Custom radio so the user doesn't
        # have to click it manually.
        self.font_size_custom_entry.bind("<KeyRelease>", self._on_custom_entry_keyrelease)
        ttk.Label(size_frame, text=f"  (must be < {self.FONT_SIZE_MAX})",
                  font=("TkSmallCaptionFont",)).pack(side="left")
        row += 1

        ttk.Label(
            parent,
            text="(Font changes may require restarting the launcher to fully take effect.)",
            font=("TkSmallCaptionFont",),
        ).grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(0, 10))
        row += 1

        # --- Action buttons ---
        btns = ttk.Frame(parent)
        btns.grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(10, 5))
        ttk.Button(btns, text="Apply & Save", command=self._apply_and_save) \
            .pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Reset to Defaults", command=self._reset_defaults) \
            .pack(side="left", padx=(0, 8))
        row += 1

        self._status_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._status_var,
                  foreground="#5a9", font=("TkSmallCaptionFont",)) \
            .grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(0, 10))
        row += 1

        # --- Active appearance readout ---
        info_frame = ttk.LabelFrame(parent, text="Active appearance")
        info_frame.grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=(10, 10))
        info_frame.columnconfigure(1, weight=1)
        self._info_theme_var = tk.StringVar(value="")
        self._info_font_var = tk.StringVar(value="")
        ttk.Label(info_frame, text="Theme:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self._info_theme_var).grid(column=1, row=0, sticky="w", padx=6, pady=2)
        ttk.Label(info_frame, text="Font:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self._info_font_var).grid(column=1, row=1, sticky="w", padx=6, pady=2)
        row += 1

        self._on_theme_mode_changed()
        self._on_font_size_choice_changed()
        self._refresh_active_info()

    # ------------------------------------------------------------------ events
    def _on_theme_mode_changed(self):
        mode = self.theme_mode_var.get()
        state = "readonly" if mode == "specific" else "disabled"
        try:
            self.theme_combo.configure(state=state)
        except tk.TclError:
            pass

    def _on_font_size_choice_changed(self):
        """Enable the custom Entry only when 'Custom' is selected."""
        custom_selected = self.font_size_choice_var.get() == "custom"
        try:
            self.font_size_custom_entry.configure(
                state="normal" if custom_selected else "disabled",
            )
        except tk.TclError:
            pass

    def _on_custom_entry_keyrelease(self, _event):
        """User typed in the Custom entry — auto-select the Custom radio."""
        self.font_size_choice_var.set("custom")
        self._on_font_size_choice_changed()

    def _validate_custom_digit(self, proposed):
        """Entry validatecommand: only allow empty or pure digit strings up to 3 chars."""
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        if len(proposed) > 3:
            return False
        return True

    def _resolve_font_size(self):
        """Read the radio+entry combo and return an int, or raise ValueError on invalid."""
        choice = self.font_size_choice_var.get()
        if choice == "custom":
            raw = self.font_size_custom_var.get().strip()
            if not raw:
                raise ValueError("Enter a number for Custom font size.")
            if not raw.isdigit():
                raise ValueError(f"Font size must be an integer (got '{raw}').")
            size = int(raw)
            if size <= 0:
                raise ValueError("Font size must be greater than 0.")
            if size >= self.FONT_SIZE_MAX:
                raise ValueError(
                    f"Font size must be less than {self.FONT_SIZE_MAX} (got {size}).",
                )
            return size
        try:
            return int(choice)
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------ actions
    def _apply_and_save(self):
        mode = self.theme_mode_var.get()
        theme_name = self.theme_name_var.get() if mode == "specific" else ""
        family = self.font_family_var.get().strip()

        try:
            size = self._resolve_font_size()
        except ValueError as e:
            messagebox.showwarning("Invalid font size", str(e))
            return

        if mode == "specific" and not theme_name:
            messagebox.showwarning(
                "Select a theme",
                "Choose a theme name from the list when mode is 'Specific theme…'.",
            )
            return

        s = self.launcher.app_settings
        s["ui_theme_mode"] = mode
        s["ui_theme_name"] = theme_name
        s["ui_font_family"] = family
        s["ui_font_size"] = size

        # Always start from the shipped named-font values before layering the
        # user's overrides. apply_fonts only touches attributes that are set
        # (blank family or size<=0 are treated as "don't change"), so if the
        # user clears just one field after previously customising both, the old
        # value would otherwise stick — e.g. Arial/16 → blank family + 12
        # would render as Arial/12 instead of system-default/12.
        try:
            ui_theme.reset_fonts_to_system(self.root)
        except Exception as e:
            print(f"Font reset error: {e}", file=sys.stderr)

        try:
            ui_theme.apply_ui_preferences(
                self.root,
                theme_mode=mode,
                explicit_theme=theme_name or None,
                font_family=family,
                font_size=size,
            )
        except Exception as e:
            print(f"Settings apply error: {e}", file=sys.stderr)

        try:
            self.launcher._save_configs()
        except Exception as e:
            print(f"Settings save error: {e}", file=sys.stderr)
            messagebox.showerror("Save failed", f"Could not persist settings:\n{e}")
            return

        self._refresh_active_info()
        self._status_var.set("Settings applied and saved. Some changes may need a restart.")

    def _reset_defaults(self):
        """Clear all UI preferences and apply + persist immediately."""
        if not messagebox.askyesno(
            "Reset appearance",
            "Reset theme and font to system defaults?\n\n"
            "Font changes applied at runtime will revert to their shipped values. "
            "Some restored settings may only fully take effect after restarting the launcher.",
        ):
            return
        self.theme_mode_var.set("auto")
        self.theme_name_var.set("")
        self.font_family_var.set("")
        self.font_size_choice_var.set("0")
        self.font_size_custom_var.set("")
        self._on_theme_mode_changed()
        self._on_font_size_choice_changed()

        try:
            ui_theme.reset_fonts_to_system(self.root)
        except Exception as e:
            print(f"Font reset error: {e}", file=sys.stderr)

        s = self.launcher.app_settings
        s["ui_theme_mode"] = "auto"
        s["ui_theme_name"] = ""
        s["ui_font_family"] = ""
        s["ui_font_size"] = 0

        try:
            ui_theme.apply_ui_preferences(
                self.root, theme_mode="auto",
                explicit_theme=None, font_family="", font_size=0,
            )
        except Exception as e:
            print(f"Reset apply error: {e}", file=sys.stderr)

        try:
            self.launcher._save_configs()
        except Exception as e:
            print(f"Reset save error: {e}", file=sys.stderr)
            messagebox.showerror("Save failed", f"Could not persist reset:\n{e}")
            return

        self._refresh_active_info()
        self._status_var.set("Reset to defaults and saved. Restart for full effect if needed.")

    def _refresh_active_info(self):
        try:
            active_theme = ttk.Style(self.root).theme_use()
        except tk.TclError:
            active_theme = "(unknown)"
        try:
            default_font = tkfont.nametofont("TkDefaultFont", root=self.root)
            fam = default_font.cget("family")
            sz = int(default_font.cget("size") or 0)
            # Tk stores pixel sizes as negatives; display the absolute value and
            # whether it's points (positive) or pixels (negative original).
            unit = "pt" if sz >= 0 else "px"
            self._info_font_var.set(f"{fam} @ {abs(sz)}{unit}")
        except Exception:
            self._info_font_var.set("(unknown)")
        self._info_theme_var.set(str(active_theme))


def create_settings_tab(launcher):
    return SettingsTab(launcher)
