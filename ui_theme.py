"""UI appearance helpers: ttk theme selection, font overrides, and DPI scaling.

Used by both startup (to apply persisted UI preferences before widgets are built)
and the Settings tab (to re-apply preferences when the user changes them).
"""

import sys
import tkinter as tk
from tkinter import ttk, font as tkfont


LIGHT_PREFERRED = [
    "forest-light", "win11light", "vista", "xpnative", "winnative",
    "aqua", "clam", "alt", "default",
]
DARK_PREFERRED = [
    "forest-dark", "win11dark", "black", "equilux", "clam", "alt", "default",
]
AUTO_PREFERRED = [
    "forest-dark", "forest-light", "win11dark", "win11light",
    "vista", "xpnative", "winnative", "clam", "alt", "default",
]


def list_available_themes(root):
    try:
        return list(ttk.Style(root).theme_names())
    except tk.TclError:
        return []


def _looks_dark(theme_name):
    if not theme_name:
        return False
    n = theme_name.lower()
    return ("dark" in n) or ("black" in n) or ("highcontrast" in n) or n.endswith("-night")


def _pick_theme(available, preferred, fallback=None):
    for name in preferred:
        if name in available:
            return name
    if fallback and fallback in available:
        return fallback
    return available[0] if available else None


_STARTUP_THEME = None  # Tk's default theme captured before any user override


def _capture_startup_theme(style):
    """Remember the theme Tk was using before this module started applying
    anything, so 'auto' can fall back to it when OS dark-mode detection is
    unavailable instead of blindly picking a dark-leaning preset."""
    global _STARTUP_THEME
    if _STARTUP_THEME is None:
        try:
            _STARTUP_THEME = style.theme_use()
        except tk.TclError:
            _STARTUP_THEME = None
    return _STARTUP_THEME


def _detect_os_dark_mode():
    """Best-effort OS dark-mode detection.

    Returns True if the OS is in dark mode, False for light, None if unknown.
    Short timeouts and swallowed errors keep startup responsive when the
    detection tool isn't available.
    """
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            try:
                apps_use_light, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            finally:
                winreg.CloseKey(key)
            return not bool(apps_use_light)
        except (OSError, FileNotFoundError, ImportError):
            return None
    if sys.platform == "darwin":
        try:
            import subprocess
            r = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
            )
            return r.returncode == 0 and "Dark" in r.stdout
        except (OSError, subprocess.SubprocessError):
            return None
    # Linux / other — try GNOME first, then generic GTK.
    try:
        import subprocess
        r = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "dark" in r.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "dark" in r.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Colour matrices
# ═══════════════════════════════════════════════════════════════════════
# A "palette" is a dict mapping symbolic roles → colour strings. A single
# shared routine (_apply_style_palette) pushes the palette through every
# ttk style, option-database entry and existing classic-tk widget the app
# uses, so every theme goes through the exact same code path. That keeps
# transitions (auto → dark → auto, dark → specific → dark, etc.) clean:
# each call fully rewrites colour state, so there is never a half-dark /
# half-light residue from a previous theme.

_PALETTE_KEYS = (
    "bg",          # main frame background
    "bg_alt",      # secondary panels / hover
    "field_bg",    # entry / combobox / listbox background
    "fg",          # primary text
    "fg_muted",    # secondary / inactive text
    "sel_bg",      # selection background
    "sel_fg",      # selection foreground
    "border",      # widget borders
    "accent",      # progressbar / highlight colour
    "disabled_fg", # disabled text
    "btn_bg",      # ttk button background
    "btn_hover",   # button hover
    "tab_bg",      # unselected notebook tab background
    "tab_active",  # selected notebook tab background
)

DARK_PALETTE = {
    "bg":          "#1e1e1e",
    "bg_alt":      "#252526",
    "field_bg":    "#2d2d30",
    "fg":          "#e6e6e6",
    "fg_muted":    "#c0c0c0",
    "sel_bg":      "#094771",
    "sel_fg":      "#ffffff",
    "border":      "#3f3f46",
    "accent":      "#0e639c",
    "disabled_fg": "#7a7a7a",
    "btn_bg":      "#3c3c3c",
    "btn_hover":   "#505050",
    "tab_bg":      "#2d2d30",
    "tab_active":  "#1e1e1e",
}

# Generic light palette used when a specific theme doesn't define its own.
LIGHT_PALETTE = {
    "bg":          "#f0f0f0",
    "bg_alt":      "#e6e6e6",
    "field_bg":    "#ffffff",
    "fg":          "#000000",
    "fg_muted":    "#404040",
    "sel_bg":      "#0078d4",
    "sel_fg":      "#ffffff",
    "border":      "#a0a0a0",
    "accent":      "#0078d4",
    "disabled_fg": "#808080",
    "btn_bg":      "#e1e1e1",
    "btn_hover":   "#cccccc",
    "tab_bg":      "#e6e6e6",
    "tab_active":  "#f0f0f0",
}

# Per-theme palettes. Any theme not listed here falls back to LIGHT_PALETTE
# (or DARK_PALETTE when mode == "dark"). Tweak individual entries here if you
# want a specific theme to look more like its native appearance.
THEME_PALETTES = {
    # ----- Basic cross-platform ttk themes -----
    "clam":         {**LIGHT_PALETTE, "bg": "#dcdad5", "bg_alt": "#d0cec8",
                     "btn_bg": "#d9d9d9", "btn_hover": "#bfbfbf",
                     "tab_bg": "#c8c8c0", "tab_active": "#dcdad5"},
    "alt":          {**LIGHT_PALETTE, "bg": "#d9d9d9", "btn_bg": "#d9d9d9"},
    "default":      LIGHT_PALETTE,
    "classic":      {**LIGHT_PALETTE, "bg": "#d9d9d9", "field_bg": "#ffffff"},

    # ----- Windows-native themes (these already style well; palette just
    #       keeps classic-tk widgets in sync) -----
    "vista":        LIGHT_PALETTE,
    "xpnative":     LIGHT_PALETTE,
    "winnative":    LIGHT_PALETTE,
    "win11light":   {**LIGHT_PALETTE, "bg": "#f3f3f3", "field_bg": "#ffffff",
                     "accent": "#0067c0", "sel_bg": "#0067c0"},
    "win11dark":    {**DARK_PALETTE,  "bg": "#202020", "bg_alt": "#2b2b2b",
                     "field_bg": "#2d2d2d", "accent": "#60cdff",
                     "sel_bg":  "#4cc2ff", "sel_fg": "#000000"},

    # ----- macOS -----
    "aqua":         LIGHT_PALETTE,

    # ----- Third-party themes seen in the wild -----
    "forest-light": {**LIGHT_PALETTE, "bg": "#fafafa", "field_bg": "#ffffff",
                     "accent": "#217346", "sel_bg": "#217346"},
    "forest-dark":  {**DARK_PALETTE,  "bg": "#313131", "bg_alt": "#2b2b2b",
                     "field_bg": "#1d1d1d", "accent": "#217346",
                     "sel_bg": "#217346"},
    "black":        {**DARK_PALETTE,  "bg": "#000000", "bg_alt": "#0a0a0a",
                     "field_bg": "#1a1a1a"},
    "equilux":      {**DARK_PALETTE,  "bg": "#464646", "bg_alt": "#3f3f3f",
                     "field_bg": "#525252"},
}


def _is_dark_palette(palette):
    """Heuristic: is this a dark palette? (bg luminance roughly < 0.35)"""
    if not palette:
        return False
    bg = palette.get("bg", "")
    if not bg.startswith("#") or len(bg) < 7:
        return False
    try:
        r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
        # Perceptual luminance (sRGB approximation)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return lum < 0.35
    except ValueError:
        return False


def _palette_for(chosen_theme, mode):
    """Return the palette to apply given the chosen ttk theme and the user's mode."""
    if mode == "dark":
        # User explicitly asked for dark → prefer the theme's dark variant if
        # one exists (e.g. win11dark), otherwise the generic DARK_PALETTE.
        themed = THEME_PALETTES.get(chosen_theme)
        if themed and _is_dark_palette(themed):
            return themed
        return DARK_PALETTE
    if mode == "light":
        themed = THEME_PALETTES.get(chosen_theme)
        if themed and not _is_dark_palette(themed):
            return themed
        return LIGHT_PALETTE
    # auto / specific → honour the theme's own palette if we have one; else
    # infer from the theme name so a third-party dark theme (sun-valley-dark,
    # azure-dark, etc.) selected via "Specific" doesn't get a light palette
    # blasted over it, leaving unreadable text on dark backgrounds.
    themed = THEME_PALETTES.get(chosen_theme)
    if themed:
        return themed
    return DARK_PALETTE if _looks_dark(chosen_theme) else LIGHT_PALETTE


# ═══════════════════════════════════════════════════════════════════════
#  Shared palette application
# ═══════════════════════════════════════════════════════════════════════


def _apply_style_palette(root, style, palette):
    """Push ``palette`` through every widget class the app uses.

    This is the single entry point for colouring the UI — dark mode, light
    mode, auto and specific-theme selections all flow through here. That is
    why dark→light→dark transitions are clean: every call fully rewrites
    each style key, so nothing leaks between themes.
    """
    if not palette:
        return
    p = palette
    bg, bg_alt, field_bg = p["bg"], p["bg_alt"], p["field_bg"]
    fg, fg_muted = p["fg"], p["fg_muted"]
    sel_bg, sel_fg = p["sel_bg"], p["sel_fg"]
    btn_bg, btn_hover = p["btn_bg"], p["btn_hover"]
    disabled_fg = p["disabled_fg"]
    border = p["border"]
    tab_bg = p["tab_bg"]
    tab_active = p["tab_active"]
    accent = p["accent"]

    try:
        root.configure(bg=bg)
    except tk.TclError:
        pass

    # --- Base ttk styles ---------------------------------------------------
    try:
        style.configure(".", background=bg, foreground=fg,
                        fieldbackground=field_bg, bordercolor=border,
                        lightcolor=bg_alt, darkcolor=bg,
                        selectbackground=sel_bg, selectforeground=sel_fg)
        style.map(".",
                  foreground=[("disabled", disabled_fg)],
                  background=[("active", btn_hover)])

        for widget in ("TFrame", "TLabelframe", "TPanedwindow", "TSeparator"):
            style.configure(widget, background=bg)
        style.configure("TLabelframe.Label", background=bg, foreground=fg)

        for widget in ("TLabel", "TCheckbutton", "TRadiobutton", "TMenubutton"):
            style.configure(widget, background=bg, foreground=fg)
            style.map(widget,
                      background=[("active", bg_alt)],
                      foreground=[("disabled", disabled_fg)])

        style.configure("TButton", background=btn_bg, foreground=fg,
                        bordercolor=border, focusthickness=1, focuscolor=sel_bg)
        style.map("TButton",
                  background=[("active", btn_hover), ("pressed", sel_bg),
                              ("disabled", bg_alt)],
                  foreground=[("disabled", disabled_fg)])

        style.configure("TEntry", fieldbackground=field_bg, foreground=fg,
                        insertcolor=fg, bordercolor=border,
                        lightcolor=border, darkcolor=border,
                        selectbackground=sel_bg, selectforeground=sel_fg)
        style.map("TEntry",
                  fieldbackground=[("disabled", bg_alt), ("readonly", field_bg)],
                  foreground=[("disabled", disabled_fg), ("readonly", fg)])

        style.configure("TSpinbox", fieldbackground=field_bg, foreground=fg,
                        arrowcolor=fg, bordercolor=border, background=btn_bg,
                        insertcolor=fg, selectbackground=sel_bg,
                        selectforeground=sel_fg)
        style.map("TSpinbox",
                  fieldbackground=[("disabled", bg_alt), ("readonly", field_bg)],
                  foreground=[("disabled", disabled_fg), ("readonly", fg)],
                  background=[("active", btn_hover)])

        style.configure("TCombobox", fieldbackground=field_bg, foreground=fg,
                        background=btn_bg, arrowcolor=fg, bordercolor=border,
                        insertcolor=fg, selectbackground=sel_bg,
                        selectforeground=sel_fg)
        style.map("TCombobox",
                  fieldbackground=[("readonly", field_bg), ("!readonly", field_bg),
                                   ("disabled", bg_alt)],
                  foreground=[("readonly", fg), ("!readonly", fg),
                              ("disabled", disabled_fg)],
                  selectbackground=[("readonly", sel_bg)],
                  selectforeground=[("readonly", sel_fg)])

        style.configure("TNotebook", background=bg, bordercolor=border,
                        tabmargins=(2, 2, 2, 0))
        style.configure("TNotebook.Tab", background=tab_bg, foreground=fg_muted,
                        bordercolor=border, padding=(10, 4),
                        lightcolor=bg_alt, darkcolor=bg)
        style.map("TNotebook.Tab",
                  background=[("selected", tab_active), ("active", btn_hover)],
                  foreground=[("selected", fg), ("active", fg)])

        style.configure("TScrollbar", background=btn_bg, troughcolor=bg,
                        bordercolor=border, arrowcolor=fg,
                        lightcolor=border, darkcolor=border)
        style.map("TScrollbar",
                  background=[("active", btn_hover), ("pressed", sel_bg)],
                  arrowcolor=[("disabled", disabled_fg)])

        style.configure("TScale", background=bg, troughcolor=field_bg)
        style.configure("TProgressbar", background=accent, troughcolor=field_bg,
                        bordercolor=border, lightcolor=accent, darkcolor=accent)

        style.configure("Treeview", background=field_bg, foreground=fg,
                        fieldbackground=field_bg, bordercolor=border)
        style.map("Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", sel_fg)])
        style.configure("Treeview.Heading", background=btn_bg, foreground=fg,
                        bordercolor=border)
        style.map("Treeview.Heading",
                  background=[("active", btn_hover)])
    except tk.TclError as e:
        print(f"Note: ttk style palette application partial: {e}", file=sys.stderr)

    # --- Classic tk widgets via the option database -----------------------
    # option_add only affects widgets created AFTER the call, so we also
    # retroactively recolour existing ones below.
    option_defaults = (
        ("*Listbox.background", field_bg),
        ("*Listbox.foreground", fg),
        ("*Listbox.selectBackground", sel_bg),
        ("*Listbox.selectForeground", sel_fg),
        ("*Listbox.highlightBackground", border),
        ("*Listbox.highlightColor", border),
        ("*Text.background", field_bg),
        ("*Text.foreground", fg),
        ("*Text.insertBackground", fg),
        ("*Text.selectBackground", sel_bg),
        ("*Text.selectForeground", sel_fg),
        ("*Text.highlightBackground", border),
        ("*ScrolledText.background", field_bg),
        ("*ScrolledText.foreground", fg),
        ("*ScrolledText.insertBackground", fg),
        ("*ScrolledText.selectBackground", sel_bg),
        ("*ScrolledText.selectForeground", sel_fg),
        ("*Entry.background", field_bg),
        ("*Entry.foreground", fg),
        ("*Entry.insertBackground", fg),
        ("*Entry.selectBackground", sel_bg),
        ("*Entry.selectForeground", sel_fg),
        ("*Spinbox.background", field_bg),
        ("*Spinbox.foreground", fg),
        ("*Spinbox.insertBackground", fg),
        ("*Canvas.background", bg),
        ("*Canvas.highlightBackground", border),
        ("*Menu.background", bg_alt),
        ("*Menu.foreground", fg),
        ("*Menu.activeBackground", sel_bg),
        ("*Menu.activeForeground", sel_fg),
        ("*Menu.selectColor", fg),
        ("*Toplevel.background", bg),
        ("*TCombobox*Listbox.background", field_bg),
        ("*TCombobox*Listbox.foreground", fg),
        ("*TCombobox*Listbox.selectBackground", sel_bg),
        ("*TCombobox*Listbox.selectForeground", sel_fg),
    )
    for opt, val in option_defaults:
        try:
            root.option_add(opt, val)
        except tk.TclError:
            pass

    # --- Retroactively recolour existing classic-tk widgets ----------------
    _recolor_existing_tk_widgets(root, p)


def _recolor_existing_tk_widgets(root, p):
    """Walk the widget tree and recolour every classic-tk widget whose
    colours were baked in at construction time. ``option_add`` only affects
    widgets created after the call, so we need to touch existing ones too."""
    try:
        children = list(root.winfo_children())
    except tk.TclError:
        return
    while children:
        w = children.pop()
        try:
            children.extend(w.winfo_children())
        except tk.TclError:
            continue
        cls = w.winfo_class()
        try:
            if cls == "Listbox":
                w.configure(background=p["field_bg"], foreground=p["fg"],
                            selectbackground=p["sel_bg"],
                            selectforeground=p["sel_fg"],
                            highlightbackground=p["border"])
            elif cls == "Text":
                w.configure(background=p["field_bg"], foreground=p["fg"],
                            insertbackground=p["fg"],
                            selectbackground=p["sel_bg"],
                            selectforeground=p["sel_fg"],
                            highlightbackground=p["border"])
            elif cls == "Entry":
                w.configure(background=p["field_bg"], foreground=p["fg"],
                            insertbackground=p["fg"],
                            selectbackground=p["sel_bg"],
                            selectforeground=p["sel_fg"])
            elif cls == "Spinbox":
                w.configure(background=p["field_bg"], foreground=p["fg"],
                            insertbackground=p["fg"])
            elif cls == "Canvas":
                w.configure(background=p["bg"],
                            highlightbackground=p["border"])
            elif cls == "Frame":
                w.configure(background=p["bg"])
            elif cls == "Label":
                # Only touch classic-tk Labels; ttk ones raise TclError here.
                w.configure(background=p["bg"], foreground=p["fg"])
            elif cls == "Toplevel":
                w.configure(background=p["bg"])
            elif cls == "Menu":
                w.configure(background=p["bg_alt"], foreground=p["fg"],
                            activebackground=p["sel_bg"],
                            activeforeground=p["sel_fg"])
        except tk.TclError:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  Public theme entry point
# ═══════════════════════════════════════════════════════════════════════


def apply_theme(root, mode="auto", explicit_theme=None):
    """Apply a ttk theme chosen by mode, then push a palette over it.

    mode: "auto" | "light" | "dark" | "specific"
    explicit_theme: theme name to use when mode == "specific".

    Returns (applied_theme_name, is_dark).
    """
    try:
        style = ttk.Style(root)
        available = list(style.theme_names())
    except tk.TclError as e:
        print(f"ttk themes not available: {e}", file=sys.stderr)
        return (None, False)

    if not available:
        return (None, False)

    # Remember the theme Tk had active before any of our overrides land — used
    # as the last-resort fallback for "auto" when OS dark-mode detection fails.
    _capture_startup_theme(style)

    # ``effective_mode`` is what _palette_for actually sees. For "auto" we
    # resolve it to "dark"/"light" when we can detect the OS, so the palette
    # matches the theme we picked. "specific" falls through as-is so the
    # theme's own palette is used.
    effective_mode = mode
    if mode == "specific" and explicit_theme and explicit_theme in available:
        chosen = explicit_theme
    elif mode == "light":
        chosen = _pick_theme(available, LIGHT_PREFERRED)
    elif mode == "dark":
        chosen = _pick_theme(available, DARK_PREFERRED)
    else:  # auto — follow the OS if we can, else the startup default
        os_dark = _detect_os_dark_mode()
        if os_dark is True:
            chosen = _pick_theme(available, DARK_PREFERRED)
            effective_mode = "dark"
        elif os_dark is False:
            chosen = _pick_theme(available, LIGHT_PREFERRED)
            effective_mode = "light"
        elif _STARTUP_THEME and _STARTUP_THEME in available:
            chosen = _STARTUP_THEME
        else:
            chosen = _pick_theme(available, AUTO_PREFERRED)

    if not chosen:
        return (None, False)

    try:
        style.theme_use(chosen)
        print(f"Applied ttk theme: {chosen}", file=sys.stderr)
    except tk.TclError as e:
        print(f"Failed to apply theme {chosen}: {e}", file=sys.stderr)
        return (None, False)

    palette = _palette_for(chosen, effective_mode)
    _apply_style_palette(root, style, palette)
    is_dark = _is_dark_palette(palette)
    return (chosen, is_dark)


# Named fonts that most Tk widgets resolve against.
_NAMED_FONTS = (
    "TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont",
    "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont",
)


_ORIGINAL_FONTS = None  # cached {font_name: {"family": str, "size": int}}


def _capture_originals(root):
    """Snapshot each named font's original family + size on first call.

    We cache once, at startup before any overrides are applied, so we can
    (a) compute correct proportional deltas every time apply_fonts runs, and
    (b) restore system defaults when the user resets.
    """
    global _ORIGINAL_FONTS
    if _ORIGINAL_FONTS is not None:
        return _ORIGINAL_FONTS
    snapshot = {}
    for name in _NAMED_FONTS:
        try:
            f = tkfont.nametofont(name, root=root)
            snapshot[name] = {
                "family": f.cget("family") or "",
                "size": int(f.cget("size") or 0),
            }
        except tk.TclError:
            snapshot[name] = {"family": "", "size": 0}
    _ORIGINAL_FONTS = snapshot
    return snapshot


def apply_fonts(root, family="", size=0):
    """Update the built-in named fonts so every widget inherits the new settings.

    Empty/zero values mean "leave the existing value alone for that attribute".
    Proportional sizing uses each font's original RATIO to TkDefaultFont so the
    tooltip / small-caption / heading fonts scale together with the base size.
    We use |abs| because Tk may ship negative (pixel) defaults and the user
    always requests positive (point) values; a ratio is the only unit-agnostic
    way to preserve the proportions.
    """
    if not family and (not size or size <= 0):
        return
    originals = _capture_originals(root)
    default_orig = abs(originals.get("TkDefaultFont", {}).get("size", 0))
    for name in _NAMED_FONTS:
        try:
            f = tkfont.nametofont(name, root=root)
        except tk.TclError:
            continue
        kwargs = {}
        if family:
            kwargs["family"] = family
        if size and size > 0:
            if name == "TkDefaultFont" or not default_orig:
                kwargs["size"] = int(size)
            else:
                this_orig = abs(originals.get(name, {}).get("size", 0))
                ratio = (this_orig / default_orig) if default_orig else 1.0
                if ratio <= 0:
                    ratio = 1.0
                kwargs["size"] = max(6, int(round(int(size) * ratio)))
        try:
            f.configure(**kwargs)
        except tk.TclError as e:
            print(f"Note: could not update font {name}: {e}", file=sys.stderr)


def reset_fonts_to_system(root):
    """Restore every named font to the family/size that Tk shipped with."""
    originals = _capture_originals(root)
    for name, spec in originals.items():
        try:
            f = tkfont.nametofont(name, root=root)
            kwargs = {}
            if spec.get("family"):
                kwargs["family"] = spec["family"]
            if spec.get("size"):
                kwargs["size"] = int(spec["size"])
            if kwargs:
                f.configure(**kwargs)
        except tk.TclError:
            continue


def apply_ui_preferences(root, theme_mode="auto", explicit_theme=None,
                         font_family="", font_size=0):
    """Apply the user's persisted UI preferences: fonts first, then theme."""
    apply_fonts(root, family=font_family, size=font_size)
    return apply_theme(root, mode=theme_mode, explicit_theme=explicit_theme)


def list_font_families(root):
    """Sorted list of font family names available to Tk."""
    try:
        families = sorted({f for f in tkfont.families(root) if f and not f.startswith("@")})
        return families
    except tk.TclError:
        return []
