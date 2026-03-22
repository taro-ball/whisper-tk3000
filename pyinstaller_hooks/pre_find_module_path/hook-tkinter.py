# Preserve normal tkinter module discovery even if PyInstaller's Tcl/Tk probe
# fails under the active interpreter. The spec adds Tcl/Tk assets explicitly.


def pre_find_module_path(hook_api):
    return
