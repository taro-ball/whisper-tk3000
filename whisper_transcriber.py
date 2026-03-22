from __future__ import annotations

import argparse


def _import_startup_modules():
    import tkinter  # noqa: F401
    import tkinter.constants  # noqa: F401
    import tkinter.filedialog  # noqa: F401
    import tkinter.font  # noqa: F401
    import tkinter.ttk  # noqa: F401

    import whisper_tk3000.app as app_module

    return app_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-startup", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"Unrecognized arguments: {' '.join(unknown)}")

    app_module = _import_startup_modules()
    if args.smoke_startup:
        app_main = getattr(app_module, "main", None)
        if not callable(app_main):
            raise RuntimeError("whisper_tk3000.app.main is not callable")
        print("startup smoke passed")
        return 0

    app_module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
