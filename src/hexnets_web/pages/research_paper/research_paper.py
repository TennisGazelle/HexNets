"""Research Paper page: displays the committed `docs/latex/main.pdf`.

This page never reads or renders LaTeX sources directly. The PDF is produced
out-of-band by `make pdf` (Docker + texlive-small) and committed to the repo;
this page just embeds the resulting binary.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import platform
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from hexnets_web.pages.base_page import BasePage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _pdf_path() -> Path:
    return _repo_root() / "docs" / "latex" / "main.pdf"


class ResearchPaperPage(BasePage):
    def render(self) -> None:
        st.header("Research Paper")

        st.warning(
            "This page embeds the PDF of the research paper for this repo.  This PDF is a work in progress and is protected by a CC-BY-NC-SA license. Unauthorized use or reproduction is prohibited and will be prosecuted.  For more information, contact the author at daniellopez123456789@gmail.com"
        )

        pdf_path = _pdf_path()
        if not pdf_path.exists():
            st.warning(
                "Paper PDF not found. Build it locally with `make pdf` and commit "
                f"`{pdf_path.relative_to(_repo_root())}`."
            )
            self._render_rebuild_controls()
            return

        mtime = datetime.fromtimestamp(pdf_path.stat().st_mtime)
        size_kb = pdf_path.stat().st_size / 1024
        st.caption(f"Built {mtime.strftime('%Y-%m-%d %H:%M:%S')} · {size_kb:,.1f} KB")

        pdf_bytes = pdf_path.read_bytes()
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="hexnets.pdf",
            mime="application/pdf",
        )

        st.pdf(pdf_path, height="stretch")

        self._render_rebuild_controls()

    def _render_rebuild_controls(self) -> None:
        if platform.processor() == "":
            return

        with st.expander("Rebuild PDF"):
            has_docker = shutil.which("docker") is not None
            has_make = shutil.which("make") is not None

            if not (has_docker and has_make):
                st.info("Rebuild requires Docker and `make` on the host (not available " "on Streamlit Cloud).")
                return

            if st.button("Run `make pdf`"):
                with st.spinner("Building PDF via Docker (texlive-small)…"):
                    result = subprocess.run(
                        ["make", "pdf"],
                        cwd=str(_repo_root()),
                        capture_output=True,
                        text=True,
                    )
                if result.returncode == 0:
                    st.success("PDF rebuilt. Refresh the page to see the new file.")
                else:
                    st.error(f"`make pdf` failed (exit {result.returncode}).")
                if result.stdout:
                    st.code(result.stdout, language="text")
                if result.stderr:
                    st.code(result.stderr, language="text")
