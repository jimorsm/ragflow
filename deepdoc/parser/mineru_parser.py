#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from io import BytesIO
from os import PathLike
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Optional

import numpy as np
import pdfplumber
from PIL import Image
from strenum import StrEnum

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:  # pragma: no cover - fallback for environments with PyPDF2 only
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore

from deepdoc.parser.pdf_parser import RAGFlowPdfParser

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


class MinerUContentType(StrEnum):
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    EQUATION = "equation"


class MinerUParser(RAGFlowPdfParser):
    def __init__(self, mineru_path: str = "mineru"):
        self.mineru_path = Path(mineru_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_installation(self) -> bool:
        subprocess_kwargs = {
            "capture_output": True,
            "text": True,
            "check": True,
            "encoding": "utf-8",
            "errors": "ignore",
        }

        if platform.system() == "Windows":
            subprocess_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            result = subprocess.run([str(self.mineru_path), "--version"], **subprocess_kwargs)
            version_info = result.stdout.strip()
            if version_info:
                logging.info(f"[MinerU] Detected version: {version_info}")
            else:
                logging.info("[MinerU] Detected MinerU, but version info is empty.")
            return True
        except subprocess.CalledProcessError as e:
            logging.warning(f"[MinerU] Execution failed (exit code {e.returncode}).")
        except FileNotFoundError:
            logging.warning("[MinerU] MinerU not found. Please install it via: pip install -U 'mineru[core]'")
        except Exception as e:
            logging.error(f"[MinerU] Unexpected error during installation check: {e}")
        return False

    def _run_mineru(self, input_path: Path, output_dir: Path, method: str = "auto", lang: Optional[str] = None, *, backend: str = "pipeline"):
        cmd = [str(self.mineru_path), "-p", str(input_path), "-o", str(output_dir), "-m", method]
        if backend:
            cmd.extend(["-b", backend])
        if lang:
            cmd.extend(["-l", lang])

        self.logger.info(f"[MinerU] Running command: {' '.join(cmd)}")

        subprocess_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "ignore",
            "bufsize": 1,
        }

        if platform.system() == "Windows":
            subprocess_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        process = subprocess.Popen(cmd, **subprocess_kwargs)
        stdout_queue, stderr_queue = Queue(), Queue()

        def enqueue_output(pipe, queue, prefix):
            for line in iter(pipe.readline, ""):
                if line.strip():
                    queue.put((prefix, line.strip()))
            pipe.close()

        threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue, "STDOUT"), daemon=True).start()
        threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue, "STDERR"), daemon=True).start()

        while process.poll() is None:
            for q in (stdout_queue, stderr_queue):
                try:
                    while True:
                        prefix, line = q.get_nowait()
                        if prefix == "STDOUT":
                            self.logger.info(f"[MinerU] {line}")
                        else:
                            self.logger.warning(f"[MinerU] {line}")
                except Empty:
                    pass
            time.sleep(0.1)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"[MinerU] Process failed with exit code {return_code}")
        self.logger.info("[MinerU] Command completed successfully.")

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=600, callback=None):
        # Persist the page offset so downstream consumers can convert between
        # local slice indices (used during cropping) and absolute page numbers.
        self.page_from = page_from
        self.page_to = page_to
        try:
            with pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(BytesIO(fnm)) as pdf:
                self.pdf = pdf
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for _, p in enumerate(self.pdf.pages[page_from:page_to])]
        except Exception as e:
            self.page_images = None
            self.total_page = 0
            logging.exception(e)

    def _line_tag(self, bx):
        pn = [bx["page_idx"] + 1]
        positions = bx["bbox"]
        x0, top, x1, bott = positions

        if hasattr(self, "page_images") and self.page_images and len(self.page_images) > bx["page_idx"]:
            page_width, page_height = self.page_images[bx["page_idx"]].size
            x0 = (x0 / 1000.0) * page_width
            x1 = (x1 / 1000.0) * page_width
            top = (top / 1000.0) * page_height
            bott = (bott / 1000.0) * page_height

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), x0, x1, top, bott)

    def crop(self, text, ZM=1, need_position=False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return
        # When running on a sliced PDF we only store the pages inside the slice.
        # If page images are missing (e.g. pdfplumber failed) there is nothing to crop.
        if not getattr(self, "page_images", None):
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6

        def _local_index(page_idx: int) -> Optional[int]:
            local_idx = page_idx - getattr(self, "page_from", 0)
            if local_idx < 0 or local_idx >= len(self.page_images):
                return None
            return local_idx
        # The first and last positions correspond to the current block boundary
        # and we need to synthesise extra padding around them. Convert them to
        # local indices to avoid addressing outside the sliced page cache.

        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        end_local_idx = _local_index(pos[0][-1])
        if end_local_idx is None:
            if need_position:
                return None, None
            return
        end_page_height = self.page_images[end_local_idx].size[1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(end_page_height, pos[4] + GAP), min(end_page_height, pos[4] + 120)))

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            local_indices = []
            skip_entry = False
            for pn_abs in pns:
                local_idx = _local_index(pn_abs)
                if local_idx is None:
                    skip_entry = True
                    break
                local_indices.append(local_idx)
            if skip_entry:
                # Some spans still point to pages outside the current slice; skip
                # gracefully instead of raising an IndexError.
                continue

            right = left + max_width

            if bottom <= top:
                bottom = top + 2

            for pn_local in local_indices[1:]:
                bottom += self.page_images[pn_local - 1].size[1]

            img0 = self.page_images[local_indices[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0], x0, x1, y0, y1))

            bottom -= img0.size[1]
            for pn_abs, pn_local in zip(pns[1:], local_indices[1:]):
                page = self.page_images[pn_local]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(bottom, page.size[1]))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn_abs, x0, x1, y0, y1))
                bottom -= page.size[1]

        if not imgs:
            if need_position:
                return None, None
            return

        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    @staticmethod
    def extract_positions(txt: str):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def _read_output(self, output_dir: Path, file_stem: str, method: str = "auto", page_offset: int = 0) -> list[dict[str, Any]]:
        subdir = output_dir / file_stem / method
        json_file = subdir / f"{file_stem}_content_list.json"

        if not json_file.exists():
            raise FileNotFoundError(f"[MinerU] Missing output file: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if page_offset:
                if "page_idx" in item and isinstance(item["page_idx"], (int, float)):
                    item["page_idx"] += page_offset
                if "page_index" in item and isinstance(item["page_index"], (int, float)):
                    item["page_index"] += page_offset
                if "page_id" in item and isinstance(item["page_id"], (int, float)):
                    item["page_id"] += page_offset
                if "page_range" in item and isinstance(item["page_range"], list):
                    item["page_range"] = [p + page_offset for p in item["page_range"]]
            for key in ("img_path", "table_img_path", "equation_img_path"):
                if key in item and item[key]:
                    item[key] = str((subdir / item[key]).resolve())
        return data

    def _transfer_to_sections(self, outputs: list[dict[str, Any]]):
        sections = []
        for output in outputs:
            match output["type"]:
                case MinerUContentType.TEXT:
                    section = output["text"]
                case MinerUContentType.TABLE:
                    section = output.get("table_body", "") + "\n".join(output.get("table_caption", [])) + "\n".join(output.get("table_footnote", []))
                    if not section.strip():
                        section = "FAILED TO PARSE TABLE"
                case MinerUContentType.IMAGE:
                    section = "".join(output["image_caption"]) + "\n" + "".join(output["image_footnote"])
                case MinerUContentType.EQUATION:
                    section = output["text"]

            if section:
                sections.append((section, self._line_tag(output)))
        return sections

    def _transfer_to_tables(self, outputs: list[dict[str, Any]]):
        return []

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        backend: str = "pipeline",
        lang: Optional[str] = None,
        method: str = "auto",
        delete_output: bool = True,
        from_page: int = 0,
        to_page: int = 100000,
    ) -> tuple:
        import shutil

        temp_pdf = None
        created_tmp_dir = False
        subset_pdf = None
        subset_dir = None
        page_offset = max(0, int(from_page or 0))

        if binary:
            temp_dir = Path(tempfile.mkdtemp(prefix="mineru_bin_pdf_"))
            temp_pdf = temp_dir / Path(filepath).name
            with open(temp_pdf, "wb") as f:
                f.write(binary)
            pdf = temp_pdf
            self.logger.info(f"[MinerU] Received binary PDF -> {temp_pdf}")
            if callback:
                callback(0.15, f"[MinerU] Received binary PDF -> {temp_pdf}")
        else:
            pdf = Path(filepath)
            if not pdf.exists():
                if callback:
                    callback(-1, f"[MinerU] PDF not found: {pdf}")
                raise FileNotFoundError(f"[MinerU] PDF not found: {pdf}")

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="mineru_pdf_"))
            created_tmp_dir = True

        self.logger.info(f"[MinerU] Output directory: {out_dir}")
        if callback:
            callback(0.15, f"[MinerU] Output directory: {out_dir}")

        pdf_for_mineru = pdf
        total_pages = None
        try:
            reader = PdfReader(str(pdf))
            total_pages = len(reader.pages)
        except Exception:
            reader = None

        if total_pages is not None:
            start_idx = min(page_offset, total_pages)
            end_idx = min(int(to_page or total_pages), total_pages)
        else:
            start_idx = page_offset
            end_idx = int(to_page or 0)

        if total_pages is not None and end_idx <= start_idx:
            raise ValueError(f"[MinerU] Invalid page range: {start_idx}-{end_idx}")

        if reader is not None and (start_idx > 0 or (to_page and end_idx < total_pages)):
            subset_dir = Path(tempfile.mkdtemp(prefix="mineru_slice_"))
            subset_pdf = subset_dir / f"{Path(filepath).stem}_p{start_idx + 1}_{end_idx}.pdf"
            writer = PdfWriter()
            # Persist only the requested pages into a temporary PDF so MinerU processes this slice.
            for page_idx in range(start_idx, end_idx):
                writer.add_page(reader.pages[page_idx])
            with open(subset_pdf, "wb") as fp:
                writer.write(fp)
            pdf_for_mineru = subset_pdf
            page_offset = start_idx
        elif reader is None and (page_offset or (to_page and to_page not in (0, 100000))):
            page_offset = 0

        self.__images__(pdf_for_mineru, zoomin=1)
        self.page_from = page_offset

        try:
            self._run_mineru(pdf_for_mineru, out_dir, method=method, lang=lang, backend=backend)
            outputs = self._read_output(out_dir, pdf_for_mineru.stem, method=method, page_offset=page_offset)
            self.logger.info(f"[MinerU] Parsed {len(outputs)} blocks from PDF.")
            if callback:
                callback(0.75, f"[MinerU] Parsed {len(outputs)} blocks from PDF.")
            return self._transfer_to_sections(outputs), self._transfer_to_tables(outputs)
        finally:
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                    temp_pdf.parent.rmdir()
                except Exception:
                    pass
            if subset_pdf and subset_pdf.exists():
                try:
                    subset_pdf.unlink()
                except Exception:
                    pass
            if subset_dir and subset_dir.exists():
                try:
                    subset_dir.rmdir()
                except Exception:
                    pass
            if delete_output and created_tmp_dir and out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = MinerUParser("mineru")
    print("MinerU available:", parser.check_installation())

    filepath = ""
    with open(filepath, "rb") as file:
        outputs = parser.parse_pdf(filepath=filepath, binary=file.read())
        for output in outputs:
            print(output)
