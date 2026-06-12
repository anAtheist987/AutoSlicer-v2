"""Bilibili VOD downloader (browser-simulated web API, no login required).

Resolves BV/av ids, full URLs and b23.tv short links, then pulls DASH streams
via the public playurl API with browser headers. Without cookies bilibili
serves audio up to 192kbps (id 30280) and video up to 480p — plenty for the
16 kHz mono detection frontend; pass --cookie SESSDATA=... for higher video
quality on the cut clips.

Usage:
  python src/bilibili.py "https://www.bilibili.com/video/BV1xx411c7md" --out-dir downloads
  python src/bilibili.py BV1xx411c7md --video            # also fetch+mux video
  python src/bilibili.py "https://b23.tv/abc123" --page 2

For personal/research use only: respect bilibili's terms of service and the
streamers' rights to their content.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import requests

API_VIEW = "https://api.bilibili.com/x/web-interface/view"
API_PLAY = "https://api.bilibili.com/x/player/playurl"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
}


def _api(url: str, params: dict, cookie: str | None = None) -> dict:
    headers = dict(HEADERS)
    if cookie:
        headers["Cookie"] = cookie
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            d = r.json()
            if d.get("code") != 0:
                raise RuntimeError(f"bilibili API {url} -> code={d.get('code')} {d.get('message')}")
            return d["data"]
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def resolve_input(text: str) -> tuple[str, int]:
    """URL / BV id / av id / b23.tv short link -> (bvid, page)."""
    text = text.strip()
    if "b23.tv" in text:
        r = requests.get(text if text.startswith("http") else f"https://{text}",
                         headers=HEADERS, timeout=15, allow_redirects=True)
        text = r.url
    page = 1
    m = re.search(r"[?&]p=(\d+)", text)
    if m:
        page = int(m.group(1))
    m = re.search(r"(BV[0-9A-Za-z]{10})", text)
    if m:
        return m.group(1), page
    m = re.search(r"av(\d+)", text, re.IGNORECASE)
    if m:
        data = _api(API_VIEW, {"aid": m.group(1)})
        return data["bvid"], page
    raise ValueError(f"cannot parse bilibili id from: {text}")


def get_view(bvid: str, cookie: str | None = None) -> dict:
    return _api(API_VIEW, {"bvid": bvid}, cookie)


def get_dash(bvid: str, cid: int, cookie: str | None = None) -> dict:
    data = _api(API_PLAY, {"bvid": bvid, "cid": cid, "fnval": 16, "fourk": 1}, cookie)
    dash = data.get("dash")
    if not dash:
        raise RuntimeError("no DASH streams returned (try passing --cookie SESSDATA=...)")
    return dash


def _download(urls: list[str], out: Path, cookie: str | None = None,
              chunk: int = 1 << 20) -> None:
    """Stream one of `urls` (baseUrl + backups) to `out` with resume + retries."""
    headers = dict(HEADERS)
    if cookie:
        headers["Cookie"] = cookie
    last_err: Exception | None = None
    for attempt in range(4):
        url = urls[attempt % len(urls)]
        pos = out.stat().st_size if out.exists() else 0
        h = dict(headers)
        if pos:
            h["Range"] = f"bytes={pos}-"
        try:
            with requests.get(url, headers=h, timeout=30, stream=True) as r:
                if pos and r.status_code == 200:   # server ignored Range: restart
                    pos = 0
                r.raise_for_status()
                mode = "ab" if pos else "wb"
                with open(out, mode) as f:
                    for blk in r.iter_content(chunk):
                        f.write(blk)
            return
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after retries: {last_err}")


def _safe_name(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")[:80] or "video"


def download(url_or_id: str, out_dir: Path, want_video: bool = False,
             page: int | None = None, cookie: str | None = None) -> Path:
    """Fetch a VOD's audio (and optionally video) -> playable .m4a / .mp4 path."""
    bvid, url_page = resolve_input(url_or_id)
    page = page or url_page
    view = get_view(bvid, cookie)
    pages = view["pages"]
    if not 1 <= page <= len(pages):
        raise ValueError(f"page {page} out of range (video has {len(pages)} part(s))")
    cid = pages[page - 1]["cid"]
    title = _safe_name(view["title"] if len(pages) == 1
                       else f"{view['title']}_p{page}_{pages[page-1].get('part', '')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dash = get_dash(bvid, cid, cookie)

    audio = max(dash["audio"], key=lambda a: a["bandwidth"])
    a_raw = out_dir / f"{bvid}_p{page}.audio.m4s"
    print(f"[bilibili] {view['title']!r} p{page} ({pages[page-1]['duration']}s) "
          f"audio id={audio['id']} {audio['bandwidth']//1000}kbps", flush=True)
    _download([audio["baseUrl"]] + (audio.get("backupUrl") or []), a_raw, cookie)

    if not want_video:
        out = out_dir / f"{title}.m4a"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(a_raw),
                        "-c", "copy", str(out)], check=True)
        a_raw.unlink()
        return out

    video = max(dash["video"], key=lambda v: (v["id"], -len(v["codecs"])))
    v_raw = out_dir / f"{bvid}_p{page}.video.m4s"
    print(f"[bilibili] video id={video['id']} {video.get('width')}x{video.get('height')} "
          f"{video['codecs']}", flush=True)
    _download([video["baseUrl"]] + (video.get("backupUrl") or []), v_raw, cookie)
    out = out_dir / f"{title}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(v_raw), "-i", str(a_raw),
                    "-c", "copy", str(out)], check=True)
    v_raw.unlink()
    a_raw.unlink()
    return out


def is_bilibili_input(text: str) -> bool:
    t = text.strip()
    return bool(re.match(r"^BV[0-9A-Za-z]{10}$", t) or re.match(r"^av\d+$", t, re.IGNORECASE)
                or re.search(r"(bilibili\.com|b23\.tv)", t))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="bilibili URL / BV id / av id / b23.tv link")
    ap.add_argument("--out-dir", default="downloads")
    ap.add_argument("--video", action="store_true", help="also download video and mux")
    ap.add_argument("--page", type=int, default=None, help="part number for multi-part VODs")
    ap.add_argument("--cookie", default=None, help='e.g. "SESSDATA=..." for higher quality')
    args = ap.parse_args()
    out = download(args.input, Path(args.out_dir), want_video=args.video,
                   page=args.page, cookie=args.cookie)
    print(out)


if __name__ == "__main__":
    main()
