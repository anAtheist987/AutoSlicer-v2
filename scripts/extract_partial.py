"""Extract fully-transferred entries from a still-uploading zip (stored or deflate).

Walks local file headers sequentially; extracts every entry whose data is fully
present. Safe to re-run; skips files already extracted with the right size.
"""
import struct
import sys
import zlib
from pathlib import Path

ZIP = '/root/AgentGateway/数据集.zip'
OUT = Path('/root/Autoslicer/data/raw')


def main():
    size = Path(ZIP).stat().st_size
    f = open(ZIP, 'rb')
    pos = 0
    n_ok = n_skip = 0
    while True:
        f.seek(pos)
        hdr = f.read(30)
        if len(hdr) < 30 or hdr[:4] != b'PK\x03\x04':
            break
        flags, method = struct.unpack('<HH', hdr[6:10])
        comp_size, uncomp_size = struct.unpack('<II', hdr[18:26])
        nlen, elen = struct.unpack('<HH', hdr[26:30])
        raw_name = f.read(nlen)
        extra = f.read(elen)
        # ZIP64: 32-bit size fields are 0xFFFFFFFF, real sizes in extra field 0x0001
        if uncomp_size == 0xFFFFFFFF or comp_size == 0xFFFFFFFF:
            ep = 0
            while ep + 4 <= len(extra):
                eid, esz = struct.unpack('<HH', extra[ep:ep + 4])
                if eid == 0x0001:
                    body = extra[ep + 4: ep + 4 + esz]
                    bp = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack('<Q', body[bp:bp + 8])[0]
                        bp += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack('<Q', body[bp:bp + 8])[0]
                    break
                ep += 4 + esz
        data_start = pos + 30 + nlen + elen
        if flags & 0x8 and comp_size == 0:
            print('data-descriptor entry, cannot stream:', raw_name, file=sys.stderr)
            break
        try:
            name = raw_name.decode('utf-8') if flags & 0x800 else raw_name.decode('gbk')
        except UnicodeDecodeError:
            name = raw_name.decode('utf-8', 'replace')
        data_end = data_start + comp_size
        if data_end > size:
            print(f'INCOMPLETE tail entry: {name} (needs {data_end-size} more bytes)')
            break
        target = OUT / name
        if name.endswith('/'):
            target.mkdir(parents=True, exist_ok=True)
        elif target.exists() and target.stat().st_size == uncomp_size:
            n_skip += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            f.seek(data_start)
            remaining = comp_size
            d = zlib.decompressobj(-15) if method == 8 else None
            with open(target, 'wb') as dst:
                while remaining > 0:
                    chunk = f.read(min(1 << 22, remaining))
                    remaining -= len(chunk)
                    dst.write(d.decompress(chunk) if d else chunk)
                if d:
                    dst.write(d.flush())
            n_ok += 1
            print(f'extracted {name} ({uncomp_size/1e6:.1f}MB)')
        pos = data_end
        if flags & 0x8:  # data descriptor follows
            f.seek(pos)
            dd = f.read(16)
            pos += 16 if dd[:4] == b'PK\x07\x08' else 12
    print(f'done: {n_ok} extracted, {n_skip} already present, scanned up to {pos}/{size} bytes')


if __name__ == '__main__':
    main()
