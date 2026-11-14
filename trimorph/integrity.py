"""Optional SHA256 cache; invalidate on file identity, size, mtime or ctime changes."""
import json
import time
from pathlib import Path

from shapnet.io import digest, write_json


def feature_digests(slides, cache_path=None):
    cache = {}
    path = Path(cache_path).resolve() if cache_path else None
    if path and path.exists():
        data = json.loads(path.read_text())
        if data.get('schema') != 1:
            raise ValueError('Unsupported feature hash cache schema')
        cache = data['files']
    result, updated = {}, {}
    for index, slide in enumerate(slides):
        feature = Path(slide.feature_path).resolve()
        stat = feature.stat()
        signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        entry = cache.get(str(feature))
        # Filesystems may coalesce timestamps across rapid writes. Never trust a
        # cache entry hashed in the same one-second timestamp window as its file.
        stable_at_hash = (entry is not None and 'hash_started_ns' in entry and
                          max(signature[-2:]) // 1_000_000_000 < entry['hash_started_ns'] // 1_000_000_000)
        if entry is None or entry['signature'] != signature or not stable_at_hash:
            started_ns = time.time_ns()
            value = digest(feature)
            after = feature.stat()
            if signature != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
                raise RuntimeError(f'Feature changed while hashing: {feature}')
            entry = dict(signature=signature, sha256=value, hash_started_ns=started_ns)
        updated[str(feature)] = entry
        result[slide.slide_id] = entry['sha256']
        if path and (index+1) % 25 == 0:
            write_json(path, dict(schema=1, files={**cache, **updated}))
            print(f'feature integrity: {index+1}/{len(slides)}', flush=True)
    if path:
        write_json(path, dict(schema=1, files=updated))
    return result
