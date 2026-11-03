"""只使用标准库采集进程峰值 RSS 和可见的 Linux cgroup 内存计数。"""

import json
from pathlib import Path
import sys


def peak_rss_mib():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(value / (1024**2 if sys.platform == 'darwin' else 1024), 2)
    except (ImportError, OSError):
        return None


def snapshot():
    result = {'python': sys.executable, 'platform': sys.platform, 'peak_rss_mib': peak_rss_mib(), 'cgroup_files': {}}
    candidates = {Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')}
    try:
        entries = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        _, controllers, relative = entry.split(':', 2)
        # 挂载命名空间内路径可能不可直接解析；同时检查根目录。
        if '..' in Path(relative).parts:
            continue
        if not controllers:
            candidates.add(Path('/sys/fs/cgroup') / relative.lstrip('/'))
        elif 'memory' in controllers.split(','):
            candidates.add(Path('/sys/fs/cgroup/memory') / relative.lstrip('/'))
    for directory in sorted(candidates):
        for name in ('memory.max', 'memory.current', 'memory.events', 'memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.failcnt'):
            path = directory / name
            try:
                result['cgroup_files'][str(path)] = path.read_text().strip()
            except OSError:
                pass
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
