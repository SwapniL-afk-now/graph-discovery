import re

with open('/workspace/eval.log') as f:
    log = f.read()

blocks = re.split(r'(?=\[gvd\] )', log)
current = []
for b in blocks:
    if b.startswith('[gvd] '):
        current.append(b)
    else:
        if current:
            block = ''.join(current)
            m = re.search(r'([✓✗]) uid=(\d+)', block)
            if m:
                status, uid = m.group(1), m.group(2)
                tools = re.findall(r'\[gvd\] (\w+)\(', block)
                print(f'{status} uid={uid:>3}  {" -> ".join(tools)}')
            current = []
