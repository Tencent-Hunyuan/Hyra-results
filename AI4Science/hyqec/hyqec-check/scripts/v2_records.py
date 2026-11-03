"""Record-stream inventory for test tools (from scripts/v2-check-exported-certs.py at 0202d66); not a verifier."""
import collections


def inventory(raw, start):
    at = start
    records = []
    census = collections.Counter()
    task = None
    local_event = 0
    footer = None

    def integer():
        nonlocal at
        value = shift = 0
        while True:
            b = raw[at]
            at += 1
            value |= (b & 127) << shift
            if b < 128:
                return value
            shift += 7
            assert shift < 70

    def fixed():
        nonlocal at
        assert at + 8 <= len(raw)
        value = int.from_bytes(raw[at:at+8], 'little')
        at += 8
        return value

    while at < len(raw):
        begin = at
        tag = raw[at]
        at += 1
        record = dict(offset=begin, record_offset=begin-start, tag=f'{tag:02x}',
                      task=task, event=local_event)
        if tag == 0x10:
            record.update(parent=integer(), edge=integer())
            rule = raw[at]
            at += 1
            record.update(rule=rule, fingerprint=fixed(), weight=integer(), active=integer())
            if rule == 2 or rule in (5, 6, 7):
                record['packing'] = [integer() for _ in range(integer())]
            elif rule == 3:
                record['generator'] = integer()
            census[f'{"top" if task is None else "task"}:rule{rule}'] += 1
            local_event += 1
        elif tag in (0x11, 0x12, 0x13):
            integer(); integer(); integer(); fixed()
            local_event += 1
        elif tag == 0x1f:
            record.update(events=integer(), tasks=integer(), nodes=integer(), trace=f'{fixed():016x}')
        elif tag == 0x20:
            record.update(task_id=integer(), root=integer())
            length = integer()
            record['path'] = list(raw[at:at+length])
            at += length
            record.update(nodes=integer(), trace=f'{fixed():016x}')
        elif tag == 0x21:
            task = integer()
            local_event = 0
            record['task'] = task
        elif tag == 0x22:
            record['events'] = integer()
            task = None
        elif tag == 0x7f:
            footer = dict(tasks=integer(), gen_nodes=integer(), task_nodes=integer(),
                          total_nodes=integer(), global_trace_hex=f'{fixed():016x}')
            record.update(footer)
            assert at == len(raw), 'data follows COMPLETE'
        else:
            raise ValueError(f'unknown tag {tag:02x} at {begin}')
        record['end_offset'] = at
        records.append(record)
    assert footer is not None
    return records, dict(census), footer
