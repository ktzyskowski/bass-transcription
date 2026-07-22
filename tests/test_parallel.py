"""parallel_map: the process-pool helper. Both paths (serial fallback and the
spawn pool) must return results in input order. Workers run a module-level
function -- a closure would not pickle under spawn, which is the whole constraint
the helper documents.
"""

from stemify.parallel import parallel_map


def _square(x: int) -> int:
    return x * x


def test_serial_path_maps_in_order():
    assert parallel_map(_square, [1, 2, 3, 4], workers=1) == [1, 4, 9, 16]


def test_single_item_skips_pool():
    # <=1 item takes the serial branch even with workers>1 (pool not worth it).
    assert parallel_map(_square, [5], workers=8) == [25]


def test_empty_input():
    assert parallel_map(_square, [], workers=8) == []


def test_pool_path_matches_serial_and_preserves_order():
    items = list(range(20))
    pooled = parallel_map(_square, items, workers=4, chunksize=2)
    assert pooled == [_square(i) for i in items]


def test_on_result_streams_in_order_serial():
    seen = []
    parallel_map(_square, [2, 3, 4], workers=1, on_result=lambda i, r: seen.append((i, r)))
    assert seen == [(1, 4), (2, 9), (3, 16)]  # 1-based index, input order


def test_on_result_streams_in_order_pooled():
    seen = []
    out = parallel_map(
        _square, list(range(10)), workers=4, on_result=lambda i, r: seen.append((i, r))
    )
    # Called once per item, 1-based, in input order (imap preserves order).
    assert [i for i, _ in seen] == list(range(1, 11))
    assert [r for _, r in seen] == out
