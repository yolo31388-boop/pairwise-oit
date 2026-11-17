import pytest

from render.oit import (
    oit_weight,
    build_lists,
    resolve,
    sort_fragments,
    msaa_heads
)


def test_weight_front():
    assert oit_weight(1.0, 0.0, 1.0) == 1.0

def test_weight_deep():
    assert oit_weight(1.0, 10.0, 1.0) < 0.1

def test_weight_alpha():
    assert oit_weight(0.5, 5.0, 0.5) == pytest.approx(0.0833333)

def test_list_count():
    assert len(build_lists([(1.0, 0.0), (2.0, 1.0)], 2)) == 2

def test_list_pool():
    assert len(build_lists([(1.0, 0.0), (2.0, 1.0), (3.0, 2.0)], 2)) == 2

def test_resolve_single():
    assert resolve([(1.0, 0.0, 0.0, 0.0, 1.0)], (0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)

def test_resolve_order_independent():
    assert resolve([(1.0, 0.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 1.0, 0.5)], (0.0, 0.0, 0.0)) == resolve([(0.0, 1.0, 0.0, 1.0, 0.5), (1.0, 0.0, 0.0, 0.0, 0.5)], (0.0, 0.0, 0.0))

def test_resolve_empty():
    assert resolve([], (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)

def test_sort_depth():
    assert sort_fragments([(1, 5.0), (2, 1.0), (3, 3.0)]) == [(2, 1.0), (3, 3.0), (1, 5.0)]

def test_sort_stable():
    assert sort_fragments([(1, 2.0), (2, 2.0)]) == [(1, 2.0), (2, 2.0)]

def test_sort_empty():
    assert sort_fragments([]) == []

def test_msaa_count():
    assert len(msaa_heads(2, 2, 4)) == 16

def test_msaa_samples():
    assert len(msaa_heads(1, 1, 4)) == 4

def test_gamma_white():
    assert resolve([(1.0, 1.0, 1.0, 1.0, 1.0)], (0.0, 0.0, 0.0), gamma=2.2)[0] == pytest.approx(1.0)

def test_alpha_zero():
    assert resolve([(1.0, 0.0, 0.0, 0.0, 0.0)], (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
