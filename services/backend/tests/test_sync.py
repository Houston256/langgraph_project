import pytest

from assistant.catalog.sync import change

PRODUCT = {"emb_hash": "e1", "payload_hash": "p1"}


@pytest.mark.parametrize(
    ("stored", "want"),
    [
        ({}, "embedded"),
        ({**PRODUCT, "emb_hash": "e0"}, "embedded"),
        ({**PRODUCT, "payload_hash": "p0", "in_feed": True}, "updated"),
        # back in the feed, same data as before it left
        ({**PRODUCT, "in_feed": False}, "updated"),
        ({**PRODUCT, "in_feed": True}, "unchanged"),
    ],
)
def test_change(stored, want):
    assert change(stored, PRODUCT) == want
