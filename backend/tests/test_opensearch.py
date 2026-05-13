"""
Tests for backend/app/opensearch_client.py

Covers:
  - index_document constructs correct PUT URL and body
  - index_document propagates HTTP errors
  - search_similar parses hits and returns text list
  - search_similar passes k to the request body
  - search_similar propagates HTTP errors
"""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("OPENSEARCH_ENDPOINT", "http://mock-opensearch")


def _fake_response(json_data=None, raise_for_status=False):
    resp = MagicMock()
    if raise_for_status:
        import requests

        resp.raise_for_status.side_effect = requests.HTTPError("error")
    else:
        resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


class TestIndexDocument:
    def test_makes_put_request_to_correct_url(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(req_mod, "put", return_value=_fake_response()) as mock_put:
            osc.index_document("doc1", "user1", 0, "hello", [0.1, 0.2])
            url = mock_put.call_args[0][0]
            assert "doc1_0" in url
            assert osc.INDEX in url

    def test_body_contains_all_fields(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(req_mod, "put", return_value=_fake_response()) as mock_put:
            osc.index_document("d", "u", 2, "text here", [0.5])
            body = mock_put.call_args[1]["json"]
            assert body["text"] == "text here"
            assert body["doc_id"] == "d"
            assert body["user_id"] == "u"
            assert body["chunk_id"] == 2
            assert body["embedding"] == [0.5]

    def test_raises_on_http_error(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(
            req_mod, "put", return_value=_fake_response(raise_for_status=True)
        ):
            with pytest.raises(req_mod.HTTPError):
                osc.index_document("d", "u", 0, "t", [])


class TestSearchSimilar:
    def _hits(self, texts):
        return {"hits": {"hits": [{"_source": {"text": t}} for t in texts]}}

    def test_returns_text_list(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(
            req_mod,
            "post",
            return_value=_fake_response(json_data=self._hits(["a", "b"])),
        ):
            result = osc.search_similar([0.1], k=2)
            assert result == ["a", "b"]

    def test_empty_hits_returns_empty_list(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(
            req_mod, "post", return_value=_fake_response(json_data=self._hits([]))
        ):
            result = osc.search_similar([0.1], k=5)
            assert result == []

    def test_k_is_passed_in_request_body(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(
            req_mod, "post", return_value=_fake_response(json_data=self._hits([]))
        ) as mock_post:
            osc.search_similar([0.1, 0.2], k=7)
            body = mock_post.call_args[1]["json"]
            assert body["size"] == 7
            assert body["query"]["knn"]["embedding"]["k"] == 7

    def test_raises_on_http_error(self):
        import requests as req_mod

        import backend.app.opensearch_client as osc

        with patch.object(
            req_mod, "post", return_value=_fake_response(raise_for_status=True)
        ):
            with pytest.raises(req_mod.HTTPError):
                osc.search_similar([0.1], k=5)
