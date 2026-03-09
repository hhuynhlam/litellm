# Bedrock Vector Store Location URI Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix filename/file_id resolution for S3-backed (and other) Bedrock Knowledge Base results by reading the source URI from the `location` field when `x-amz-bedrock-kb-source-uri` is absent from metadata.

**Architecture:** Add a single new private helper `_get_uri_from_location()` that maps all Bedrock location types to their URI field. In `transform_search_vector_store_response`, extract `location` from each result item and, if `x-amz-bedrock-kb-source-uri` is not already in metadata, inject the resolved URI before calling the existing unchanged helpers.

**Tech Stack:** Python, httpx, existing LiteLLM types, pytest with unittest.mock

---

### Task 1: Add `_get_uri_from_location` helper and its tests (TDD)

**Files:**
- Modify: `litellm/llms/bedrock/vector_stores/transformation.py` (after line 298, before `_get_attributes_from_metadata`)
- Test: `tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py`

---

**Step 1: Write the failing tests**

Append to `tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py`:

```python
def test_get_uri_from_location_s3():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "S3",
        "s3Location": {"uri": "s3://my-bucket/docs/file.pdf"},
    }
    assert config._get_uri_from_location(location) == "s3://my-bucket/docs/file.pdf"


def test_get_uri_from_location_web():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "WEB",
        "webLocation": {"url": "https://example.com/page"},
    }
    assert config._get_uri_from_location(location) == "https://example.com/page"


def test_get_uri_from_location_confluence():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "CONFLUENCE",
        "confluenceLocation": {"url": "https://myorg.atlassian.net/wiki/spaces/PROJ/pages/123"},
    }
    assert config._get_uri_from_location(location) == "https://myorg.atlassian.net/wiki/spaces/PROJ/pages/123"


def test_get_uri_from_location_unknown_returns_none():
    config = BedrockVectorStoreConfig()
    assert config._get_uri_from_location({}) is None
    assert config._get_uri_from_location({"type": "UNKNOWN"}) is None
    assert config._get_uri_from_location({"type": "S3"}) is None  # missing s3Location
```

**Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_s3 tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_web tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_unknown_returns_none -v
```

Expected: FAIL with `AttributeError: 'BedrockVectorStoreConfig' object has no attribute '_get_uri_from_location'`

**Step 3: Implement `_get_uri_from_location`**

In `litellm/llms/bedrock/vector_stores/transformation.py`, add this method to `BedrockVectorStoreConfig` after `_get_filename_from_metadata` (after line 298) and before `_get_attributes_from_metadata`:

```python
def _get_uri_from_location(self, location: Dict[str, Any]) -> Optional[str]:
    """
    Extract source URI from Bedrock KB location field.

    Supports all location types from the Bedrock Retrieve API:
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html
    """
    if not location:
        return None
    location_type = location.get("type", "").upper()
    type_map = {
        "S3": ("s3Location", "uri"),
        "CONFLUENCE": ("confluenceLocation", "url"),
        "KENDRA": ("kendraDocumentLocation", "uri"),
        "SALESFORCE": ("salesforceLocation", "url"),
        "SHAREPOINT": ("sharePointLocation", "url"),
        "WEB": ("webLocation", "url"),
        "CUSTOM": ("customDocumentLocation", "id"),
    }
    entry = type_map.get(location_type)
    if not entry:
        return None
    loc_key, uri_key = entry
    loc_data = location.get(loc_key) or {}
    return loc_data.get(uri_key) or None
```

**Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_s3 tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_web tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_confluence tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_get_uri_from_location_unknown_returns_none -v
```

Expected: all 4 PASS

**Step 5: Commit**

```bash
git add litellm/llms/bedrock/vector_stores/transformation.py \
        tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py
git commit -m "feat(bedrock): add _get_uri_from_location helper for all KB location types"
```

---

### Task 2: Wire location URI into response transformer and add integration-level tests

**Files:**
- Modify: `litellm/llms/bedrock/vector_stores/transformation.py:315-325` (`transform_search_vector_store_response`)
- Test: `tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py`

---

**Step 1: Write the failing tests**

Append to the test file:

```python
def test_transform_response_uses_location_uri():
    """
    When x-amz-bedrock-kb-source-uri is absent from metadata, the URI should
    be resolved from location.s3Location.uri and used for filename/file_id.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {"query": "test query"}

    raw_response = {
        "retrievalResults": [
            {
                "content": {"text": "some content", "type": "TEXT"},
                "location": {
                    "s3Location": {"uri": "s3://renewhealth-bedrock-kb-test/SKILL.md"},
                    "type": "S3",
                },
                "metadata": {
                    "x-amz-bedrock-kb-source-file-modality": "TEXT",
                    "x-amz-bedrock-kb-chunk-id": "8befd6d7-d8d1-49f6-b01d-54cf4e77c01a",
                    "x-amz-bedrock-kb-data-source-id": "RFK56KSEMO",
                },
                "score": 0.506902021031646,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = raw_response
    mock_http_response.status_code = 200

    result = config.transform_search_vector_store_response(mock_http_response, mock_log)

    assert len(result["data"]) == 1
    item = result["data"][0]
    assert item["file_id"] == "s3://renewhealth-bedrock-kb-test/SKILL.md"
    assert item["filename"] == "SKILL.md"


def test_transform_response_metadata_uri_takes_precedence():
    """
    When x-amz-bedrock-kb-source-uri is already in metadata, it must be used
    and the location field must be ignored.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {"query": "test query"}

    raw_response = {
        "retrievalResults": [
            {
                "content": {"text": "some content", "type": "TEXT"},
                "location": {
                    "s3Location": {"uri": "s3://bucket/location-path/file.pdf"},
                    "type": "S3",
                },
                "metadata": {
                    "x-amz-bedrock-kb-source-uri": "s3://bucket/metadata-path/other.pdf",
                },
                "score": 0.9,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = raw_response
    mock_http_response.status_code = 200

    result = config.transform_search_vector_store_response(mock_http_response, mock_log)

    item = result["data"][0]
    assert item["file_id"] == "s3://bucket/metadata-path/other.pdf"
    assert item["filename"] == "other.pdf"
```

**Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_transform_response_uses_location_uri tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py::test_transform_response_metadata_uri_takes_precedence -v
```

Expected: `test_transform_response_uses_location_uri` FAIL (filename resolves to `bedrock-kb-document-RFK56KSEMO`), `test_transform_response_metadata_uri_takes_precedence` PASS (this already works since metadata URI was always used).

**Step 3: Update `transform_search_vector_store_response`**

In `litellm/llms/bedrock/vector_stores/transformation.py`, find the block inside `transform_search_vector_store_response` that currently reads (around lines 321–325):

```python
                # Extract metadata and use helper functions
                metadata = item.get("metadata", {}) or {}
                file_id = self._get_file_id_from_metadata(metadata)
                filename = self._get_filename_from_metadata(metadata)
                attributes = self._get_attributes_from_metadata(metadata)
```

Replace it with:

```python
                # Extract metadata and use helper functions
                metadata = item.get("metadata", {}) or {}
                # Resolve source URI from location field if not present in metadata
                if not metadata.get("x-amz-bedrock-kb-source-uri"):
                    location = item.get("location", {}) or {}
                    source_uri = self._get_uri_from_location(location)
                    if source_uri:
                        metadata = {**metadata, "x-amz-bedrock-kb-source-uri": source_uri}
                file_id = self._get_file_id_from_metadata(metadata)
                filename = self._get_filename_from_metadata(metadata)
                attributes = self._get_attributes_from_metadata(metadata)
```

**Step 4: Run all tests to verify they pass**

```bash
poetry run pytest tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py -v
```

Expected: all tests PASS including the pre-existing `test_transform_search_request`.

**Step 5: Run the full unit test suite to check for regressions**

```bash
make test-unit
```

Expected: no regressions.

**Step 6: Commit**

```bash
git add litellm/llms/bedrock/vector_stores/transformation.py \
        tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py
git commit -m "fix(bedrock): resolve filename/file_id from location field when metadata URI absent"
```
