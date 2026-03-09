# Bedrock Vector Store: Resolve Source URI from `location` Field

## Context

When LiteLLM proxies a search request to an AWS Bedrock Knowledge Base and transforms the response back to the OpenAI-compatible format, it needs to resolve a `filename` and `file_id` for each result.

The existing helpers (`_get_filename_from_metadata`, `_get_file_id_from_metadata`) look for `x-amz-bedrock-kb-source-uri` inside the `metadata` field. However, for S3-backed (and other) Knowledge Bases, the source URI is not present in `metadata` at all — it lives in the top-level `location` field of each `RetrievalResult`. This causes `filename` and `file_id` to fall back to chunk/data-source IDs instead of the actual file path.

**Example Bedrock response item:**
```json
{
  "content": { "text": "...", "type": "TEXT" },
  "location": {
    "s3Location": { "uri": "s3://renewhealth-bedrock-kb-test/SKILL.md" },
    "type": "S3"
  },
  "metadata": {
    "x-amz-bedrock-kb-source-file-modality": "TEXT",
    "x-amz-bedrock-kb-chunk-id": "8befd6d7-d8d1-49f6-b01d-54cf4e77c01a",
    "x-amz-bedrock-kb-data-source-id": "RFK56KSEMO"
  },
  "score": 0.506902021031646
}
```

Expected: `filename = "SKILL.md"`, `file_id = "s3://renewhealth-bedrock-kb-test/SKILL.md"`
Actual (before fix): `filename = "bedrock-kb-document-RFK56KSEMO"`, `file_id = "bedrock-kb-8befd6d7-..."`

## Design

### Files Modified
- `litellm/llms/bedrock/vector_stores/transformation.py`
- `tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py`

### Change 1: New helper `_get_uri_from_location(location)`

Add a private method to `BedrockVectorStoreConfig` that extracts the source URI from the `location` field, handling all location types supported by the Bedrock Retrieve API:

| `location.type` | Sub-key | URI field |
|---|---|---|
| `S3` | `s3Location` | `uri` |
| `CONFLUENCE` | `confluenceLocation` | `url` |
| `KENDRA` | `kendraDocumentLocation` | `uri` |
| `SALESFORCE` | `salesforceLocation` | `url` |
| `SHAREPOINT` | `sharePointLocation` | `url` |
| `WEB` | `webLocation` | `url` |
| `CUSTOM` | `customDocumentLocation` | `id` |

Returns `Optional[str]`. Returns `None` for unknown/missing types.

### Change 2: Inject location URI into metadata in `transform_search_vector_store_response`

**Priority:** `x-amz-bedrock-kb-source-uri` from metadata is used if already set. Only if it is absent do we fall back to resolving the URI from `location`.

In the response transformer loop, after extracting `metadata` from each item, also extract `location`. If `x-amz-bedrock-kb-source-uri` is not present in metadata, call `_get_uri_from_location()` and, if a URI is found, inject it into a copy of metadata before passing to the existing helpers:

```python
location = item.get("location", {}) or {}
if not metadata.get("x-amz-bedrock-kb-source-uri"):
    source_uri = self._get_uri_from_location(location)
    if source_uri:
        metadata = {**metadata, "x-amz-bedrock-kb-source-uri": source_uri}
```

The existing `_get_file_id_from_metadata` and `_get_filename_from_metadata` helpers are **unchanged** — they already parse `x-amz-bedrock-kb-source-uri` correctly once it is present.

### Tests to Add

In `tests/test_litellm/llms/bedrock/vector_stores/test_bedrock_vector_store_transformation.py`:

1. `test_get_uri_from_location_s3` — S3 type returns correct URI
2. `test_get_uri_from_location_web` — web type returns correct URL
3. `test_get_uri_from_location_unknown` — unknown/missing type returns `None`
4. `test_transform_response_uses_location_uri` — full response transform with real example payload produces `filename="SKILL.md"` and `file_id="s3://renewhealth-bedrock-kb-test/SKILL.md"`
5. `test_transform_response_metadata_uri_takes_precedence` — if `x-amz-bedrock-kb-source-uri` is already set in metadata, it is used and `location` is ignored
