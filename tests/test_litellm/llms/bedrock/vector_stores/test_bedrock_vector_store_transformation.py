from unittest.mock import MagicMock

from litellm.llms.bedrock.vector_stores.transformation import BedrockVectorStoreConfig


def test_transform_search_request():
    """
    Test that BedrockVectorStoreConfig correctly transforms search vector store requests.
    
    Verifies that the transformation creates the proper URL endpoint and request body
    with the expected retrievalQuery structure.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}

    url, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
    )

    assert url.endswith("/kb123/retrieve")
    assert body["retrievalQuery"].get("text") == "hello"


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