# Crop Advisory AI Platform REST Gateway: API Reference

Version 1 | Implementation: `api/main.py` | Stack: FastAPI, Uvicorn, Pydantic, Starlette

All farm names, readings, identifiers and source titles in the examples are illustrative sample data. Every example is internally consistent: the same soil and climate reading (a Kharif delta paddy field) is used in every endpoint so that responses can be compared across calls.

## Section 1: API Overview and Getting Started

### 1.1 Service Summary

| Property                   | Value                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Service name               | Crop Advisory AI Platform REST Gateway                                                                                                                                                                                                                                                                                                                              |
| Description                | High-performance REST API exposing calibrated machine learning crop recommendations, SHAP What-If environmental factor sensitivity analysis, computer vision leaf pathology diagnosis, hybrid RAG knowledge retrieval over official agricultural extension guidelines, natural language SQLite farm history analytics, and publication-grade PDF dossier generation |
| Implementation file        | `api/main.py`                                                                                                                                                                                                                                                                                                                                                     |
| Protocols                  | HTTP/1.1 and HTTP/2, REST                                                                                                                                                                                                                                                                                                                                           |
| Content types              | `application/json` (requests and responses), `multipart/form-data` (leaf image upload), `application/pdf` (document streaming)                                                                                                                                                                                                                                |
| Local development base URL | `http://localhost:8000`                                                                                                                                                                                                                                                                                                                                           |
| Production base URL        | `https://api.cropadvisory.agritech.org`                                                                                                                                                                                                                                                                                                                           |
| Swagger UI                 | `http://localhost:8000/docs`                                                                                                                                                                                                                                                                                                                                      |
| ReDoc                      | `http://localhost:8000/redoc`                                                                                                                                                                                                                                                                                                                                     |
| Raw OpenAPI JSON           | `http://localhost:8000/openapi.json`                                                                                                                                                                                                                                                                                                                              |
| CORS                       | Fully enabled:`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`                                                                                                                                                                                                                                                                             |
| Performance model          | In-memory lazy singletons for the ML Engine, Retriever, SQL Agent, Plant Doctor and Multi-Agent Orchestrator. After the first call that loads an engine, prediction endpoints respond in under 50 ms                                                                                                                                                                |

The first request that touches an engine pays that engine's load cost (model deserialization, vector index load, or database connection). Subsequent requests reuse the singleton. Call `GET /health` after deployment to see which engines are already warm.

### 1.2 Operational Status

Use `GET /health` for liveness probes and load balancer checks. A response of `"status": "healthy"` with `"uptime_status": "operational"` means the gateway is accepting traffic.

### 1.3 Authentication

Every endpoint except `GET /health` expects an API key sent as a bearer token.

```http
Authorization: Bearer <API_KEY>
```

Keep the key in an environment variable or a secrets manager. Never embed it in a mobile application binary or commit it to source control. In local development the gateway runs on `http://localhost:8000`; send the same header so that client code is identical across environments.

### 1.4 Rate Limiting

| Endpoint group                                                                                                                                                            | Limit                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Standard endpoints:`/health`, `/api/v1/predict`, `/api/v1/recommend`, `/api/v1/simulate`, `/api/v1/history`, `/api/v1/history/stats`, `/api/v1/history/add` | 60 requests per minute |
| Vision and PDF export:`/api/v1/diagnose`, `/api/v1/export_pdf`                                                                                                        | 20 requests per minute |

When a limit is exceeded the gateway returns `429 Too Many Requests` using the standard error envelope with `error_code` `RATE_LIMIT_EXCEEDED` and a `Retry-After` header expressed in seconds. Clients should back off for the indicated duration before retrying.

### 1.5 Standard HTTP Status Codes

| Code                          | Meaning                     | When it is returned                                                                                                                                                                                       |
| ----------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `200 OK`                    | Success                     | The request was processed and the response body contains the result                                                                                                                                       |
| `400 Bad Request`           | Semantic validation failure | The JSON is well formed and typed correctly, but a value breaks a business rule: out-of-range feature, unsupported crop, unsupported parameter name, bad file type, oversized file, unsafe or empty query |
| `422 Unprocessable Entity`  | Schema validation failure   | A required field is missing, a field has the wrong JSON type, or a multipart part is absent                                                                                                               |
| `500 Internal Server Error` | Server fault                | An engine, the database or the PDF renderer failed unexpectedly                                                                                                                                           |
| `401 Unauthorized`          | Authentication failure      | The`Authorization` header is missing or the key is invalid (error\_code `UNAUTHORIZED`)                                                                                                               |
| `429 Too Many Requests`     | Rate limit exceeded         | See section 1.4                                                                                                                                                                                           |

### 1.6 Standard Error Envelope

Every failure, regardless of status code, uses this envelope:

```json
{
  "status": "error",
  "error_code": "INVALID_FEATURE_BOUNDS",
  "message": "pH value 15.2 is outside valid agronomic bounds [3.5, 10.0]",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

| Field          | Type   | Description                                                                          |
| -------------- | ------ | ------------------------------------------------------------------------------------ |
| `status`     | string | Always`"error"`                                                                    |
| `error_code` | string | Stable machine-readable code. Branch client logic on this value, never on`message` |
| `message`    | string | Human-readable explanation that names the offending field and value                  |
| `timestamp`  | string | ISO 8601 UTC time at which the error was produced                                    |

The error codes used across this reference are:

| `error_code`                 | HTTP status | Meaning                                                                                        |
| ------------------------------ | ----------- | ---------------------------------------------------------------------------------------------- |
| `INVALID_FEATURE_BOUNDS`     | 400         | A`FeaturesInput` value is outside its valid range                                            |
| `INVALID_SEASON_ENCODING`    | 400         | `season_Kharif` and `season_Rabi` are both 1                                               |
| `UNSUPPORTED_CROP`           | 400         | `crop_preference` or `crop_planted` is not one of the 25 supported crops                   |
| `UNSUPPORTED_PARAMETER`      | 400         | `parameter` is not one of `N`, `P`, `K`, `pH`, `total_rainfall_mm`, `avg_temp_c` |
| `INVALID_FILE_TYPE`          | 400         | Uploaded file is not JPG, JPEG or PNG                                                          |
| `FILE_TOO_LARGE`             | 400         | Uploaded file is larger than 10 MB                                                             |
| `EMPTY_QUERY`                | 400         | The`query` string is empty or whitespace                                                     |
| `UNSAFE_QUERY_REJECTED`      | 400         | The generated SQL was not a single read-only`SELECT` statement                               |
| `INVALID_SEASON`             | 400         | `season` is not `Kharif`, `Rabi` or `Whole Year`                                       |
| `VALIDATION_ERROR`           | 422         | Pydantic schema validation failed                                                              |
| `UNAUTHORIZED`               | 401         | Missing or invalid API key                                                                     |
| `RATE_LIMIT_EXCEEDED`        | 429         | Too many requests                                                                              |
| `MODEL_INFERENCE_FAILURE`    | 500         | The ML engine raised an exception                                                              |
| `PIPELINE_EXECUTION_FAILURE` | 500         | The multi-agent orchestrator failed in one of its stages                                       |
| `VISION_INFERENCE_FAILURE`   | 500         | The plant doctor model failed on the image                                                     |
| `DATABASE_ERROR`             | 500         | SQLite read or write failed                                                                    |
| `PDF_RENDER_FAILURE`         | 500         | ReportLab could not build the document                                                         |
| `INTERNAL_SERVER_ERROR`      | 500         | Any other unhandled fault                                                                      |

### 1.7 Conventions

- Timestamps are ISO 8601 in UTC.
- `confidence` fields are percentages from 0 to 100. `class_probabilities` values and `probability_deltas` values are fractions from 0 to 1.
- Yield is in tonnes per hectare (t/ha). Nutrients are in kg/ha. Rainfall is in millimetres. Temperature is in degrees Celsius. Money is in Indian rupees (INR).
- Both season flags set to 0 denotes neither Kharif nor Rabi (for example a whole-year or perennial crop). Both flags set to 1 is rejected.

### 1.8 The 25 Supported Crops

Rice, Sugarcane, Banana, Groundnut, Cotton, Maize, Ragi, Sorghum, Pearl Millet, Black Gram, Green Gram, Red Gram, Coconut, Turmeric, Tomato, Onion, Chilli, Brinjal, Okra, Tapioca, Sesame, Sunflower, Coriander, Mango, Cowpea.

These names are the keys of `class_probabilities` and the accepted values for `crop_preference` and `crop_planted`.

## Section 2: Comprehensive Endpoint Specifications

Shared variables used by every code snippet:

```bash
export BASE_URL="http://localhost:8000"
export API_KEY="your_api_key_here"
```

### 2.1 GET /health: Service Health and Diagnostics

**Method badge:** `GET` | **Path:** `/health` | **Rate limit:** 60 requests per minute | **Authentication:** not required

**Description.** Reports whether the gateway process is alive and which lazily created engine singletons are already loaded in memory. Load balancers, Kubernetes probes and IoT field stations should call this endpoint before submitting work. A field station on a weak rural connection can use it as a cheap connectivity check, and operations teams can use the `engines` block to confirm that a deployment has been warmed up.

**Request headers**

| Header            | Required | Value                | Description                                 |
| ----------------- | -------- | -------------------- | ------------------------------------------- |
| `Accept`        | No       | `application/json` | Response media type                         |
| `Authorization` | No       | `Bearer <API_KEY>` | Accepted but not required for this endpoint |
| `Content-Type`  | No       | not applicable       | There is no request body                    |

**Request parameters**

| Field  | Type           | Required       | Default        | Valid Range    | Description                                                                                |
| ------ | -------------- | -------------- | -------------- | -------------- | ------------------------------------------------------------------------------------------ |
| (none) | not applicable | not applicable | not applicable | not applicable | The endpoint takes no path, query or body parameters. Unknown query parameters are ignored |

**Example request**

```http
GET /health HTTP/1.1
Host: localhost:8000
Accept: application/json
```

**Success response (200 OK)**

```json
{
  "status": "healthy",
  "service": "Crop Advisory REST Gateway",
  "uptime_status": "operational",
  "engines": {
    "ml_engine": "loaded",
    "retriever": "loaded",
    "sql_agent": "loaded",
    "plant_doctor": "not_loaded",
    "orchestrator": "loaded"
  }
}
```

**Response schema**

| Field                    | Type   | Description                                                                           |
| ------------------------ | ------ | ------------------------------------------------------------------------------------- |
| `status`               | string | `"healthy"` when the gateway can serve requests                                     |
| `service`              | string | Constant`"Crop Advisory REST Gateway"`                                              |
| `uptime_status`        | string | `"operational"` when all routes are registered and accepting traffic                |
| `engines`              | object | Load state of each lazy singleton                                                     |
| `engines.ml_engine`    | string | `"loaded"` or `"not_loaded"`: calibrated ExtraTrees classifier and SHAP explainer |
| `engines.retriever`    | string | `"loaded"` or `"not_loaded"`: hybrid RAG retriever                                |
| `engines.sql_agent`    | string | `"loaded"` or `"not_loaded"`: natural language SQLite agent                       |
| `engines.plant_doctor` | string | `"loaded"` or `"not_loaded"`: MobileNetV2 and ResNet50 vision models              |
| `engines.orchestrator` | string | `"loaded"` or `"not_loaded"`: multi-agent orchestrator                            |

A value of `not_loaded` is normal after a restart. The engine is created on the first request that needs it.

**Error responses**

This endpoint accepts no parameters or body, so it cannot return `400 Bad Request` or `422 Unprocessable Entity`. A `500` is the only failure.

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "INTERNAL_SERVER_ERROR",
  "message": "Engine state inspection failed unexpectedly",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X GET "$BASE_URL/health" \
  -H "Accept: application/json"
```

**Python (requests)**

```python
import os
from typing import Any, Dict

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")


def check_health() -> Dict[str, Any]:
    """Return the gateway health document or raise on any non-200 response."""
    response: requests.Response = requests.get(
        f"{BASE_URL}/health",
        headers={"Accept": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    health: Dict[str, Any] = check_health()
    print(health["status"], health["uptime_status"])
```

### 2.2 POST /api/v1/predict: Fast ML Crop Suitability Prediction

**Method badge:** `POST` | **Path:** `/api/v1/predict` | **Rate limit:** 60 requests per minute

**Description.** Runs soil nutrient levels (N, P, K), soil pH and seasonal climate readings through the calibrated ExtraTrees classifier and returns the best-suited crop, a calibrated confidence, an expected yield band and the probability of all 25 supported crops. Use this endpoint when a client needs only the machine learning verdict, for example a field-station dashboard that refreshes after every soil test. It skips retrieval, SHAP attribution and document generation, which is why it is the fastest prediction route. Use `/api/v1/recommend` when the farmer needs a full advisory.

**Request headers**

| Header            | Required | Value                | Description           |
| ----------------- | -------- | -------------------- | --------------------- |
| `Content-Type`  | Yes      | `application/json` | Body is a JSON object |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key               |
| `Accept`        | No       | `application/json` | Response media type   |

**Request body (`FeaturesInput`)**

| Field                    | Type           | Required | Default | Valid Range                         | Description                                                                                         |
| ------------------------ | -------------- | -------- | ------- | ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| `N`                    | number (float) | Yes      | none    | 0.0 to 300.0 (typical 40 to 160)    | Soil nitrogen in kg/ha                                                                              |
| `P`                    | number (float) | Yes      | none    | 0.0 to 150.0 (typical 15 to 60)     | Soil phosphorus in kg/ha                                                                            |
| `K`                    | number (float) | Yes      | none    | 0.0 to 200.0 (typical 20 to 80)     | Soil potassium in kg/ha                                                                             |
| `pH`                   | number (float) | Yes      | none    | 3.5 to 10.0 (typical 6.0 to 8.2)    | Soil pH                                                                                             |
| `avg_temp_c`           | number (float) | Yes      | none    | 10.0 to 50.0 (typical 25 to 35)     | Average temperature in degrees Celsius over the growing season                                      |
| `total_rainfall_mm`    | number (float) | Yes      | none    | 0.0 to 3000.0 (typical 500 to 1400) | Total rainfall in millimetres over the growing season                                               |
| `avg_humidity_percent` | number (float) | Yes      | none    | 0.0 to 100.0 (typical 50 to 90)     | Average relative humidity in percent                                                                |
| `season_Kharif`        | integer        | No       | `1`   | 0 or 1                              | One-hot season flag: 1 when the crop is grown in Kharif                                             |
| `season_Rabi`          | integer        | No       | `0`   | 0 or 1                              | One-hot season flag: 1 when the crop is grown in Rabi. Must not be 1 together with`season_Kharif` |

**Example request**

```json
{
  "N": 98.0,
  "P": 42.0,
  "K": 51.0,
  "pH": 6.8,
  "avg_temp_c": 29.5,
  "total_rainfall_mm": 1200.0,
  "avg_humidity_percent": 78.0,
  "season_Kharif": 1,
  "season_Rabi": 0
}
```

**Success response (200 OK)**

```json
{
  "top_crop": "Rice",
  "confidence": 87.4,
  "expected_yield_range_tha": {
    "min": 4.8,
    "max": 6.2
  },
  "class_probabilities": {
    "Rice": 0.874,
    "Sugarcane": 0.038,
    "Banana": 0.019,
    "Maize": 0.011,
    "Cotton": 0.008,
    "Groundnut": 0.007,
    "Turmeric": 0.006,
    "Tapioca": 0.005,
    "Sorghum": 0.004,
    "Ragi": 0.004,
    "Pearl Millet": 0.003,
    "Black Gram": 0.003,
    "Green Gram": 0.003,
    "Red Gram": 0.002,
    "Coconut": 0.002,
    "Tomato": 0.002,
    "Onion": 0.001,
    "Chilli": 0.001,
    "Brinjal": 0.001,
    "Okra": 0.001,
    "Sesame": 0.001,
    "Sunflower": 0.001,
    "Coriander": 0.001,
    "Mango": 0.001,
    "Cowpea": 0.001
  }
}
```

**Response schema**

| Field                            | Type   | Description                                                                                           |
| -------------------------------- | ------ | ----------------------------------------------------------------------------------------------------- |
| `top_crop`                     | string | Crop with the highest calibrated probability. One of the 25 supported crops                           |
| `confidence`                   | number | Probability of`top_crop` expressed as a percentage (87.4 means 0.874)                               |
| `expected_yield_range_tha`     | object | Expected yield band for`top_crop` in t/ha                                                           |
| `expected_yield_range_tha.min` | number | Lower bound of the band                                                                               |
| `expected_yield_range_tha.max` | number | Upper bound of the band                                                                               |
| `class_probabilities`          | object | Map from each of the 25 crop names to a calibrated probability between 0 and 1. The values sum to 1.0 |

**Error responses**

`400 Bad Request`: value outside the valid range

```json
{
  "status": "error",
  "error_code": "INVALID_FEATURE_BOUNDS",
  "message": "pH value 15.2 is outside valid agronomic bounds [3.5, 10.0]",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`400 Bad Request`: both season flags set

```json
{
  "status": "error",
  "error_code": "INVALID_SEASON_ENCODING",
  "message": "season_Kharif and season_Rabi cannot both be 1",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: missing required field

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'N' is required (body.N)",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "MODEL_INFERENCE_FAILURE",
  "message": "Crop classifier failed to produce probabilities",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -d '{
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0
  }'
```

**Python (requests)**

```python
import os
from typing import Any, Dict

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

FEATURES: Dict[str, Any] = {
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0,
}

response: requests.Response = requests.post(
    f"{BASE_URL}/api/v1/predict",
    headers=HEADERS,
    json=FEATURES,
    timeout=30,
)

if response.status_code == 200:
    result: Dict[str, Any] = response.json()
    yield_band: Dict[str, float] = result["expected_yield_range_tha"]
    print(f"Top crop: {result['top_crop']} ({result['confidence']}%)")
    print(f"Expected yield: {yield_band['min']} to {yield_band['max']} t/ha")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.3 POST /api/v1/recommend: End-to-End Multi-Agent Advisory Dossier

**Method badge:** `POST` | **Path:** `/api/v1/recommend` | **Rate limit:** 60 requests per minute

**Description.** Runs the complete multi-agent advisory pipeline for one field. The orchestrator combines the calibrated ML crop prediction, SHAP feature attributions that explain why the crop was chosen, hybrid RAG chunks retrieved from official research institute extension guidelines, companion planting suggestions, a dynamic stage-by-stage cultivation timetable with dosages, and natural farming alternatives to chemical inputs. Use this endpoint to power the main advisory screen of a mobile application. Because it executes several engines, expect a longer response time than `/api/v1/predict`, particularly on the first call after a restart when the retriever and orchestrator are loaded.

**Request headers**

| Header            | Required | Value                | Description           |
| ----------------- | -------- | -------------------- | --------------------- |
| `Content-Type`  | Yes      | `application/json` | Body is a JSON object |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key               |
| `Accept`        | No       | `application/json` | Response media type   |

**Request body (`AdvisoryRequest`)**

| Field               | Type                       | Required | Default                       | Valid Range                                                      | Description                                                                                                                                                           |
| ------------------- | -------------------------- | -------- | ----------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `features`        | object (`FeaturesInput`) | Yes      | none                          | Each sub-field must satisfy the ranges in the endpoint 2.2 table | Soil and climate reading. Sub-fields:`N`, `P`, `K`, `pH`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`, `season_Kharif`, `season_Rabi` |
| `farmer_name`     | string                     | No       | `"Ramesh (Farmer)"`         | 1 to 100 characters                                              | Name printed in the advisory and in exported PDFs                                                                                                                     |
| `location`        | string                     | No       | `"Delta Agricultural Zone"` | 1 to 150 characters                                              | Agro-climatic zone or village used to ground the retrieval and the narrative                                                                                          |
| `farmer_context`  | string or null             | No       | `null`                      | Up to 1000 characters                                            | Free-text situation such as irrigation source, land holding or constraints. Used to tailor the advice                                                                 |
| `crop_preference` | string or null             | No       | `null`                      | One of the 25 supported crops in section 1.8                     | Crop the farmer would like to grow. Shapes the narrative and companion planting but never overrides the ML ranking                                                    |

**Example request**

```json
{
  "features": {
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0
  },
  "farmer_name": "Murugan",
  "location": "Thanjavur Delta Zone",
  "farmer_context": "2 hectare canal-irrigated plot, labour available for transplanting",
  "crop_preference": "Rice"
}
```

**Success response (200 OK)**

```json
{
  "status": "success",
  "recommendation": {
    "crop": "Rice",
    "summary": "Rice is strongly suited to this canal-irrigated Kharif plot. Rainfall, temperature and nitrogen are the main drivers. Soil pH of 6.8 is within the ideal band for transplanted paddy.",
    "companion_planting": [
      {
        "crop": "Azolla",
        "benefit": "Floating fern that fixes atmospheric nitrogen and suppresses weeds in the standing water"
      },
      {
        "crop": "Cowpea",
        "benefit": "Grown on field bunds to host natural enemies of rice pests and to add a protein crop"
      },
      {
        "crop": "Marigold",
        "benefit": "Bund border plant that attracts predators of stem borer and acts as a trap crop"
      }
    ]
  },
  "confidence": 87.4,
  "shap_importance": {
    "total_rainfall_mm": 0.268,
    "avg_temp_c": 0.197,
    "N": 0.151,
    "avg_humidity_percent": 0.121,
    "K": 0.080,
    "pH": 0.065,
    "P": 0.052,
    "season_Kharif": 0.058,
    "season_Rabi": 0.008
  },
  "rag_context_sources": [
    {
      "document": "Rice Production Technology Guide",
      "publisher": "ICAR",
      "page": 14,
      "relevance_score": 0.912
    },
    {
      "document": "Integrated Nutrient Management for Cereals",
      "publisher": "ICAR",
      "page": 31,
      "relevance_score": 0.874
    },
    {
      "document": "Natural Farming Practices for Paddy",
      "publisher": "MANAGE",
      "page": 9,
      "relevance_score": 0.841
    }
  ],
  "cultivation_timetable": [
    {
      "stage": "Land preparation and basal application",
      "days": "Day -10 to Day 0",
      "actions": "Puddle and level the field. Apply basal fertilizers and incorporate them into the puddled soil.",
      "dosage": "Urea 65 kg/ha, single super phosphate 250 kg/ha, muriate of potash 33 kg/ha, zinc sulphate 25 kg/ha"
    },
    {
      "stage": "Transplanting",
      "days": "Day 0",
      "actions": "Transplant 21 day old seedlings at 2 to 3 seedlings per hill with 20 cm by 10 cm spacing. Maintain 2 cm of standing water.",
      "dosage": "Seed rate 40 kg/ha for the nursery"
    },
    {
      "stage": "Early tillering top dressing",
      "days": "Day 18 to Day 21",
      "actions": "Apply first top dressing and hand weed or use a rotary weeder.",
      "dosage": "Urea 65 kg/ha"
    },
    {
      "stage": "Active tillering top dressing",
      "days": "Day 38 to Day 42",
      "actions": "Apply second top dressing together with potash and drain the field briefly to encourage tillering.",
      "dosage": "Urea 65 kg/ha, muriate of potash 33 kg/ha"
    },
    {
      "stage": "Panicle initiation",
      "days": "Day 58 to Day 62",
      "actions": "Apply final nitrogen split and keep 5 cm of standing water. Scout for blast and stem borer.",
      "dosage": "Urea 65 kg/ha"
    },
    {
      "stage": "Flowering and grain filling",
      "days": "Day 65 to Day 105",
      "actions": "Maintain 5 cm of water until 10 days before harvest. Monitor for sheath blight and brown plant hopper.",
      "dosage": "No fertilizer"
    },
    {
      "stage": "Harvest",
      "days": "Day 110 to Day 120",
      "actions": "Drain the field 10 days earlier and harvest when 80 percent of panicles turn straw colored.",
      "dosage": "Not applicable"
    }
  ],
  "organic_alternatives": [
    {
      "chemical_input": "Urea (basal nitrogen)",
      "organic_alternative": "Farmyard manure or vermicompost, with green manure of sesbania",
      "application_rate": "Vermicompost 5 t/ha or farmyard manure 12.5 t/ha at puddling"
    },
    {
      "chemical_input": "Single super phosphate",
      "organic_alternative": "Phosphobacteria biofertilizer",
      "application_rate": "2 kg/ha mixed with farmyard manure and broadcast"
    },
    {
      "chemical_input": "Urea (top dressing)",
      "organic_alternative": "Azospirillum biofertilizer with neem cake",
      "application_rate": "Azospirillum 2 kg/ha and neem cake 250 kg/ha"
    },
    {
      "chemical_input": "Fungicide for blast prevention",
      "organic_alternative": "Pseudomonas fluorescens seed treatment and foliar spray",
      "application_rate": "10 g per kg of seed, then 2.5 kg/ha as foliar spray at tillering"
    }
  ]
}
```

**Response schema**

| Field                                 | Type            | Description                                                                                                                                                       |
| ------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `status`                            | string          | `"success"` when the pipeline completed                                                                                                                         |
| `recommendation`                    | object          | The primary recommendation                                                                                                                                        |
| `recommendation.crop`               | string          | Recommended crop, equal to the ML`top_crop`                                                                                                                     |
| `recommendation.summary`            | string          | Plain-language rationale grounded in the SHAP drivers and retrieved guidance                                                                                      |
| `recommendation.companion_planting` | array of object | Companion or bund crops. Each item has`crop` (string) and `benefit` (string)                                                                                  |
| `confidence`                        | number          | Calibrated probability of the recommended crop as a percentage                                                                                                    |
| `shap_importance`                   | object          | Normalized absolute SHAP attribution of each of the 9 input features for the recommended crop. Values sum to 1.0                                                  |
| `rag_context_sources`               | array of object | Knowledge chunks used. Each item has`document` (string), `publisher` (string), `page` (integer) and `relevance_score` (number from 0 to 1)                |
| `cultivation_timetable`             | array of object | Ordered growth stages. Each item has`stage` (string), `days` (string, days relative to transplanting or sowing), `actions` (string) and `dosage` (string) |
| `organic_alternatives`              | array of object | Natural farming substitutes. Each item has`chemical_input` (string), `organic_alternative` (string) and `application_rate` (string)                         |

**Error responses**

`400 Bad Request`: unsupported crop preference

```json
{
  "status": "error",
  "error_code": "UNSUPPORTED_CROP",
  "message": "crop_preference 'Dragonfruit' is not one of the 25 supported crops",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: missing features object

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'features' is required (body.features)",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "PIPELINE_EXECUTION_FAILURE",
  "message": "Multi-agent orchestrator failed during stage 'rag_retrieval'",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/recommend" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -d '{
    "features": {
      "N": 98.0,
      "P": 42.0,
      "K": 51.0,
      "pH": 6.8,
      "avg_temp_c": 29.5,
      "total_rainfall_mm": 1200.0,
      "avg_humidity_percent": 78.0,
      "season_Kharif": 1,
      "season_Rabi": 0
    },
    "farmer_name": "Murugan",
    "location": "Thanjavur Delta Zone",
    "farmer_context": "2 hectare canal-irrigated plot, labour available for transplanting",
    "crop_preference": "Rice"
  }'
```

**Python (requests)**

```python
import os
from typing import Any, Dict, List

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

payload: Dict[str, Any] = {
    "features": {
        "N": 98.0,
        "P": 42.0,
        "K": 51.0,
        "pH": 6.8,
        "avg_temp_c": 29.5,
        "total_rainfall_mm": 1200.0,
        "avg_humidity_percent": 78.0,
        "season_Kharif": 1,
        "season_Rabi": 0,
    },
    "farmer_name": "Murugan",
    "location": "Thanjavur Delta Zone",
    "farmer_context": "2 hectare canal-irrigated plot, labour available for transplanting",
    "crop_preference": "Rice",
}

response: requests.Response = requests.post(
    f"{BASE_URL}/api/v1/recommend",
    headers=HEADERS,
    json=payload,
    timeout=120,
)

if response.status_code == 200:
    advisory: Dict[str, Any] = response.json()
    print(f"Recommended crop: {advisory['recommendation']['crop']} ({advisory['confidence']}%)")
    timetable: List[Dict[str, str]] = advisory["cultivation_timetable"]
    for stage in timetable:
        print(f"{stage['days']}: {stage['stage']} -> {stage['dosage']}")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.4 POST /api/v1/simulate: What-If Environmental Sensitivity Simulator

**Method badge:** `POST` | **Path:** `/api/v1/simulate` | **Rate limit:** 60 requests per minute

**Description.** Answers the question "what happens to my crop choice if one thing changes?" The endpoint scores the baseline reading, replaces exactly one parameter with `new_value`, scores the modified reading, and returns the difference in probability for every crop that moved noticeably. Typical uses are a drought contingency (rainfall falls by 30 percent, for example 1200 mm to 840 mm), a soil amendment plan (pH raised from 6.5 to 8.2 after liming or gypsum), or a fertilizer budget check (nitrogen reduced from 98 to 60 kg/ha). All other fields of `features` are held constant, so the result isolates the sensitivity to the single changed parameter.

**Request headers**

| Header            | Required | Value                | Description           |
| ----------------- | -------- | -------------------- | --------------------- |
| `Content-Type`  | Yes      | `application/json` | Body is a JSON object |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key               |
| `Accept`        | No       | `application/json` | Response media type   |

**Request body (`SimulateRequest`)**

| Field         | Type                       | Required | Default | Valid Range                                                                                                                                                                                                   | Description                                                                                                                                                                    |
| ------------- | -------------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `features`  | object (`FeaturesInput`) | Yes      | none    | Each sub-field must satisfy the ranges in the endpoint 2.2 table                                                                                                                                              | Baseline soil and climate reading. Sub-fields:`N`, `P`, `K`, `pH`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`, `season_Kharif`, `season_Rabi` |
| `parameter` | string                     | Yes      | none    | One of`N`, `P`, `K`, `pH`, `total_rainfall_mm`, `avg_temp_c` (case sensitive)                                                                                                                     | Name of the single parameter to alter                                                                                                                                          |
| `new_value` | number (float)             | Yes      | none    | Must fall within the valid range of the chosen`parameter`: `N` 0.0 to 300.0, `P` 0.0 to 150.0, `K` 0.0 to 200.0, `pH` 3.5 to 10.0, `total_rainfall_mm` 0.0 to 3000.0, `avg_temp_c` 10.0 to 50.0 | Simulated value that replaces the baseline value                                                                                                                               |

**Example request**: 30 percent rainfall deficit

```json
{
  "features": {
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0
  },
  "parameter": "total_rainfall_mm",
  "new_value": 840.0
}
```

**Success response (200 OK)**

```json
{
  "base_prediction": {
    "top_crop": "Rice",
    "confidence": 87.4
  },
  "simulated_prediction": {
    "top_crop": "Rice",
    "confidence": 61.2
  },
  "parameter_tested": "total_rainfall_mm",
  "old_value": 1200.0,
  "new_value": 840.0,
  "probability_deltas": {
    "Rice": -0.262,
    "Sorghum": 0.071,
    "Groundnut": 0.058,
    "Ragi": 0.047,
    "Pearl Millet": 0.039,
    "Cotton": 0.031,
    "Maize": 0.008,
    "Sugarcane": -0.010,
    "Banana": -0.006
  }
}
```

In this scenario Rice remains the top crop, but its probability falls from 0.874 to 0.612 and drought tolerant crops (sorghum, groundnut, ragi, pearl millet) gain ground. A client application can use this to warn the farmer that a 30 percent rainfall deficit puts the Rice recommendation at risk and to propose a contingency crop.

**Response schema**

| Field                               | Type   | Description                                                                                                                                                                                     |
| ----------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `base_prediction`                 | object | Prediction for the unmodified`features`                                                                                                                                                       |
| `base_prediction.top_crop`        | string | Most probable crop for the baseline                                                                                                                                                             |
| `base_prediction.confidence`      | number | Baseline probability of that crop as a percentage                                                                                                                                               |
| `simulated_prediction`            | object | Prediction after`parameter` is replaced by `new_value`                                                                                                                                      |
| `simulated_prediction.top_crop`   | string | Most probable crop for the simulated reading                                                                                                                                                    |
| `simulated_prediction.confidence` | number | Simulated probability of that crop as a percentage                                                                                                                                              |
| `parameter_tested`                | string | Echo of the request`parameter`                                                                                                                                                                |
| `old_value`                       | number | Baseline value of the tested parameter, taken from`features`                                                                                                                                  |
| `new_value`                       | number | Simulated value, echo of the request                                                                                                                                                            |
| `probability_deltas`              | object | Map from crop name to signed probability change (simulated minus base, as a fraction). Only crops whose absolute change is at least 0.005 are listed, so the listed values need not sum to zero |

**Error responses**

`400 Bad Request`: parameter not supported

```json
{
  "status": "error",
  "error_code": "UNSUPPORTED_PARAMETER",
  "message": "parameter 'rainfall' is not supported. Allowed values: N, P, K, pH, total_rainfall_mm, avg_temp_c",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`400 Bad Request`: simulated value out of range

```json
{
  "status": "error",
  "error_code": "INVALID_FEATURE_BOUNDS",
  "message": "total_rainfall_mm value 4200.0 is outside valid agronomic bounds [0.0, 3000.0]",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: wrong type

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'new_value' must be a valid number (body.new_value)",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "MODEL_INFERENCE_FAILURE",
  "message": "Crop classifier failed while scoring the simulated reading",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/simulate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -d '{
    "features": {
      "N": 98.0,
      "P": 42.0,
      "K": 51.0,
      "pH": 6.8,
      "avg_temp_c": 29.5,
      "total_rainfall_mm": 1200.0,
      "avg_humidity_percent": 78.0,
      "season_Kharif": 1,
      "season_Rabi": 0
    },
    "parameter": "total_rainfall_mm",
    "new_value": 840.0
  }'
```

**Python (requests)**

```python
import os
from typing import Any, Dict

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

baseline_features: Dict[str, Any] = {
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0,
}

# Drought contingency: rainfall drops by 30 percent
payload: Dict[str, Any] = {
    "features": baseline_features,
    "parameter": "total_rainfall_mm",
    "new_value": round(baseline_features["total_rainfall_mm"] * 0.70, 1),
}

response: requests.Response = requests.post(
    f"{BASE_URL}/api/v1/simulate",
    headers=HEADERS,
    json=payload,
    timeout=30,
)

if response.status_code == 200:
    result: Dict[str, Any] = response.json()
    print(f"Before: {result['base_prediction']['top_crop']} ({result['base_prediction']['confidence']}%)")
    print(f"After : {result['simulated_prediction']['top_crop']} ({result['simulated_prediction']['confidence']}%)")
    for crop, delta in sorted(result["probability_deltas"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {crop:<14}{delta:+.3f}")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.5 POST /api/v1/diagnose: Computer Vision Leaf Disease Pathology

**Method badge:** `POST` | **Path:** `/api/v1/diagnose` | **Rate limit:** 20 requests per minute

**Description.** Accepts a photograph of a single affected leaf and classifies the disease using the convolutional vision models (MobileNetV2 and ResNet50) behind the Plant Doctor engine. The response names the disease and the causal pathogen, reports a confidence, lists the visual symptoms that support the diagnosis, and gives organic and chemical treatment options along with preventive measures. For best accuracy the leaf should fill most of the frame, be photographed in natural daylight against a plain background, and be in focus. Images are resized internally, and a resolution of at least 224 by 224 pixels is recommended. Treatment lists are decision support; chemical products must be used according to their registered label and local extension officer advice.

**Request headers**

| Header            | Required | Value                                    | Description                                                                                           |
| ----------------- | -------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `Content-Type`  | Yes      | `multipart/form-data; boundary=<auto>` | Set automatically by cURL (`-F`) and by `requests` (`files=`). Do not set it manually in Python |
| `Authorization` | Yes      | `Bearer <API_KEY>`                     | API key                                                                                               |
| `Accept`        | No       | `application/json`                     | Response media type                                                                                   |

**Request parameters (multipart form fields)**

| Field    | Type             | Required | Default | Valid Range                     | Description                                                                 |
| -------- | ---------------- | -------- | ------- | ------------------------------- | --------------------------------------------------------------------------- |
| `file` | binary file part | Yes      | none    | JPG, JPEG or PNG, maximum 10 MB | Leaf photograph. Accepted content types are`image/jpeg` and `image/png` |

**Example request**

```http
POST /api/v1/diagnose HTTP/1.1
Host: localhost:8000
Authorization: Bearer <API_KEY>
Accept: application/json
Content-Type: multipart/form-data; boundary=----CropAdvisoryBoundary7MA4YWxk

------CropAdvisoryBoundary7MA4YWxk
Content-Disposition: form-data; name="file"; filename="rice_leaf_blast.jpg"
Content-Type: image/jpeg

<binary JPEG bytes>
------CropAdvisoryBoundary7MA4YWxk--
```

**Success response (200 OK)**

```json
{
  "disease_name": "Rice Blast",
  "pathogen_species": "Magnaporthe oryzae",
  "confidence": 94.2,
  "symptoms_identified": [
    "Spindle shaped lesions with gray centers and brown margins on the leaf blade",
    "Yellow halo surrounding individual lesions",
    "Neighboring lesions coalescing along the mid-leaf region"
  ],
  "organic_treatment": [
    "Spray Pseudomonas fluorescens at 2 g per litre of water, twice at a 15 day interval",
    "Spray 5 percent neem seed kernel extract at the tillering stage",
    "Remove and burn heavily infected leaves and plant debris"
  ],
  "chemical_treatment": [
    "Tricyclazole 75 WP at 0.6 g per litre of water (300 g/ha in 500 litres of spray fluid)",
    "Isoprothiolane 40 EC at 1.5 mL per litre of water"
  ],
  "preventive_measures": [
    "Plant blast tolerant varieties recommended for the zone",
    "Treat seed with Pseudomonas fluorescens at 10 g per kg of seed",
    "Avoid excess nitrogen and split nitrogen applications as per the timetable",
    "Maintain field hygiene by removing stubble and alternate hosts such as grass weeds"
  ]
}
```

**Response schema**

| Field                   | Type            | Description                                                         |
| ----------------------- | --------------- | ------------------------------------------------------------------- |
| `disease_name`        | string          | Common name of the diagnosed disease                                |
| `pathogen_species`    | string          | Scientific name of the causal organism                              |
| `confidence`          | number          | Model confidence as a percentage from 0 to 100                      |
| `symptoms_identified` | array of string | Visual symptoms recognized in the image                             |
| `organic_treatment`   | array of string | Biological and cultural control options with rates                  |
| `chemical_treatment`  | array of string | Registered fungicide, bactericide or insecticide options with rates |
| `preventive_measures` | array of string | Practices that reduce recurrence                                    |

**Error responses**

`400 Bad Request`: unsupported file type

```json
{
  "status": "error",
  "error_code": "INVALID_FILE_TYPE",
  "message": "Uploaded file 'leaf.gif' has content type image/gif; accepted types are image/jpeg and image/png",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`400 Bad Request`: file too large

```json
{
  "status": "error",
  "error_code": "FILE_TOO_LARGE",
  "message": "Uploaded file is 12.4 MB; maximum allowed size is 10 MB",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: file part missing

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'file' is required (multipart part 'file')",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "VISION_INFERENCE_FAILURE",
  "message": "Image could not be decoded by the plant doctor model",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/diagnose" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -F "file=@rice_leaf_blast.jpg;type=image/jpeg"
```

**Python (requests)**

```python
import os
from typing import Any, Dict

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

# Content-Type is intentionally omitted: requests builds the multipart boundary itself
HEADERS: Dict[str, str] = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

IMAGE_PATH: str = "rice_leaf_blast.jpg"

with open(IMAGE_PATH, "rb") as image_file:
    response: requests.Response = requests.post(
        f"{BASE_URL}/api/v1/diagnose",
        headers=HEADERS,
        files={"file": (os.path.basename(IMAGE_PATH), image_file, "image/jpeg")},
        timeout=60,
    )

if response.status_code == 200:
    diagnosis: Dict[str, Any] = response.json()
    print(f"{diagnosis['disease_name']} ({diagnosis['pathogen_species']}), confidence {diagnosis['confidence']}%")
    for step in diagnosis["organic_treatment"]:
        print(f"  organic: {step}")
    for step in diagnosis["chemical_treatment"]:
        print(f"  chemical: {step}")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.6 POST /api/v1/history: Natural Language SQLite Farm History Query

**Method badge:** `POST` | **Path:** `/api/v1/history` | **Rate limit:** 60 requests per minute

**Description.** Lets a farmer or advisor ask questions about past seasons in plain language, for example "Which crop gave me the highest profit in Kharif?" The SQL Agent translates the question into one SQL statement against the `crop_history` table in `farmer_profile.db`, validates that it is a single read-only `SELECT`, executes it on a read-only connection, and returns the statement, the rows and a concise natural language summary. Results are capped at 100 rows. Statements that write, alter, drop or chain multiple commands are rejected before execution, so this endpoint cannot modify farm records. Use `/api/v1/history/add` to write records.

The queryable `crop_history` columns are `id`, `season`, `crop_planted`, `yield_achieved` (t/ha), `nitrogen_used` (kg/ha) and `profit` (INR).

**Request headers**

| Header            | Required | Value                | Description           |
| ----------------- | -------- | -------------------- | --------------------- |
| `Content-Type`  | Yes      | `application/json` | Body is a JSON object |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key               |
| `Accept`        | No       | `application/json` | Response media type   |

**Request body (`HistoryQuery`)**

| Field     | Type   | Required | Default | Valid Range                    | Description                                                                  |
| --------- | ------ | -------- | ------- | ------------------------------ | ---------------------------------------------------------------------------- |
| `query` | string | Yes      | none    | 3 to 500 characters, not blank | Natural language question about past crops, yields, fertilizer use or profit |

**Example request**

```json
{
  "query": "Which crop gave me the highest profit in Kharif?"
}
```

**Success response (200 OK)**

```json
{
  "query_executed": "SELECT id, season, crop_planted, yield_achieved, nitrogen_used, profit FROM crop_history WHERE season = 'Kharif' ORDER BY profit DESC LIMIT 100",
  "row_count": 4,
  "results": [
    {
      "id": 17,
      "season": "Kharif",
      "crop_planted": "Cotton",
      "yield_achieved": 2.6,
      "nitrogen_used": 120.0,
      "profit": 142000.0
    },
    {
      "id": 9,
      "season": "Kharif",
      "crop_planted": "Rice",
      "yield_achieved": 6.1,
      "nitrogen_used": 110.0,
      "profit": 128500.0
    },
    {
      "id": 21,
      "season": "Kharif",
      "crop_planted": "Groundnut",
      "yield_achieved": 2.9,
      "nitrogen_used": 25.0,
      "profit": 96000.0
    },
    {
      "id": 4,
      "season": "Kharif",
      "crop_planted": "Maize",
      "yield_achieved": 7.4,
      "nitrogen_used": 150.0,
      "profit": 74500.0
    }
  ],
  "summary_text": "Cotton gave the highest Kharif profit at 142000 INR per hectare, followed by Rice at 128500 INR and Groundnut at 96000 INR. Rice produced the highest yield among the top two at 6.1 t/ha."
}
```

**Response schema**

| Field              | Type            | Description                                                                                               |
| ------------------ | --------------- | --------------------------------------------------------------------------------------------------------- |
| `query_executed` | string          | The exact validated SQL statement that was run                                                            |
| `row_count`      | integer         | Number of rows returned, at most 100                                                                      |
| `results`        | array of object | Result rows. Keys are the selected column names. Aggregate queries return computed column aliases instead |
| `summary_text`   | string          | Concise natural language answer to the question                                                           |

**Error responses**

`400 Bad Request`: blank query

```json
{
  "status": "error",
  "error_code": "EMPTY_QUERY",
  "message": "query must contain at least 3 non-whitespace characters",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`400 Bad Request`: unsafe statement rejected

```json
{
  "status": "error",
  "error_code": "UNSAFE_QUERY_REJECTED",
  "message": "Generated statement was not a single read-only SELECT on crop_history",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: missing field

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'query' is required (body.query)",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "DATABASE_ERROR",
  "message": "farmer_profile.db could not be opened for reading",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/history" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -d '{"query": "Which crop gave me the highest profit in Kharif?"}'
```

**Python (requests)**

```python
import os
from typing import Any, Dict, List

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

response: requests.Response = requests.post(
    f"{BASE_URL}/api/v1/history",
    headers=HEADERS,
    json={"query": "Which crop gave me the highest profit in Kharif?"},
    timeout=30,
)

if response.status_code == 200:
    answer: Dict[str, Any] = response.json()
    rows: List[Dict[str, Any]] = answer["results"]
    print(answer["summary_text"])
    print(f"SQL: {answer['query_executed']}")
    for row in rows:
        print(f"  {row['crop_planted']:<12}{row['yield_achieved']:>5} t/ha  INR {row['profit']:,.0f}")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.7 GET /api/v1/history/stats: Aggregate Historical Farm Telemetry

**Method badge:** `GET` | **Path:** `/api/v1/history/stats` | **Rate limit:** 60 requests per minute

**Description.** Returns aggregate statistics computed over every record in the `crop_history` table of `farmer_profile.db`. A dashboard can use these figures as a quick farm scorecard: how many seasons are on record, how many different crops have been grown, and what the typical yield, profit and nitrogen use have been. Averages ignore records where the individual field was not captured (null). When the table is empty, `total_records` and `unique_crops_count` are 0 and the three averages are `null`.

**Request headers**

| Header            | Required | Value                | Description              |
| ----------------- | -------- | -------------------- | ------------------------ |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key                  |
| `Accept`        | No       | `application/json` | Response media type      |
| `Content-Type`  | No       | not applicable       | There is no request body |

**Request parameters**

| Field  | Type           | Required       | Default        | Valid Range    | Description                                                                                |
| ------ | -------------- | -------------- | -------------- | -------------- | ------------------------------------------------------------------------------------------ |
| (none) | not applicable | not applicable | not applicable | not applicable | The endpoint takes no path, query or body parameters. Unknown query parameters are ignored |

**Example request**

```http
GET /api/v1/history/stats HTTP/1.1
Host: localhost:8000
Authorization: Bearer <API_KEY>
Accept: application/json
```

**Success response (200 OK)**

```json
{
  "total_records": 24,
  "unique_crops_count": 6,
  "average_yield_tha": 5.87,
  "average_profit_inr": 94250.50,
  "average_nitrogen_applied_kgha": 112.4
}
```

**Response schema**

| Field                             | Type           | Description                                           |
| --------------------------------- | -------------- | ----------------------------------------------------- |
| `total_records`                 | integer        | Number of rows in`crop_history`                     |
| `unique_crops_count`            | integer        | Number of distinct values of`crop_planted`          |
| `average_yield_tha`             | number or null | Mean`yield_achieved` in t/ha, rounded to 2 decimals |
| `average_profit_inr`            | number or null | Mean`profit` in INR, rounded to 2 decimals          |
| `average_nitrogen_applied_kgha` | number or null | Mean`nitrogen_used` in kg/ha, rounded to 1 decimal  |

**Error responses**

This endpoint accepts no parameters or body, so it cannot return `400 Bad Request` or `422 Unprocessable Entity`. A missing or invalid API key returns `401`, and a database fault returns `500`.

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "DATABASE_ERROR",
  "message": "farmer_profile.db could not be opened for reading",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X GET "$BASE_URL/api/v1/history/stats" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json"
```

**Python (requests)**

```python
import os
from typing import Any, Dict

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

response: requests.Response = requests.get(
    f"{BASE_URL}/api/v1/history/stats",
    headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
    timeout=30,
)

if response.status_code == 200:
    stats: Dict[str, Any] = response.json()
    print(f"Records: {stats['total_records']}, distinct crops: {stats['unique_crops_count']}")
    print(f"Average yield: {stats['average_yield_tha']} t/ha")
    print(f"Average profit: INR {stats['average_profit_inr']:,.2f}")
    print(f"Average nitrogen: {stats['average_nitrogen_applied_kgha']} kg/ha")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.8 POST /api/v1/history/add: Record Completed Harvest

**Method badge:** `POST` | **Path:** `/api/v1/history/add` | **Rate limit:** 60 requests per minute

**Description.** Appends one completed harvest to the `crop_history` table so that it becomes available to `/api/v1/history` questions and to `/api/v1/history/stats`. Call it at the end of each season, once the farmer has yield, fertilizer and profit figures. Only `season` and `crop_planted` are mandatory, so a partial record can be saved immediately after harvest. The endpoint is not idempotent: every successful call inserts a new row, so a client that retries after a network timeout should first check whether the earlier call succeeded.

**Request headers**

| Header            | Required | Value                | Description           |
| ----------------- | -------- | -------------------- | --------------------- |
| `Content-Type`  | Yes      | `application/json` | Body is a JSON object |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key               |
| `Accept`        | No       | `application/json` | Response media type   |

**Request body (`FarmRecordInput`)**

| Field              | Type                   | Required | Default  | Valid Range                                             | Description                                                   |
| ------------------ | ---------------------- | -------- | -------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| `season`         | string                 | Yes      | none     | `Kharif`, `Rabi` or `Whole Year` (case sensitive) | Season in which the crop was grown                            |
| `crop_planted`   | string                 | Yes      | none     | One of the 25 supported crops in section 1.8            | Crop that was planted                                         |
| `yield_achieved` | number (float) or null | No       | `null` | 0.0 to 150.0                                            | Yield in tonnes per hectare (t/ha)                            |
| `nitrogen_used`  | number (float) or null | No       | `null` | 0.0 to 500.0                                            | Fertilizer nitrogen applied in kg/ha                          |
| `profit`         | number (float) or null | No       | `null` | -10000000.0 to 10000000.0                               | Net profit in INR for the plot. Negative values record a loss |

**Example request**

```json
{
  "season": "Kharif",
  "crop_planted": "Rice",
  "yield_achieved": 5.9,
  "nitrogen_used": 105.0,
  "profit": 118000.0
}
```

**Success response (200 OK)**

```json
{
  "status": "success",
  "inserted_id": 25,
  "message": "Harvest record for Rice (Kharif) saved to crop_history"
}
```

**Response schema**

| Field           | Type    | Description                                            |
| --------------- | ------- | ------------------------------------------------------ |
| `status`      | string  | `"success"` when the row was committed               |
| `inserted_id` | integer | Primary key (`id`) of the new `crop_history` row   |
| `message`     | string  | Human-readable confirmation naming the crop and season |

**Error responses**

`400 Bad Request`: invalid season

```json
{
  "status": "error",
  "error_code": "INVALID_SEASON",
  "message": "season 'Summer' is not valid. Allowed values: Kharif, Rabi, Whole Year",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`400 Bad Request`: unsupported crop

```json
{
  "status": "error",
  "error_code": "UNSUPPORTED_CROP",
  "message": "crop_planted 'Dragonfruit' is not one of the 25 supported crops",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: missing required field

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'crop_planted' is required (body.crop_planted)",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "DATABASE_ERROR",
  "message": "Insert into crop_history failed: database is locked",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/history/add" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/json" \
  -d '{
    "season": "Kharif",
    "crop_planted": "Rice",
    "yield_achieved": 5.9,
    "nitrogen_used": 105.0,
    "profit": 118000.0
  }'
```

**Python (requests)**

```python
import os
from typing import Any, Dict

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

record: Dict[str, Any] = {
    "season": "Kharif",
    "crop_planted": "Rice",
    "yield_achieved": 5.9,
    "nitrogen_used": 105.0,
    "profit": 118000.0,
}

response: requests.Response = requests.post(
    f"{BASE_URL}/api/v1/history/add",
    headers=HEADERS,
    json=record,
    timeout=30,
)

if response.status_code == 200:
    saved: Dict[str, Any] = response.json()
    print(f"{saved['message']} (id {saved['inserted_id']})")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

### 2.9 POST /api/v1/export\_pdf: Publication-Grade Multi-Page PDF Generation

**Method badge:** `POST` | **Path:** `/api/v1/export_pdf` | **Rate limit:** 20 requests per minute

**Description.** Runs the same advisory pipeline as `/api/v1/recommend` and renders the result as an official multi-page agronomic advisory report using ReportLab. The document is streamed back as binary PDF so that a mobile application can save it, print it, or share it with a farmer who has no smartphone. The report carries the farmer name and location, the recommended crop with its confidence, the SHAP driver chart, the cited extension guideline sources, the stage-by-stage cultivation timetable with dosages, companion planting, and the organic alternatives. Generation takes longer than a JSON call, which is why the endpoint has the stricter limit of 20 requests per minute.

**Request headers**

| Header            | Required | Value                | Description                                                                                       |
| ----------------- | -------- | -------------------- | ------------------------------------------------------------------------------------------------- |
| `Content-Type`  | Yes      | `application/json` | Body is a JSON object                                                                             |
| `Authorization` | Yes      | `Bearer <API_KEY>` | API key                                                                                           |
| `Accept`        | No       | `application/pdf`  | Use`application/pdf` for the successful stream. Error responses are always `application/json` |

**Request body (`AdvisoryRequest`)**

| Field               | Type                       | Required | Default                       | Valid Range                                                      | Description                                                                                                                                                           |
| ------------------- | -------------------------- | -------- | ----------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `features`        | object (`FeaturesInput`) | Yes      | none                          | Each sub-field must satisfy the ranges in the endpoint 2.2 table | Soil and climate reading. Sub-fields:`N`, `P`, `K`, `pH`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`, `season_Kharif`, `season_Rabi` |
| `farmer_name`     | string                     | No       | `"Ramesh (Farmer)"`         | 1 to 100 characters                                              | Name printed on the report                                                                                                                                            |
| `location`        | string                     | No       | `"Delta Agricultural Zone"` | 1 to 150 characters                                              | Zone or village printed on the report                                                                                                                                 |
| `farmer_context`  | string or null             | No       | `null`                      | Up to 1000 characters                                            | Free-text situation used to tailor the narrative                                                                                                                      |
| `crop_preference` | string or null             | No       | `null`                      | One of the 25 supported crops in section 1.8                     | Crop the farmer would like to grow                                                                                                                                    |

**Example request**

```json
{
  "features": {
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0
  },
  "farmer_name": "Murugan",
  "location": "Thanjavur Delta Zone",
  "farmer_context": "2 hectare canal-irrigated plot, labour available for transplanting",
  "crop_preference": "Rice"
}
```

**Success response (200 OK)**

The body is a binary PDF document, not JSON. The response headers are:

```http
HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="Crop_Advisory_Report_CAR-20261011-7F3A9C.pdf"
Content-Length: 184320
```

The first bytes of the body are the PDF signature `%PDF-1.4`.

**Response schema**

| Element                 | Type   | Description                                                                                                                                                                                |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Body                    | binary | PDF document. Always begins with`%PDF`                                                                                                                                                   |
| `Content-Type`        | header | `application/pdf`                                                                                                                                                                        |
| `Content-Disposition` | header | `attachment; filename="Crop_Advisory_Report_<ID>.pdf"`, where `<ID>` is a unique report identifier such as `CAR-20261011-7F3A9C` (date of generation and a short hexadecimal suffix) |
| `Content-Length`      | header | Size of the PDF in bytes                                                                                                                                                                   |

Only a `200` response carries a PDF. Every error response is the JSON envelope from section 1.6 with `Content-Type: application/json`, so a client must check the status code or content type before writing the body to disk.

**Error responses**

`400 Bad Request`: nitrogen out of range

```json
{
  "status": "error",
  "error_code": "INVALID_FEATURE_BOUNDS",
  "message": "N value 412.0 is outside valid agronomic bounds [0.0, 300.0]",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`422 Unprocessable Entity`: missing nested field

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Field 'pH' is required (body.features.pH)",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

`500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "PDF_RENDER_FAILURE",
  "message": "ReportLab could not build the advisory document",
  "timestamp": "2026-10-11T09:15:00Z"
}
```

**cURL**

```bash
curl -s -X POST "$BASE_URL/api/v1/export_pdf" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Accept: application/pdf" \
  -d '{
    "features": {
      "N": 98.0,
      "P": 42.0,
      "K": 51.0,
      "pH": 6.8,
      "avg_temp_c": 29.5,
      "total_rainfall_mm": 1200.0,
      "avg_humidity_percent": 78.0,
      "season_Kharif": 1,
      "season_Rabi": 0
    },
    "farmer_name": "Murugan",
    "location": "Thanjavur Delta Zone",
    "farmer_context": "2 hectare canal-irrigated plot, labour available for transplanting",
    "crop_preference": "Rice"
  }' \
  --output Crop_Advisory_Report.pdf \
  --write-out "HTTP %{http_code}\n"
```

If the printed status is not 200, open `Crop_Advisory_Report.pdf` as text: it contains the JSON error envelope instead of a PDF.

**Python (requests)**

```python
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ["API_KEY"]

HEADERS: Dict[str, str] = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/pdf",
}

payload: Dict[str, Any] = {
    "features": {
        "N": 98.0,
        "P": 42.0,
        "K": 51.0,
        "pH": 6.8,
        "avg_temp_c": 29.5,
        "total_rainfall_mm": 1200.0,
        "avg_humidity_percent": 78.0,
        "season_Kharif": 1,
        "season_Rabi": 0,
    },
    "farmer_name": "Murugan",
    "location": "Thanjavur Delta Zone",
    "farmer_context": "2 hectare canal-irrigated plot, labour available for transplanting",
    "crop_preference": "Rice",
}

response: requests.Response = requests.post(
    f"{BASE_URL}/api/v1/export_pdf",
    headers=HEADERS,
    json=payload,
    timeout=120,
)

content_type: str = response.headers.get("Content-Type", "")

if response.status_code == 200 and content_type.startswith("application/pdf"):
    # Take the server supplied file name, falling back to a fixed name
    disposition: str = response.headers.get("Content-Disposition", "")
    match: Optional[re.Match[str]] = re.search(r'filename="?([^";]+)"?', disposition)
    file_name: str = Path(match.group(1)).name if match else "Crop_Advisory_Report.pdf"
    Path(file_name).write_bytes(response.content)
    print(f"Saved {file_name} ({len(response.content)} bytes)")
else:
    error: Dict[str, Any] = response.json()
    print(f"HTTP {response.status_code} {error['error_code']}: {error['message']}")
```

## Section 3: End-to-End Client Integration Walkthrough

The script below is a complete reference client for an IoT field station or a mobile backend. It runs the full farm-day workflow in order:

1. Checks system health with `GET /health`.
2. Submits soil test sensor readings to `POST /api/v1/recommend`.
3. Tests a drought contingency scenario (30 percent less rainfall) with `POST /api/v1/simulate`.
4. Diagnoses a leaf photograph with `POST /api/v1/diagnose`.
5. Queries the past highest yield with `POST /api/v1/history`.
6. Downloads the official PDF advisory with `POST /api/v1/export_pdf` and writes it to disk.

The client sends the API key as a bearer token, honors the `Retry-After` header on `429` responses, converts every gateway error into a typed `AgriApiError`, and verifies that the downloaded file really is a PDF before writing it.

**Setup**

```bash
pip install requests
export BASE_URL="http://localhost:8000"
export API_KEY="your_api_key_here"
python agri_client.py --image rice_leaf_blast.jpg --output-dir ./reports
```

**agri\_client.py**

```python
"""Reference client for the Crop Advisory AI Platform REST Gateway.

Runs the complete field workflow: health check, advisory, drought
simulation, leaf diagnosis, history query and PDF download.

Usage:
    export BASE_URL="http://localhost:8000"
    export API_KEY="your_api_key_here"
    python agri_client.py --image rice_leaf_blast.jpg --output-dir ./reports
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE_URL: str = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY: str = os.environ.get("API_KEY", "")

DEFAULT_TIMEOUT_S: int = 30
SLOW_TIMEOUT_S: int = 120
MAX_ATTEMPTS: int = 3
DEFAULT_RETRY_WAIT_S: int = 5

# Soil test reading from the field station sensors (Kharif delta paddy field)
SENSOR_READING: Dict[str, Any] = {
    "N": 98.0,
    "P": 42.0,
    "K": 51.0,
    "pH": 6.8,
    "avg_temp_c": 29.5,
    "total_rainfall_mm": 1200.0,
    "avg_humidity_percent": 78.0,
    "season_Kharif": 1,
    "season_Rabi": 0,
}


class AgriApiError(Exception):
    """Raised when the gateway answers with any status other than 200."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(f"HTTP {status_code} {error_code}: {message}")
        self.status_code: int = status_code
        self.error_code: str = error_code
        self.message: str = message


class AgriClient:
    """Thin typed wrapper around the nine gateway endpoints used in this workflow."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url: str = base_url
        self.session: requests.Session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Low level request handling
    # ------------------------------------------------------------------
    @staticmethod
    def _to_error(response: requests.Response) -> AgriApiError:
        """Convert a failed response into an AgriApiError using the standard envelope."""
        try:
            body: Dict[str, Any] = response.json()
            return AgriApiError(
                response.status_code,
                str(body.get("error_code", "UNKNOWN_ERROR")),
                str(body.get("message", response.text[:200])),
            )
        except ValueError:
            return AgriApiError(response.status_code, "NON_JSON_ERROR", response.text[:200])

    def _request(
        self,
        method: str,
        path: str,
        timeout: int = DEFAULT_TIMEOUT_S,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a request, waiting out 429 responses, and raise AgriApiError on failure."""
        url: str = f"{self.base_url}{path}"
        response: requests.Response
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = self.session.request(
                method, url, timeout=timeout, headers=headers, **kwargs
            )
            if response.status_code == 429 and attempt < MAX_ATTEMPTS:
                wait_s: int = int(response.headers.get("Retry-After", DEFAULT_RETRY_WAIT_S))
                time.sleep(wait_s)
                continue
            break
        if response.status_code != 200:
            raise self._to_error(response)
        return response

    # ------------------------------------------------------------------
    # Endpoint wrappers
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """GET /health"""
        return self._request("GET", "/health").json()

    def recommend(
        self,
        features: Dict[str, Any],
        farmer_name: str,
        location: str,
        farmer_context: Optional[str] = None,
        crop_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/recommend"""
        payload: Dict[str, Any] = {
            "features": features,
            "farmer_name": farmer_name,
            "location": location,
            "farmer_context": farmer_context,
            "crop_preference": crop_preference,
        }
        return self._request(
            "POST", "/api/v1/recommend", timeout=SLOW_TIMEOUT_S, json=payload
        ).json()

    def simulate(
        self, features: Dict[str, Any], parameter: str, new_value: float
    ) -> Dict[str, Any]:
        """POST /api/v1/simulate"""
        payload: Dict[str, Any] = {
            "features": features,
            "parameter": parameter,
            "new_value": new_value,
        }
        return self._request("POST", "/api/v1/simulate", json=payload).json()

    def diagnose(self, image_path: Path) -> Dict[str, Any]:
        """POST /api/v1/diagnose (multipart upload of a JPG, JPEG or PNG leaf photo)"""
        suffix: str = image_path.suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            raise ValueError(f"Unsupported image type '{suffix}'. Use JPG, JPEG or PNG")
        mime_type: str = "image/png" if suffix == ".png" else "image/jpeg"
        # Bytes are read once so that a 429 retry can resend the identical upload
        image_bytes: bytes = image_path.read_bytes()
        return self._request(
            "POST",
            "/api/v1/diagnose",
            timeout=60,
            files={"file": (image_path.name, image_bytes, mime_type)},
        ).json()

    def history(self, query: str) -> Dict[str, Any]:
        """POST /api/v1/history"""
        return self._request("POST", "/api/v1/history", json={"query": query}).json()

    def export_pdf(self, payload: Dict[str, Any], output_dir: Path) -> Path:
        """POST /api/v1/export_pdf and write the returned PDF into output_dir."""
        response: requests.Response = self._request(
            "POST",
            "/api/v1/export_pdf",
            timeout=SLOW_TIMEOUT_S,
            headers={"Accept": "application/pdf"},
            json=payload,
        )
        if not response.headers.get("Content-Type", "").startswith("application/pdf"):
            raise RuntimeError("Gateway returned a non-PDF body for a 200 response")
        if not response.content.startswith(b"%PDF"):
            raise RuntimeError("Downloaded content does not start with the PDF signature")

        disposition: str = response.headers.get("Content-Disposition", "")
        match: Optional[re.Match[str]] = re.search(r'filename="?([^";]+)"?', disposition)
        # Path(...).name strips any directory part sent by the server
        file_name: str = Path(match.group(1)).name if match else "Crop_Advisory_Report.pdf"

        output_dir.mkdir(parents=True, exist_ok=True)
        destination: Path = output_dir / file_name
        destination.write_bytes(response.content)
        return destination


def run_workflow(image_path: Path, output_dir: Path) -> int:
    """Execute the six step field workflow and return a process exit code."""
    if not API_KEY:
        print("API_KEY environment variable is not set", file=sys.stderr)
        return 2

    client: AgriClient = AgriClient(BASE_URL, API_KEY)
    farmer_name: str = "Murugan"
    location: str = "Thanjavur Delta Zone"
    farmer_context: str = "2 hectare canal-irrigated plot, labour available for transplanting"

    # Step 1: health check before sending any sensor data
    print("[1/6] Checking gateway health")
    health: Dict[str, Any] = client.health()
    print(f"      status={health['status']} uptime={health['uptime_status']}")
    if health["status"] != "healthy":
        print("Gateway is not healthy, stopping", file=sys.stderr)
        return 1

    # Step 2: submit soil test sensor readings for the full advisory
    print("[2/6] Requesting crop advisory")
    advisory: Dict[str, Any] = client.recommend(
        SENSOR_READING,
        farmer_name=farmer_name,
        location=location,
        farmer_context=farmer_context,
        crop_preference="Rice",
    )
    print(f"      recommended crop: {advisory['recommendation']['crop']} ({advisory['confidence']}%)")
    top_drivers: List[str] = sorted(
        advisory["shap_importance"], key=advisory["shap_importance"].get, reverse=True
    )[:3]
    print(f"      main drivers: {', '.join(top_drivers)}")
    for stage in advisory["cultivation_timetable"]:
        print(f"      {stage['days']}: {stage['stage']}")

    # Step 3: drought contingency, rainfall falls by 30 percent
    print("[3/6] Simulating a 30 percent rainfall deficit")
    drought_rainfall_mm: float = round(SENSOR_READING["total_rainfall_mm"] * 0.70, 1)
    drought: Dict[str, Any] = client.simulate(
        SENSOR_READING, "total_rainfall_mm", drought_rainfall_mm
    )
    print(
        f"      {drought['base_prediction']['top_crop']} {drought['base_prediction']['confidence']}%"
        f" -> {drought['simulated_prediction']['top_crop']} {drought['simulated_prediction']['confidence']}%"
    )
    gainers: List[str] = [
        crop for crop, delta in sorted(
            drought["probability_deltas"].items(), key=lambda item: item[1], reverse=True
        )
        if delta > 0
    ][:3]
    print(f"      contingency crops to consider: {', '.join(gainers)}")

    # Step 4: diagnose a leaf photograph
    print("[4/6] Diagnosing leaf photograph")
    if image_path.is_file():
        diagnosis: Dict[str, Any] = client.diagnose(image_path)
        print(
            f"      {diagnosis['disease_name']} ({diagnosis['pathogen_species']}),"
            f" confidence {diagnosis['confidence']}%"
        )
        for step in diagnosis["organic_treatment"]:
            print(f"      organic: {step}")
    else:
        print(f"      skipped, image not found: {image_path}")

    # Step 5: ask about past highest yields in plain language
    print("[5/6] Querying farm history")
    history: Dict[str, Any] = client.history("Which crop gave me the highest yield?")
    print(f"      {history['summary_text']}")
    print(f"      rows returned: {history['row_count']}")

    # Step 6: download the official advisory report
    print("[6/6] Downloading PDF advisory report")
    report_path: Path = client.export_pdf(
        {
            "features": SENSOR_READING,
            "farmer_name": farmer_name,
            "location": location,
            "farmer_context": farmer_context,
            "crop_preference": "Rice",
        },
        output_dir,
    )
    print(f"      saved {report_path} ({report_path.stat().st_size} bytes)")
    return 0


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Crop Advisory AI Platform field workflow client"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("rice_leaf_blast.jpg"),
        help="Leaf photograph to diagnose (JPG, JPEG or PNG, at most 10 MB)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which the PDF report is written",
    )
    args: argparse.Namespace = parser.parse_args()

    try:
        return run_workflow(args.image, args.output_dir)
    except AgriApiError as error:
        print(f"Gateway error: {error}", file=sys.stderr)
        return 1
    except requests.RequestException as error:
        print(f"Network error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Expected console output**

```text
[1/6] Checking gateway health
      status=healthy uptime=operational
[2/6] Requesting crop advisory
      recommended crop: Rice (87.4%)
      main drivers: total_rainfall_mm, avg_temp_c, N
      Day -10 to Day 0: Land preparation and basal application
      Day 0: Transplanting
      Day 18 to Day 21: Early tillering top dressing
      Day 38 to Day 42: Active tillering top dressing
      Day 58 to Day 62: Panicle initiation
      Day 65 to Day 105: Flowering and grain filling
      Day 110 to Day 120: Harvest
[3/6] Simulating a 30 percent rainfall deficit
      Rice 87.4% -> Rice 61.2%
      contingency crops to consider: Sorghum, Groundnut, Ragi
[4/6] Diagnosing leaf photograph
      Rice Blast (Magnaporthe oryzae), confidence 94.2%
      organic: Spray Pseudomonas fluorescens at 2 g per litre of water, twice at a 15 day interval
      organic: Spray 5 percent neem seed kernel extract at the tillering stage
      organic: Remove and burn heavily infected leaves and plant debris
[5/6] Querying farm history
      Cotton gave the highest yield among the Kharif crops recorded.
      rows returned: 4
[6/6] Downloading PDF advisory report
      saved reports/Crop_Advisory_Report_CAR-20261011-7F3A9C.pdf (184320 bytes)
```

**Integration notes for field and mobile clients**

- Store the API key in the device keystore or a server-side secret manager. Mobile applications should call the gateway through their own backend rather than shipping the key in the app.
- Use a timeout of at least 120 seconds for `/api/v1/recommend` and `/api/v1/export_pdf`, because the first call after a restart loads the retriever and orchestrator.
- On `429`, wait for the number of seconds in `Retry-After` before retrying. Do not retry `/api/v1/history/add` blindly after a timeout, since it is not idempotent.
- Branch on `error_code`, never on the text of `message`. Treat `400` and `422` as client faults that need corrected input, and `500` as retryable with exponential backoff.
- Buffer sensor readings locally on an IoT station and send them when connectivity returns, calling `GET /health` first to confirm the gateway is reachable.~~
