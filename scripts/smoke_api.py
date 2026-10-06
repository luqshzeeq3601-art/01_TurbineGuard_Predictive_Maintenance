"""Exercise an actual service with synthetic history, including model readiness."""

import argparse
import json
import math
from urllib.request import Request, urlopen


def smoke(base_url: str) -> None:
    def request(path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(base_url.rstrip("/") + path, data=body,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as response:
            assert response.status == 200
            return json.load(response)

    assert request("/health")["status"] == "healthy"
    ready = request("/ready")
    assert ready["model_loaded"] is True
    assert ready["bundle_version"] == "v0.2.1"
    history = []
    for cycle in range(1, 26):
        row = {"cycle": cycle, "op_1": -0.0005, "op_2": 0.0002, "op_3": 100.0}
        row.update({f"s{sensor:02d}": float(sensor * 10 + cycle * 0.1)
                    for sensor in range(1, 22)})
        history.append(row)
    result = request("/predict", {"dataset_id": "FD001", "engine_id": "synthetic-smoke",
                                  "history": history, "explain": True})
    assert result["bundle_version"] == "v0.2.1"
    assert math.isfinite(result["estimated_rul_cycles"]) and result["estimated_rul_cycles"] >= 0
    assert len(result["explanation"]["top_contributions"]) == 3
    print(json.dumps({"ready": ready, "synthetic_prediction": result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    smoke(parser.parse_args().url)
