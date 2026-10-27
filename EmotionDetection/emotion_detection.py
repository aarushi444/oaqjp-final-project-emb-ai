
"""Emotion detection using IBM Watson NLP."""

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
}

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""

    if not text_to_analyze or not text_to_analyze.strip():
        return {key: None for key in (*EMOTIONS, "dominant_emotion")}

    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(
        URL, headers=HEADERS, json=payload, timeout=30
    )

    if response.status_code == 400:
        return {key: None for key in (*EMOTIONS, "dominant_emotion")}

    response.raise_for_status()

    result = response.json()
    emotions = result["emotionPredictions"][0]["emotion"]

    scores = {key: emotions[key] for key in EMOTIONS}
    scores["dominant_emotion"] = max(scores, key=scores.get)

    return scores
