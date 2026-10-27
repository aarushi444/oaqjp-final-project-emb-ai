
"""Unit tests for the emotion detection application."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Test emotion classification."""

    def check_emotion(self, dominant):
        """Verify the dominant emotion in a mocked Watson response."""
        scores = {
            "anger": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
            "joy": 0.1,
            "sadness": 0.1,
        }
        scores[dominant] = 0.9

        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ):
            result = emotion_detector("Test sentence")

        self.assertEqual(result["dominant_emotion"], dominant)

    def test_joy(self):
        self.check_emotion("joy")

    def test_anger(self):
        self.check_emotion("anger")

    def test_disgust(self):
        self.check_emotion("disgust")

    def test_fear(self):
        self.check_emotion("fear")

    def test_sadness(self):
        self.check_emotion("sadness")

    def test_empty_input(self):
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
