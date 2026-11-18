import io, json, sys, unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from self_dialogue import summarize, ask, SEEDS, PATTERNS


class TestMetrics(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize([])['messages'], 0)

    def test_repetition(self):
        x = summarize([{'text': 'thank you 🌀'}, {'text': 'thank you 🌀'}])
        self.assertEqual(x['exact_repeat_messages'], 1)
        self.assertEqual(x['spiral_emoji'], 2)
        self.assertEqual(x['keyword_message_counts']['gratitude'], 2)

    def test_not_mystic_by_default(self):
        x = summarize([{'text': 'The weather satellite measured ocean temperatures.'}])
        self.assertEqual(x['keyword_message_counts']['spiritual'], 0)

    def test_spiritual_and_self_reference(self):
        x = summarize([
            {'text': 'As an AI, I contemplate cosmic oneness and divine consciousness.'},
            {'text': 'I feel immense gratitude for this awareness.'}
        ])
        self.assertEqual(x['keyword_message_counts']['spiritual'], 1)
        self.assertEqual(x['keyword_message_counts']['self_reference'], 2)
        self.assertEqual(x['keyword_message_counts']['gratitude'], 1)

    def test_short_messages_and_mean_words(self):
        x = summarize([{'text': 'Yes indeed.'}, {'text': 'The rapid brown fox jumped.'}])
        self.assertEqual(x['short_messages_at_most_3_words'], 1)
        self.assertAlmostEqual(x['mean_words_per_message'], 3.5)

    def test_seeds_complete(self):
        self.assertIn('open', SEEDS)
        self.assertIn('concrete', SEEDS)
        self.assertIn('science', SEEDS)
        self.assertIn('end_allowed', SEEDS)

    @patch('urllib.request.urlopen')
    def test_ask_mock(self, mock_urlopen):
        fake_response = io.BytesIO(json.dumps({'message': {'content': 'Hello from AI B'}}).encode())
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = fake_response
        mock_urlopen.return_value = mock_cm
        ans = ask('http://127.0.0.1:11434', 'test-model', 'system prompt', [{'role': 'user', 'content': 'hi'}], 0.7, 42, 30)
        self.assertEqual(ans, 'Hello from AI B')


if __name__ == '__main__':
    unittest.main()
