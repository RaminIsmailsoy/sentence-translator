import os
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.fakes import GOOD, Provider, api_failure

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SOURCE = {"sentence": "I love my little dog."}
NEXT = {"sentence": "My mother drinks warm milk."}


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        self.environment = patch.dict(os.environ, {"GROQ_API_KEY": "unit-test-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def start(self, provider):
        factory = patch("groq.Groq", return_value=provider.client)
        factory.start()
        self.addCleanup(factory.stop)
        return AppTest.from_file(APP, default_timeout=15).run()

    def submit(self, app, answer="Я люблю свою маленькую собаку."):
        app.text_input[0].set_value(answer)
        next(button for button in app.button if button.label == "✅ Cavabı yoxla").click()
        app.run()

    def test_missing_key_is_a_setup_screen(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            app = AppTest.from_file(APP).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Başlamaq", app.info[0].value)
        self.assertEqual(app.text_input[0].label, "Groq API açarı")

    def test_success_auto_advances_and_rerun_does_not_call_api(self):
        provider = Provider(SOURCE, GOOD, NEXT)
        app = self.start(provider)
        self.assertEqual(len(app.exception), 0)
        app.run()
        self.assertEqual(len(provider.requests), 1)
        self.submit(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.practice.current.sentence, NEXT["sentence"])
        self.assertEqual(app.session_state.practice.counts["Düzgün"], 1)
        self.assertEqual(app.text_input[0].value, "")
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(provider.model_calls, 1)

    def test_rate_limit_preserves_draft_and_sentence_then_retry_works(self):
        provider = Provider(SOURCE, api_failure(), GOOD, NEXT)
        app = self.start(provider)
        self.submit(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.text_input[0].value, "Я люблю свою маленькую собаку.")
        self.assertEqual(app.session_state.practice.current.sentence, SOURCE["sentence"])
        self.assertEqual(sum(app.session_state.practice.counts.values()), 0)
        self.assertTrue(any("limitinə" in error.value for error in app.error))
        self.submit(app)
        self.assertEqual(app.session_state.practice.counts["Düzgün"], 1)
        self.assertEqual(len(provider.requests), 4)

    def test_new_sentence_failure_keeps_result(self):
        provider = Provider(SOURCE, GOOD, api_failure(), NEXT)
        app = self.start(provider)
        self.submit(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.practice.last.evaluation["result"], "Düzgün")
        self.assertTrue(app.session_state.practice.current.answered)
        self.assertEqual(len(app.text_input), 0)
        app.button(key="next_sentence").click().run()
        self.assertEqual(app.session_state.practice.current.sentence, NEXT["sentence"])
        self.assertEqual(app.session_state.practice.counts["Düzgün"], 1)

    def test_manual_mode_waits_for_next_button(self):
        provider = Provider(SOURCE, GOOD, NEXT)
        app = self.start(provider)
        app.toggle(key="auto_next").set_value(False).run()
        self.submit(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(provider.requests), 2)
        self.assertTrue(app.session_state.practice.current.answered)
        app.button(key="next_sentence").click().run()
        self.assertEqual(app.session_state.practice.current.sentence, NEXT["sentence"])

    def test_direction_change_resets_draft_and_uses_russian(self):
        provider = Provider(SOURCE, {"sentence": "Моя сестра сейчас читает книгу."})
        app = self.start(provider)
        app.text_input[0].set_value("unfinished draft").run()
        app.radio(key="direction").set_value("ru_en").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.text_input[0].value, "")
        self.assertEqual(app.session_state.practice.current.direction, "ru_en")
        self.assertEqual(len(provider.requests), 2)

    def test_initial_failure_waits_for_explicit_retry(self):
        provider = Provider(api_failure(), SOURCE)
        app = self.start(provider)
        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state.practice.current)
        app.run()
        self.assertEqual(len(provider.requests), 1)
        app.button(key="next_sentence").click().run()
        self.assertEqual(app.session_state.practice.current.sentence, SOURCE["sentence"])


if __name__ == "__main__":
    unittest.main()
