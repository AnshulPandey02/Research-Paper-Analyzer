"""
Failure-mode tests for the analysis pipeline. No network access is needed:
every external service (CrossRef, arXiv, OpenAlex, Semantic Scholar, Gemini)
is replaced by a stub that is slow, down, rate limited or returns bad data.

Run from the project root:
    backend\\.venv\\Scripts\\python.exe -m unittest tests.test_resilience -v
"""

import os
import sys
import time
import unittest
from unittest import mock

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, BACKEND)

os.environ["PLAGIARISM_SEARCH_BUDGET_SECONDS"] = "4"
os.environ["GEMINI_PAPER_SEARCH"] = "false"  # keep the stubs in charge

import requests  # noqa: E402

from src import gemini_client, plagiarism_checker as pc  # noqa: E402
from src.bias_analyzer import BiasAnalyzer, result_to_dict  # noqa: E402
from src.gemini_client import GeminiError, parse_json_lenient  # noqa: E402
from src.text_extraction import PDFExtractionError, extract_text_from_bytes  # noqa: E402

SECTION = (
    "We propose a deepfake video detection framework that combines a convolutional "
    "neural network for spatial feature extraction with a temporal attention module "
    "that models inconsistencies across consecutive frames of manipulated videos."
)
SECTIONS = {"Title": "Deepfake video detection with temporal attention networks",
            "Abstract": SECTION, "Methodology": SECTION + " Training uses the Adam optimizer.",
            "Conclusions": SECTION + " Future work covers adversarial robustness."}

PAPER = {"title": "Deepfake detection with convolutional networks and attention",
         "abstract": "A convolutional neural network with temporal attention for deepfake video detection.",
         "url": "https://example.org/paper", "source": "stub"}


def ok_fetch(query, max_results, timeout):
    return [dict(PAPER)]


def down_fetch(query, max_results, timeout):
    raise requests.exceptions.ConnectionError("provider is down")


def rate_limited_fetch(query, max_results, timeout):
    raise pc._RateLimited(retry_after=30)


def hanging_fetch(query, max_results, timeout):
    time.sleep(30)  # far longer than the 4s search budget
    return [dict(PAPER)]


class ProviderStubs(unittest.TestCase):
    """Swap every provider's fetch function and remove politeness delays."""

    def stub(self, **fetch_by_name):
        default = fetch_by_name.pop("default", ok_fetch)
        self.calls = {p.name: 0 for p in pc.PROVIDERS}
        for provider in pc.PROVIDERS:
            fetch = fetch_by_name.get(provider.name, default)

            def counted(query, max_results, timeout, _fetch=fetch, _name=provider.name):
                self.calls[_name] += 1
                return _fetch(query, max_results, timeout)

            patcher = mock.patch.object(provider, "_fetch", counted)
            patcher.start()
            self.addCleanup(patcher.stop)
            interval = mock.patch.object(provider, "_min_interval", 0.0)
            interval.start()
            self.addCleanup(interval.stop)

    def setUp(self):
        pc.reset_search_state()
        self.addCleanup(pc.reset_search_state)


class PlagiarismResilience(ProviderStubs):

    def test_hanging_provider_does_not_block_or_crash(self):
        """Regression for 'TimeoutError: 1 (of 4) futures unfinished'."""
        self.stub(SemanticScholar=hanging_fetch)
        started = time.time()
        report = pc.analyze_plagiarism("", sections=SECTIONS)
        elapsed = time.time() - started
        self.assertLess(elapsed, 8, "a hanging provider must not hold the request")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["providers"]["SemanticScholar"], "timeout")
        self.assertEqual(report["providers"]["CrossRef"], "ok")
        self.assertGreater(report["overall_percent"], 0)
        self.assertEqual(report["warnings"], [], "no notice when the other databases covered every section")

    def test_one_provider_down_keeps_other_results(self):
        self.stub(arXiv=down_fetch)
        report = pc.analyze_plagiarism("", sections=SECTIONS)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["providers"]["arXiv"], "error")
        self.assertEqual(len(report["sections"]["Abstract"]["matches"]), 1)

    def test_all_providers_down_is_reported_as_failure(self):
        """No database reachable must not look like '0% similarity'."""
        self.stub(default=down_fetch)
        report = pc.analyze_plagiarism("", sections=SECTIONS)
        self.assertEqual(report["status"], "failed")
        self.assertIn("could not be performed", report["error"])
        self.assertEqual(report["overall_category"], "Not checked")
        for section in report["sections"].values():
            self.assertEqual(section["status"], "unavailable")

    def test_rate_limited_provider_is_skipped_not_hammered(self):
        self.stub(CrossRef=rate_limited_fetch)
        report = pc.analyze_plagiarism("", sections=SECTIONS)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["providers"]["CrossRef"], "rate_limited")
        self.assertEqual(self.calls["CrossRef"], 1, "after a 429 the provider must cool down")

    def test_repeated_queries_are_served_from_cache(self):
        self.stub()
        pc.analyze_plagiarism("", sections=SECTIONS)
        first = dict(self.calls)
        pc.analyze_plagiarism("", sections=SECTIONS)
        self.assertEqual(self.calls, first, "second run must not repeat provider requests")

    def test_section_extraction_never_raises(self):
        for text in ["", "short", "Abstract\n" + SECTION * 20 + "\n1. Introduction\nText\n5. Conclusion\nDone."]:
            sections = pc.extract_sections(text)
            self.assertEqual(set(sections), {"Title", "Abstract", "Methodology", "Conclusions"})


class TitleAndRateLimit(ProviderStubs):
    """Regression tests for 'Could not check: Title' and Semantic Scholar 429s."""

    TITLE_SECTIONS = dict(SECTIONS, Title="Attention Is All You Need")

    def test_short_titles_produce_a_real_query(self):
        self.assertEqual(pc.build_search_query("Attention Is All You Need"), "attention is all you need")
        self.assertEqual(pc.build_search_query("AttentionIsAllYouNeed"), "attention is all you need")
        self.assertEqual(pc._fix_concatenated_text("Attention and Recognition by Ashish"),
                         "Attention and Recognition by Ashish", "normal words must not be split")
        self.assertIn("paper", pc._fix_concatenated_text("thepaperpresentsanewmodel"))

    def test_title_is_checked_when_semantic_scholar_is_rate_limited(self):
        queries = []

        def recording(query, max_results, timeout):
            queries.append(query)
            return [dict(PAPER)]

        self.stub(default=recording, SemanticScholar=rate_limited_fetch)
        report = pc.analyze_plagiarism("", sections=self.TITLE_SECTIONS)
        title = report["sections"]["Title"]
        self.assertEqual(title["status"], "ok")
        self.assertEqual(len(title["matches"]), 1)
        self.assertIn("attention is all you need", queries)
        self.assertEqual(report["status"], "ok")
        self.assertFalse(any("Could not check" in w for w in report["warnings"]))
        self.assertEqual(report["providers"]["SemanticScholar"], "rate_limited")

    def test_text_without_words_is_skipped_not_reported_as_outage(self):
        self.stub()
        report = pc.analyze_plagiarism("", sections=dict(SECTIONS, Title="--- *** --- *** --- *** --- *** ---"))
        self.assertEqual(report["sections"]["Title"]["status"], "skipped")
        self.assertEqual(report["sections"]["Title"]["matches"], [])
        self.assertEqual(report["status"], "ok")
        self.assertFalse(any("Could not check" in w for w in report["warnings"]))

    def test_429_is_retried_with_exponential_backoff_then_succeeds(self):
        attempts, sleeps = [], []

        def flaky(query, max_results, timeout):
            attempts.append(query)
            if len(attempts) <= 2:
                raise pc._RateLimited()  # no Retry-After header, like Semantic Scholar
            return [dict(PAPER)]

        self.stub(SemanticScholar=flaky)
        provider = pc._PROVIDER_BY_NAME["SemanticScholar"]
        with mock.patch.object(pc.time, "sleep", side_effect=sleeps.append):
            results, status = provider.search("attention is all you need", 10, time.time() + 30)
        self.assertEqual(status, "ok")
        self.assertEqual(len(results), 1)
        self.assertEqual(len(attempts), 3)
        self.assertEqual([s for s in sleeps if s >= 1], [2.0, 4.0], "backoff must double")

    def test_retry_after_header_is_respected(self):
        sleeps, attempts = [], []

        def limited_once(query, max_results, timeout):
            attempts.append(1)
            if len(attempts) == 1:
                raise pc._RateLimited(retry_after=5)
            return [dict(PAPER)]

        self.stub(SemanticScholar=limited_once)
        with mock.patch.object(pc.time, "sleep", side_effect=sleeps.append):
            _, status = pc._PROVIDER_BY_NAME["SemanticScholar"].search("some query words", 10, time.time() + 30)
        self.assertEqual(status, "ok")
        self.assertIn(5, sleeps)

    def test_persistent_429_gives_up_and_cools_down(self):
        def always(query, max_results, timeout):
            raise pc._RateLimited()

        self.stub(SemanticScholar=always)
        provider = pc._PROVIDER_BY_NAME["SemanticScholar"]
        with mock.patch.object(pc.time, "sleep"):
            _, first = provider.search("query one two", 10, time.time() + 30)
            _, second = provider.search("another query here", 10, time.time() + 30)
        self.assertEqual((first, second), ("rate_limited", "rate_limited"))
        self.assertEqual(self.calls["SemanticScholar"], 1 + pc.RATE_LIMIT_RETRIES,
                         "bounded retries, then no further requests during the cool-down")

    def test_backoff_never_exceeds_the_search_budget(self):
        def always(query, max_results, timeout):
            raise pc._RateLimited(retry_after=8)

        self.stub(SemanticScholar=always)
        started = time.time()
        _, status = pc._PROVIDER_BY_NAME["SemanticScholar"].search("query one two", 10, time.time() + 4)
        self.assertEqual(status, "rate_limited")
        self.assertLess(time.time() - started, 1, "must not sleep past the deadline")


class GeminiParsing(unittest.TestCase):

    def test_lenient_json(self):
        self.assertEqual(parse_json_lenient('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_lenient('Here you go: {"a": [1, 2]} hope it helps'), {"a": [1, 2]})
        self.assertEqual(parse_json_lenient('[{"title": "x"}]'), [{"title": "x"}])
        for bad in ["", "   ", "I cannot help with that", '{"a": '] :
            with self.assertRaises(ValueError):
                parse_json_lenient(bad)

    def test_placeholder_keys_are_not_real(self):
        self.assertFalse(gemini_client.is_real_key("your_gemini_api_key_here"))
        self.assertFalse(gemini_client.is_real_key(""))
        self.assertTrue(gemini_client.is_real_key("AIza" + "b" * 35))


class GeminiFallback(unittest.TestCase):
    """Model chain behaviour with a stubbed SDK client (no network)."""

    class FakeError(Exception):
        def __init__(self, code):
            super().__init__(f"{code} error")
            self.code = code

    def run_chain(self, behaviour, **kwargs):
        """behaviour: model name -> callable() returning response text or raising."""
        calls = []

        class Models:
            def generate_content(_self, model, contents, config):
                calls.append(model)
                return mock.Mock(text=behaviour[model]())

        env = {"GEMINI_API_KEY": "AIza" + "b" * 35, "GEMINI_MODEL": "primary",
               "GEMINI_FALLBACK_MODELS": "second,third", "GEMINI_HEDGE_SECONDS": "2"}
        gemini_client._cooldown.clear()
        self.addCleanup(gemini_client._cooldown.clear)
        with mock.patch.dict(os.environ, env),              mock.patch.object(gemini_client, "_get_client", return_value=mock.Mock(models=Models())):
            return gemini_client.generate_json("prompt", **kwargs), calls

    def raiser(self, code):
        def fn():
            raise self.FakeError(code)
        return fn

    def test_503_falls_back_to_next_model(self):
        (data, model), calls = self.run_chain({
            "primary": self.raiser(503), "second": lambda: '{"ok": 1}', "third": lambda: '{"ok": 3}'})
        self.assertEqual((data, model), ({"ok": 1}, "second"))
        self.assertEqual(calls.count("primary"), 2, "a fast 503 gets exactly one retry")

    def test_429_and_malformed_json_fall_through(self):
        (data, model), _ = self.run_chain({
            "primary": self.raiser(429), "second": lambda: "Sorry, I cannot", "third": lambda: '```json\n{"ok": 3}\n```'})
        self.assertEqual((data, model), ({"ok": 3}, "third"))
        self.assertTrue(gemini_client._cooling_down("primary"), "rate-limited model must cool down")

    def test_slow_model_is_hedged_not_waited_for(self):
        def slow():
            time.sleep(15)
            return '{"ok": "slow"}'
        started = time.time()
        (data, model), _ = self.run_chain({"primary": slow, "second": lambda: '{"ok": 2}', "third": lambda: "{}"})
        self.assertEqual(model, "second")
        self.assertLess(time.time() - started, 6)

    def test_all_models_down_raises_clean_error(self):
        with self.assertRaises(GeminiError) as ctx:
            self.run_chain({m: self.raiser(503) for m in ("primary", "second", "third")})
        self.assertEqual(ctx.exception.kind, "unavailable")

    def test_invalid_key_stops_immediately(self):
        with self.assertRaises(GeminiError) as ctx:
            self.run_chain({"primary": self.raiser(403), "second": lambda: "{}", "third": lambda: "{}"})
        self.assertEqual(ctx.exception.kind, "auth")


class BiasResilience(unittest.TestCase):

    def analyzer(self):
        env = {"GEMINI_API_KEY": "AIza" + "b" * 35, "OPENROUTER_API_KEY": "your_openrouter_api_key_here",
               "AI_PROVIDER": "gemini", "BIAS_ANALYSIS_ENABLED": "true"}
        with mock.patch.dict(os.environ, env):
            return BiasAnalyzer()

    def test_gemini_503_is_a_failure_not_a_score(self):
        analyzer = self.analyzer()
        self.assertIsNone(analyzer.openrouter_key, "placeholder key must not be used as a fallback")
        with mock.patch.object(gemini_client, "generate_json",
                               side_effect=GeminiError("unavailable", "Gemini service unavailable (503)")):
            result = result_to_dict(analyzer.analyze_paper_sections(SECTIONS))
        self.assertEqual(result["status"], "failed")
        self.assertIn("503", result["error"])
        self.assertEqual(result["biases"], [])

    def test_malformed_gemini_json_is_a_failure(self):
        analyzer = self.analyzer()
        for payload in (["not", "an", "object"], {"unexpected": True}, {"biases": [], "overall_score": "n/a"}):
            with mock.patch.object(gemini_client, "generate_json", return_value=(payload, "stub-model")):
                result = result_to_dict(analyzer.analyze_paper_sections(SECTIONS))
            self.assertEqual(result["status"], "failed", payload)

    def test_valid_response_is_normalised(self):
        analyzer = self.analyzer()
        payload = {"overall_score": "140", "summary": "ok", "strengths": ["a", None, 3],
                   "biases": [{"type": "Selection Bias", "severity": "Medium", "excerpt": "x"}, "junk"]}
        with mock.patch.object(gemini_client, "generate_json", return_value=(payload, "stub-model")):
            result = result_to_dict(analyzer.analyze_paper_sections(SECTIONS))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["model"], "stub-model")
        self.assertEqual(len(result["biases"]), 1)
        self.assertEqual(result["biases"][0]["severity"], "moderate")
        self.assertEqual(result["strengths"], ["a", "3"])

    def test_no_key_disables_instead_of_crashing(self):
        env = {"GEMINI_API_KEY": "your_gemini_api_key_here", "OPENROUTER_API_KEY": ""}
        with mock.patch.dict(os.environ, env):
            analyzer = BiasAnalyzer()
        result = result_to_dict(analyzer.analyze_paper_sections(SECTIONS))
        self.assertEqual(result["status"], "disabled")


class PdfValidation(unittest.TestCase):

    def expect(self, data, code, status):
        with self.assertRaises(PDFExtractionError) as ctx:
            extract_text_from_bytes(data)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.status_code, status)

    def test_bad_uploads_get_specific_errors(self):
        self.expect(b"", "empty_file", 400)
        self.expect(b"just some text, not a pdf", "not_a_pdf", 415)
        self.expect(b"%PDF-1.7\n" + os.urandom(2000), "corrupted_pdf", 422)


class RunTogetherText(unittest.TestCase):

    def test_pdf_with_narrow_word_gaps_keeps_its_spaces(self):
        """Regression: pdfplumber glued words ('AttentionIsAllYouNeed'), ruining title and queries."""
        path = os.path.join(os.path.dirname(__file__), "sample_paper_tight_spacing.pdf")
        with open(path, "rb") as f:
            text, info = extract_text_from_bytes(f.read())
        self.assertTrue(text.startswith("Attention Is All You Need"), text[:60])
        self.assertGreater(info["words"], 300)
        sections = pc.extract_sections(text)
        self.assertEqual(sections["Title"], "Attention Is All You Need")
        self.assertEqual(pc.build_search_query(sections["Title"]), "attention is all you need")


class ApiPipeline(ProviderStubs):
    """End-to-end pipeline status handling (PDF parsing stubbed out)."""

    def run_pipeline(self, bias_side_effect=None, bias_return=None):
        import api
        api._result_cache.clear()
        analyzer = BiasResilience().analyzer()
        text = "\n".join(f"{k}\n{v}" for k, v in SECTIONS.items())
        with mock.patch.object(api, "bias_analyzer", analyzer), \
             mock.patch.object(api, "extract_text_from_bytes", return_value=(text, {"method": "stub"})), \
             mock.patch.object(api, "extract_sections", return_value=SECTIONS), \
             mock.patch.object(gemini_client, "generate_json", side_effect=bias_side_effect, return_value=bias_return):
            return api._analyze_pdf(os.urandom(64), "paper.pdf")

    GOOD_BIAS = ({"overall_score": 20, "summary": "fine", "biases": [], "strengths": ["clear"]}, "stub-model")
    DOWN = GeminiError("unavailable", "Gemini service unavailable (503)")

    def test_everything_ok(self):
        self.stub()
        out = self.run_pipeline(bias_return=self.GOOD_BIAS)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["warnings"], [])

    def test_bias_down_gives_partial_result_with_warning(self):
        self.stub()
        out = self.run_pipeline(bias_side_effect=self.DOWN)
        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["plagiarism"]["status"], "ok")
        self.assertEqual(out["bias_analysis"]["status"], "failed")
        self.assertTrue(any("Bias analysis failed" in w for w in out["warnings"]))

    def test_search_down_gives_partial_result_with_warning(self):
        self.stub(default=down_fetch)
        out = self.run_pipeline(bias_return=self.GOOD_BIAS)
        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["plagiarism"]["status"], "failed")
        self.assertEqual(out["bias_analysis"]["status"], "ok")

    def test_everything_down_is_an_http_503_not_a_fake_report(self):
        import api
        self.stub(default=down_fetch)
        with self.assertRaises(api.AnalysisError) as ctx:
            self.run_pipeline(bias_side_effect=self.DOWN)
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
