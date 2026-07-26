"""Unit tests for the pure filename-collision helper in backend/admin/routes.py
(IMPROVEMENTS.md #5.3). No index, no filesystem -- _display_name_for_upload is a plain
regex derivation, same shape as backend/rag/generation.py's
_display_name_from_source_file and backend/rag/retrieval.py's get_program_catalog.
"""
from backend.admin.routes import _display_name_for_upload


class TestDisplayNameForUpload:
    def test_strips_trailing_year_and_underscores(self):
        assert _display_name_for_upload("Computer_Science_2026.pdf") == "Computer Science"

    def test_a_new_catalog_year_reduces_to_the_same_name(self):
        # The exact scenario IMPROVEMENTS.md #5.3 names: a newer catalog year uploaded
        # alongside the old one should be detected as the same program.
        assert (
            _display_name_for_upload("Computer_Science_2026.pdf")
            == _display_name_for_upload("Computer_Science_2027.pdf")
        )

    def test_no_year_suffix_is_left_unchanged_besides_underscores(self):
        assert _display_name_for_upload("Admission_Requirements.pdf") == "Admission Requirements"

    def test_unrelated_filenames_do_not_collide(self):
        assert (
            _display_name_for_upload("Computer_Science_2026.pdf")
            != _display_name_for_upload("Cyber_Security_2025.pdf")
        )
