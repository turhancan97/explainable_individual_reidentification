import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_unseen_eval_metadata import UnseenSplitError, generate_unseen_split


class UnseenEvalMetadataTests(unittest.TestCase):
    def _write_source(self, root: Path, *, duplicate_query_content: bool = False) -> Path:
        image_dir = root / "images"
        image_dir.mkdir(parents=True)
        rows = [
            {"identity": "seen", "split": "train", "encounter": "0", "date": "01-01-2020", "path": "images/seen.jpg", "extra": "keep"},
            {"identity": "seen", "split": "test", "encounter": "1", "date": "02-01-2020", "path": "images/seen_test.jpg", "extra": "keep"},
            {"identity": "good", "split": "test", "encounter": "2", "date": "02-01-2020", "path": "images/good_gallery.jpg", "extra": "gallery"},
            {"identity": "good", "split": "test", "encounter": "3", "date": "03-01-2020", "path": "images/good_query.jpg", "extra": "query"},
            {"identity": "good", "split": "test", "encounter": "4", "date": "04-01-2020", "path": "images/good_query_2.jpg", "extra": "query"},
            {"identity": "one_group", "split": "test", "encounter": "9", "date": "09-01-2020", "path": "images/one_group.jpg", "extra": "excluded"},
        ]
        for row in rows:
            content = f"{row['identity']}-{row['encounter']}".encode()
            if duplicate_query_content and row["path"] == "images/good_query.jpg":
                content = b"good-2"
            (root / row["path"]).write_bytes(content)
        if duplicate_query_content:
            (root / "images/good_gallery.jpg").write_bytes(b"good-2")

        metadata = root / "source.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return metadata

    def _generate(self, root: Path, output_name: str = "out") -> dict:
        if not (root / "source.csv").exists():
            self._write_source(root)
        return generate_unseen_split(
            metadata=root / "source.csv",
            output_dir=root / output_name,
            label_col="identity",
            source_split_col="split",
            database_value="train",
            query_value="test",
            group_cols=("encounter",),
            order_cols=("date",),
            root=root,
        )

    def test_selects_query_only_identities_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._generate(root)
            with (root / "out" / "metadata_unseen_eval.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["identity"] for row in rows], ["good", "good", "good"])
            self.assertEqual([row["unseen_eval_split"] for row in rows], ["database", "query", "query"])
            self.assertEqual(rows[0]["extra"], "gallery")
            self.assertEqual(manifest["selected_identities"], ["good"])
            self.assertEqual(manifest["selected_counts"]["database_rows"], 1)
            self.assertEqual(manifest["selected_counts"]["query_rows"], 2)
            self.assertEqual(manifest["configuration"]["dataset"], root.name)
            self.assertEqual(manifest["excluded_identities"], [{"identity": "one_group", "num_groups": 1, "reason": "fewer_than_two_groups"}])
            self.assertTrue(manifest["validation"]["selected_identities_absent_from_source_database"])

    def test_output_order_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._generate(root, "first")
            self._generate(root, "second")
            first = (root / "first" / "metadata_unseen_eval.csv").read_bytes()
            second = (root / "second" / "metadata_unseen_eval.csv").read_bytes()
            self.assertEqual(first, second)

    def test_duplicate_content_across_sides_fails_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_source(root, duplicate_query_content=True)
            with self.assertRaisesRegex(UnseenSplitError, "duplicate image content"):
                self._generate(root)
            manifest = json.loads((root / "out" / "unseen_eval_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["validation"]["num_duplicate_content_hashes"], 1)
            self.assertFalse((root / "out" / "metadata_unseen_eval.csv").exists())

    def test_missing_file_fails_before_writing_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_source(root)
            (root / "images/good_query_2.jpg").unlink()
            with self.assertRaisesRegex(UnseenSplitError, "missing or unreadable"):
                self._generate(root)
            manifest = json.loads((root / "out" / "unseen_eval_manifest.json").read_text())
            self.assertEqual(len(manifest["validation"]["missing_or_unreadable_files"]), 1)

    def test_required_columns_and_order_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = self._write_source(root)
            text = metadata.read_text().replace(",date,", ",missing_date,")
            metadata.write_text(text)
            with self.assertRaisesRegex(UnseenSplitError, "required metadata columns"):
                self._generate(root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = self._write_source(root)
            text = metadata.read_text().replace("03-01-2020", "not-a-date")
            metadata.write_text(text)
            with self.assertRaisesRegex(UnseenSplitError, "not a supported date"):
                self._generate(root)

    def test_exact_path_overlap_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = self._write_source(root)
            text = metadata.read_text().replace("images/good_query.jpg", "images/good_gallery.jpg")
            metadata.write_text(text)
            with self.assertRaisesRegex(UnseenSplitError, "paths overlap"):
                self._generate(root)


if __name__ == "__main__":
    unittest.main()
