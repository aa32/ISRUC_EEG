import unittest
from sleep_scoring.features.edf_loader import PhysiobankEDFLoader


class PhysiobankEDFLoaderTest(unittest.TestCase):

    def setUp(self):
        self.loader = PhysiobankEDFLoader()

    def test_psg_record_paths(self):
        self.assertEqual(len(self.loader.psg_record_paths), 197)

    def test_load_sc_records(self):
        records = self.loader.load_sc_records(save=False)
        self.assertEqual(len(records), 153)


if __name__ == '__main__':
    unittest.main()