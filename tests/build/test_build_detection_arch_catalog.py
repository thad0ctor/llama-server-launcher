from modules.build import detection


def test_arch_catalog_names_current_embedded_and_dev_systems():
    expected = {
        "8.7": "Jetson AGX Orin",
        "11.0": "Jetson T5000",
        "12.1": "DGX Spark",
    }

    for cc, label in expected.items():
        arch = detection.known_arch_for(cc)
        assert arch is not None
        assert label in arch.name
