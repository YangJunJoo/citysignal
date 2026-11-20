"""Command entry points sharing the existing checked execution paths."""


def extract():
    from feature_extractor.process_local_atspm import main

    return main()


def pipeline():
    from run_pipeline import main

    return main()
