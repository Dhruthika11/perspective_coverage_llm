for pkg in ['transformers', 'torch', 'spacy', 'nltk', 'scipy', 'sklearn', 'sentence_transformers']:
    try:
        m = __import__(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"{pkg}: NOT installed")
