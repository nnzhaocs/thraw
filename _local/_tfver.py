import transformers, os, sys
print("python:", sys.version)
print("transformers:", transformers.__version__)
print("path:", os.path.dirname(transformers.__file__))
