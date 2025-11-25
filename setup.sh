pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Install the packages in r1-v .
cd src/r1-v
pip install -e ".[dev]"

cd ../qwen-vl-utils
pip install -e .

