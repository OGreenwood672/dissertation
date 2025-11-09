python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
mv environment-rs/environment.cpython-312-x86_64-linux-gnu.so environment/environment.cpython-312-x86_64-linux-gnu.so
