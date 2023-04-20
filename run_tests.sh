#! sh

pytest
lit -v tests/filecheck/ --order=smart
lit -v docs/Toy/examples --order=smart
