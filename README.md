# AudioDeepFakeDetection

Application that detects the originality of audio files with artificial intelligence.

## Setup Environment

```bash
# Make sure your PIP is up to date
pip install -U pip wheel setuptools

# Install required dependencies
pip install -r requirements.txt
```

## Setup Datasets

You may download the datasets used in the project from the following URL:

https://drive.google.com/file/d/1O_PckJtEbQWlHEMSA5gDdxRooa1S1N2p/view

-   (Real) Human Voice Dataset:
    -   This dataset consists of 10.000 short audio clips of a single speaker reading passages from 7 non-fiction books.
-   (Fake) Synthetic Voice Dataset: 
    -   The dataset consists of fake audio clips (16-bit PCM wav).

After downloading the datasets, you may extract them under `data/real` and `data/fake` respectively. In the end, the `data` directory should look like this:

```
data
├── real
    └── LJ001-0001
    └── LJ001-0002
    └── LJ001-0003
    └── LJ001-0004
    └── LJ001-0005
    └── LJ001-0006
    ...
└── fake
    └── LJ001-0001_gen
    └── LJ001-0002_gen
    └── LJ001-0003_gen
    └── LJ001-0004_gen
    └── LJ001-0005_gen
    └── LJ001-0006_gen
    ...
```


## License

Our project is licensed under the [MIT License].

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


