# 【Demo06】用_RAG_打造心靈處方籤機器人

*This notebook was created for Google Colab*

*Language: python*

---



```python
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


### 1. 讀入需要的套件

這裡主要用 `LangChain`, 這可以說整合各式 LLM 功能的方便套件。



```python
!pip install -U nltk
```

**Output:**
```
Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.8.1)
Collecting nltk
  Downloading nltk-3.9.1-py3-none-any.whl.metadata (2.9 kB)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)
Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2024.9.11)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.5)
Downloading nltk-3.9.1-py3-none-any.whl (1.5 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m29.2 MB/s[0m eta [36m0:00:00[0m
[?25hInstalling collected packages: nltk
  Attempting uninstall: nltk
    Found existing installation: nltk 3.8.1
    Uninstalling nltk-3.8.1:
      Successfully uninstalled nltk-3.8.1
Successfully installed nltk-3.9.1

```



```python
!pip install langchain langchain-community openai faiss-cpu unstructured tiktoken
```

**Output:**
```
Requirement already satisfied: langchain in /usr/local/lib/python3.10/dist-packages (0.3.4)
Collecting langchain-community
  Downloading langchain_community-0.3.4-py3-none-any.whl.metadata (2.9 kB)
Requirement already satisfied: openai in /usr/local/lib/python3.10/dist-packages (1.52.2)
Collecting faiss-cpu
  Downloading faiss_cpu-1.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.4 kB)
Collecting unstructured
  Downloading unstructured-0.16.3-py3-none-any.whl.metadata (24 kB)
Collecting tiktoken
  Downloading tiktoken-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (6.0.2)
Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.0.36)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (3.10.10)
Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (4.0.3)
Requirement already satisfied: langchain-core<0.4.0,>=0.3.12 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.3.13)
Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.3.0)
Requirement already satisfied: langsmith<0.2.0,>=0.1.17 in /usr/local/lib/python3.10/dist-packages (from langchain) (0.1.137)
Requirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.26.4)
Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.9.2)
Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.32.3)
Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (9.0.0)
Collecting dataclasses-json<0.7,>=0.5.7 (from langchain-community)
  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
Collecting httpx-sse<0.5.0,>=0.4.0 (from langchain-community)
  Downloading httpx_sse-0.4.0-py3-none-any.whl.metadata (9.0 kB)
Collecting langchain
  Downloading langchain-0.3.6-py3-none-any.whl.metadata (7.1 kB)
Collecting langchain-core<0.4.0,>=0.3.12 (from langchain)
  Downloading langchain_core-0.3.14-py3-none-any.whl.metadata (6.3 kB)
Collecting pydantic-settings<3.0.0,>=2.4.0 (from langchain-community)
  Downloading pydantic_settings-2.6.0-py3-none-any.whl.metadata (3.5 kB)
Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai) (3.7.1)
Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.10/dist-packages (from openai) (1.9.0)
Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from openai) (0.27.2)
Requirement already satisfied: jiter<1,>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from openai) (0.6.1)
Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai) (1.3.1)
Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.5)
Requirement already satisfied: typing-extensions<5,>=4.11 in /usr/local/lib/python3.10/dist-packages (from openai) (4.12.2)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from faiss-cpu) (24.1)
Requirement already satisfied: chardet in /usr/local/lib/python3.10/dist-packages (from unstructured) (5.2.0)
Collecting filetype (from unstructured)
  Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)
Collecting python-magic (from unstructured)
  Downloading python_magic-0.4.27-py2.py3-none-any.whl.metadata (5.8 kB)
Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from unstructured) (4.9.4)
Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (from unstructured) (3.9.1)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from unstructured) (4.12.3)
Collecting emoji (from unstructured)
  Downloading emoji-2.14.0-py3-none-any.whl.metadata (5.7 kB)
Collecting python-iso639 (from unstructured)
  Downloading python_iso639-2024.10.22-py3-none-any.whl.metadata (13 kB)
Collecting langdetect (from unstructured)
  Downloading langdetect-1.0.9.tar.gz (981 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m981.5/981.5 kB[0m [31m26.9 MB/s[0m eta [36m0:00:00[0m
[?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
Collecting rapidfuzz (from unstructured)
  Downloading rapidfuzz-3.10.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Collecting backoff (from unstructured)
  Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
Collecting unstructured-client (from unstructured)
  Downloading unstructured_client-0.26.2-py3-none-any.whl.metadata (20 kB)
Requirement already satisfied: wrapt in /usr/local/lib/python3.10/dist-packages (from unstructured) (1.16.0)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from unstructured) (5.9.5)
Collecting python-oxmsg (from unstructured)
  Downloading python_oxmsg-0.0.1-py3-none-any.whl.metadata (5.0 kB)
Requirement already satisfied: html5lib in /usr/local/lib/python3.10/dist-packages (from unstructured) (1.1)
Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2024.9.11)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.3)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (24.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.5.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
Requirement already satisfied: yarl<2.0,>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.16.0)
Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (3.10)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (1.2.2)
Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community)
  Downloading marshmallow-3.23.0-py3-none-any.whl.metadata (7.6 kB)
Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community)
  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (2024.8.30)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.6)
Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
Requirement already satisfied: jsonpatch<2.0,>=1.33 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.4.0,>=0.3.12->langchain) (1.33)
Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /usr/local/lib/python3.10/dist-packages (from langsmith<0.2.0,>=0.1.17->langchain) (3.10.10)
Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from langsmith<0.2.0,>=0.1.17->langchain) (1.0.0)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.7.0)
Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.23.4)
Collecting python-dotenv>=0.21.0 (from pydantic-settings<3.0.0,>=2.4.0->langchain-community)
  Downloading python_dotenv-1.0.1-py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (3.4.0)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (2.2.3)
Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (3.1.1)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->unstructured) (2.6)
Requirement already satisfied: six>=1.9 in /usr/local/lib/python3.10/dist-packages (from html5lib->unstructured) (1.16.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from html5lib->unstructured) (0.5.1)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk->unstructured) (8.1.7)
Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk->unstructured) (1.4.2)
Collecting olefile (from python-oxmsg->unstructured)
  Downloading olefile-0.47-py2.py3-none-any.whl.metadata (9.7 kB)
Requirement already satisfied: cryptography>=3.1 in /usr/local/lib/python3.10/dist-packages (from unstructured-client->unstructured) (43.0.3)
Requirement already satisfied: eval-type-backport<0.3.0,>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from unstructured-client->unstructured) (0.2.0)
Collecting jsonpath-python<2.0.0,>=1.0.6 (from unstructured-client->unstructured)
  Downloading jsonpath_python-1.0.6-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: nest-asyncio>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from unstructured-client->unstructured) (1.6.0)
Collecting pypdf>=4.0 (from unstructured-client->unstructured)
  Downloading pypdf-5.1.0-py3-none-any.whl.metadata (7.2 kB)
Requirement already satisfied: python-dateutil==2.8.2 in /usr/local/lib/python3.10/dist-packages (from unstructured-client->unstructured) (2.8.2)
Requirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.10/dist-packages (from cryptography>=3.1->unstructured-client->unstructured) (1.17.1)
Requirement already satisfied: jsonpointer>=1.9 in /usr/local/lib/python3.10/dist-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.12->langchain) (3.0.0)
Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain-community)
  Downloading mypy_extensions-1.0.0-py3-none-any.whl.metadata (1.1 kB)
Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from yarl<2.0,>=1.12.0->aiohttp<4.0.0,>=3.8.3->langchain) (0.2.0)
Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.12->cryptography>=3.1->unstructured-client->unstructured) (2.22)
Downloading langchain_community-0.3.4-py3-none-any.whl (2.4 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m50.0 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading langchain-0.3.6-py3-none-any.whl (1.0 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m34.5 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading faiss_cpu-1.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (27.5 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.5/27.5 MB[0m [31m24.7 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading unstructured-0.16.3-py3-none-any.whl (1.7 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m43.0 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading tiktoken-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m45.3 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
Downloading httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)
Downloading langchain_core-0.3.14-py3-none-any.whl (408 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m408.7/408.7 kB[0m [31m27.6 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading pydantic_settings-2.6.0-py3-none-any.whl (28 kB)
Downloading backoff-2.2.1-py3-none-any.whl (15 kB)
Downloading emoji-2.14.0-py3-none-any.whl (586 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m586.9/586.9 kB[0m [31m33.2 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)
Downloading python_iso639-2024.10.22-py3-none-any.whl (274 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m274.9/274.9 kB[0m [31m20.5 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading python_magic-0.4.27-py2.py3-none-any.whl (13 kB)
Downloading python_oxmsg-0.0.1-py3-none-any.whl (31 kB)
Downloading rapidfuzz-3.10.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m48.8 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading unstructured_client-0.26.2-py3-none-any.whl (59 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.0/60.0 kB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading jsonpath_python-1.0.6-py3-none-any.whl (7.6 kB)
Downloading marshmallow-3.23.0-py3-none-any.whl (49 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.5/49.5 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading pypdf-5.1.0-py3-none-any.whl (297 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m298.0/298.0 kB[0m [31m17.6 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
Downloading olefile-0.47-py2.py3-none-any.whl (114 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.6/114.6 kB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
Building wheels for collected packages: langdetect
  Building wheel for langdetect (setup.py) ... [?25l[?25hdone
  Created wheel for langdetect: filename=langdetect-1.0.9-py3-none-any.whl size=993222 sha256=3a478d63f804154d14245d90b4a6b76473f4f686d41d3227d2ac883a833083e2
  Stored in directory: /root/.cache/pip/wheels/95/03/7d/59ea870c70ce4e5a370638b5462a7711ab78fba2f655d05106
Successfully built langdetect
Installing collected packages: filetype, rapidfuzz, python-magic, python-iso639, python-dotenv, pypdf, olefile, mypy-extensions, marshmallow, langdetect, jsonpath-python, httpx-sse, faiss-cpu, emoji, backoff, typing-inspect, tiktoken, python-oxmsg, unstructured-client, pydantic-settings, dataclasses-json, unstructured, langchain-core, langchain, langchain-community
  Attempting uninstall: langchain-core
    Found existing installation: langchain-core 0.3.13
    Uninstalling langchain-core-0.3.13:
      Successfully uninstalled langchain-core-0.3.13
  Attempting uninstall: langchain
    Found existing installation: langchain 0.3.4
    Uninstalling langchain-0.3.4:
      Successfully uninstalled langchain-0.3.4
Successfully installed backoff-2.2.1 dataclasses-json-0.6.7 emoji-2.14.0 faiss-cpu-1.9.0 filetype-1.2.0 httpx-sse-0.4.0 jsonpath-python-1.0.6 langchain-0.3.6 langchain-community-0.3.4 langchain-core-0.3.14 langdetect-1.0.9 marshmallow-3.23.0 mypy-extensions-1.0.0 olefile-0.47 pydantic-settings-2.6.0 pypdf-5.1.0 python-dotenv-1.0.1 python-iso639-2024.10.22 python-magic-0.4.27 python-oxmsg-0.0.1 rapidfuzz-3.10.1 tiktoken-0.8.0 typing-inspect-0.9.0 unstructured-0.16.3 unstructured-client-0.26.2

```


讀入正確的 `nltk` 所需資料。



```python
import nltk
```



```python
nltk.data.path.append("/root/nltk_data")
nltk.download('punkt')
```

**Output:**
```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.

True
```


讀入一大票需要的函式。



```python
import os
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
```


### 2. 讀入範例資料

我們這裡用聖嚴法師的《真正的快樂》一書為範例, 當然其實可以用更多的資料, 直接放入相對的資料夾 (這裡是設在 `books` 之下) 即可。包括這本書都在[《法鼓全集》](https://ddc.shengyen.org/)之下, 請注意版權屬「法鼓文化」所有, 我們只是作為範例。



```python
# 下載 books.zip 檔案
!wget -O books.zip https://github.com/yenlung/AI-Demo/raw/refs/heads/master/books.zip

# 解壓縮 books.zip 到 books 資料夾
!unzip -o books.zip
```

**Output:**
```
--2024-10-31 08:40:25--  https://github.com/yenlung/AI-Demo/raw/refs/heads/master/books.zip
Resolving github.com (github.com)... 140.82.121.4
Connecting to github.com (github.com)|140.82.121.4|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://raw.githubusercontent.com/yenlung/AI-Demo/refs/heads/master/books.zip [following]
--2024-10-31 08:40:25--  https://raw.githubusercontent.com/yenlung/AI-Demo/refs/heads/master/books.zip
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 63072 (62K) [application/zip]
Saving to: ‘books.zip’

books.zip           100%[===================>]  61.59K  --.-KB/s    in 0.007s  

2024-10-31 08:40:26 (9.02 MB/s) - ‘books.zip’ saved [63072/63072]

Archive:  books.zip
   creating: books/
  inflating: books/book1.txt         
  inflating: __MACOSX/books/._book1.txt  

```


### 3. 設定 OpenAI 金鑰



```python
from getpass import getpass
```



```python
api_key = getpass("請輸入您的 OpenAI API key: ")
```

**Output:**
```
請輸入您的 OpenAI API key: ··········

```



```python
# from google.colab import userdata
# # 將 OpenAI API Key 記在 colab 當中
# api_key = userdata.get('keyFor108')
```



```python
os.environ["OPENAI_API_KEY"] = api_key
```


### 4. 建立向量資料庫

#### Step 1: 加載資料夾中的文件



```python
loader = DirectoryLoader("books", glob="*.txt")  # 替換為你的資料夾路徑
documents = loader.load()
```


#### Step 2: 將文件分割成較小的片段



```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)
```


#### Step 3: 使用 OpenAI 的嵌入來將文件轉為向量嵌入



```python
embeddings = OpenAIEmbeddings()
```

**Output:**
```
<ipython-input-13-73ad2f8e367a>:1: LangChainDeprecationWarning: The class `OpenAIEmbeddings` was deprecated in LangChain 0.0.9 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import OpenAIEmbeddings``.
  embeddings = OpenAIEmbeddings()

```


#### Step 4: 使用 FAISS 建立向量資料庫



```python
vector_store = FAISS.from_documents(split_docs, embeddings)
```


#### Step 5: 建立檢索器



```python
retriever = vector_store.as_retriever()
```


### 5. 打造心靈處方籤機器人


#### 選定語言模型



```python
llm = ChatOpenAI(model="gpt-4o")
```

**Output:**
```
<ipython-input-16-e8ae6cf3228a>:1: LangChainDeprecationWarning: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import ChatOpenAI``.
  llm = ChatOpenAI(model="gpt-4o")

```


#### 定義一些心靈處方籤



```python
spiritual_prescriptions = [
    "感謝給我們機會，順境、逆境，皆是恩人。",
    "身心常放鬆，逢人面帶笑；放鬆能使我們身心健康，帶笑容易增進彼此友誼。",
    "識人識己識進退，時時身心平安；知福惜福多培福，處處廣結善緣。",
    "平常心就是最自在、最愉快的心。",
    "知道自己的缺點愈多，成長的速度愈快，對自己的信心也就愈堅定。"
]
```


#### 建立一個結合檢索與生成的 RAG 問答鏈



```python
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```


#### 定義真正的心靈處方籤主函式

注意最主要還是設計 `prompt` 的型式。



```python
def answer_user_question(question):
    # 抽取一條隨機的心靈處方籤
    chosen_prescription = np.random.choice(spiritual_prescriptions)

    # 檢索資料夾中的相關內容
    retriever_result = qa_chain.run(question)

    print(f"你抽到的心靈處方籤: {chosen_prescription}")

    # 自訂 prompt，結合心靈處方籤、上下文和使用者問題
    prompt = f"""
    使用者抽到了一個心靈處方籤，它的內容是：{chosen_prescription}

    以下是我們從資料庫中檢索到的內容，這些內容來自書中的資料，並與使用者的問題相關：
    {retriever_result}

    請根據「心靈處方籤」中的訊息，結合書中的資料，用類似的語氣和觀念來回應使用者的問題：
    「{question}」
    """

    # 使用 HumanMessage 包裝 prompt 並生成回答
    final_response = llm.invoke([HumanMessage(content=prompt)])

    return final_response.content
```



```python
# 使用範例：回答一個使用者問題
user_question = "今天有颱風好可怕，該如何面對？"
response = answer_user_question(user_question)

print(f'\n經過機器人得到的內容是 \n==================== \n{response}')
```

**Output:**
```
<ipython-input-19-c147d0f04a9d>:6: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  retriever_result = qa_chain.run(question)

你抽到的心靈處方籤: 平常心就是最自在、最愉快的心。

經過機器人得到的內容是 
==================== 
面對颱風這樣的自然災害，我們可以從「平常心就是最自在、最愉快的心」中獲得啟發。保持平常心，讓自己能冷靜思考和有效行動，這樣可以更自在地應對挑戰。以下是一些建議來幫助您面對颱風：

1. **提前準備**：在颱風來臨前做好準備，將物品妥善收納，這樣可以讓您在面對風雨時更加安心。

2. **保持冷靜**：以平常心應對颱風，讓自己保持冷靜和沉著，這樣您能夠更清晰地思考和做決策。

3. **依賴信仰的力量**：如果您有信仰，可以通過禱告或誦念經文來獲得心靈的平和，這樣能讓您在風雨中保持穩定。

4. **接受現實**：有時候損害是無法避免的，接受現實並專注於災後的處理工作，這樣能讓您更有效地面對挑戰。

5. **遵循安全指引**：確保您和家人的安全是最重要的，聽從政府的指引，這樣能讓您更有信心地度過颱風。

記住，颱風過後總會有晴天，保持希望和信心，讓自己以平常心迎接未來的每一天。

```


# 建立 Gradio 互動介面



```python
!pip install gradio
import gradio as gr
```

**Output:**
```
Collecting gradio
  Downloading gradio-5.4.0-py3-none-any.whl.metadata (16 kB)
Collecting aiofiles<24.0,>=22.0 (from gradio)
  Downloading aiofiles-23.2.1-py3-none-any.whl.metadata (9.7 kB)
Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.7.1)
Collecting fastapi<1.0,>=0.115.2 (from gradio)
  Downloading fastapi-0.115.4-py3-none-any.whl.metadata (27 kB)
Collecting ffmpy (from gradio)
  Downloading ffmpy-0.4.0-py3-none-any.whl.metadata (2.9 kB)
Collecting gradio-client==1.4.2 (from gradio)
  Downloading gradio_client-1.4.2-py3-none-any.whl.metadata (7.1 kB)
Requirement already satisfied: httpx>=0.24.1 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.27.2)
Collecting huggingface-hub>=0.25.1 (from gradio)
  Downloading huggingface_hub-0.26.2-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: jinja2<4.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.1.4)
Collecting markupsafe~=2.0 (from gradio)
  Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
Requirement already satisfied: numpy<3.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (1.26.4)
Requirement already satisfied: orjson~=3.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.10.10)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gradio) (24.1)
Requirement already satisfied: pandas<3.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.2.2)
Requirement already satisfied: pillow<12.0,>=8.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (10.4.0)
Requirement already satisfied: pydantic>=2.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.9.2)
Collecting pydub (from gradio)
  Downloading pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting python-multipart==0.0.12 (from gradio)
  Downloading python_multipart-0.0.12-py3-none-any.whl.metadata (1.9 kB)
Requirement already satisfied: pyyaml<7.0,>=5.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (6.0.2)
Collecting ruff>=0.2.2 (from gradio)
  Downloading ruff-0.7.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (25 kB)
Collecting safehttpx<1.0,>=0.1.1 (from gradio)
  Downloading safehttpx-0.1.1-py3-none-any.whl.metadata (4.1 kB)
Collecting semantic-version~=2.0 (from gradio)
  Downloading semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)
Collecting starlette<1.0,>=0.40.0 (from gradio)
  Downloading starlette-0.41.2-py3-none-any.whl.metadata (6.0 kB)
Collecting tomlkit==0.12.0 (from gradio)
  Downloading tomlkit-0.12.0-py3-none-any.whl.metadata (2.7 kB)
Requirement already satisfied: typer<1.0,>=0.12 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.12.5)
Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (4.12.2)
Collecting uvicorn>=0.14.0 (from gradio)
  Downloading uvicorn-0.32.0-py3-none-any.whl.metadata (6.6 kB)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from gradio-client==1.4.2->gradio) (2024.6.1)
Collecting websockets<13.0,>=10.0 (from gradio-client==1.4.2->gradio)
  Downloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (3.10)
Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (1.2.2)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx>=0.24.1->gradio) (2024.8.30)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx>=0.24.1->gradio) (1.0.6)
Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.25.1->gradio) (3.16.1)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.25.1->gradio) (2.32.3)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.25.1->gradio) (4.66.5)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2024.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2024.2)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic>=2.0->gradio) (0.7.0)
Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic>=2.0->gradio) (2.23.4)
Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (8.1.7)
Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (1.5.4)
Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (13.9.3)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3.0,>=1.0->gradio) (1.16.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.18.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.25.1->gradio) (3.4.0)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.25.1->gradio) (2.2.3)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.2)
Downloading gradio-5.4.0-py3-none-any.whl (56.7 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.7/56.7 MB[0m [31m12.6 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading gradio_client-1.4.2-py3-none-any.whl (319 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m319.8/319.8 kB[0m [31m21.7 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading python_multipart-0.0.12-py3-none-any.whl (23 kB)
Downloading tomlkit-0.12.0-py3-none-any.whl (37 kB)
Downloading aiofiles-23.2.1-py3-none-any.whl (15 kB)
Downloading fastapi-0.115.4-py3-none-any.whl (94 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m94.7/94.7 kB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading huggingface_hub-0.26.2-py3-none-any.whl (447 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m447.5/447.5 kB[0m [31m30.9 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
Downloading ruff-0.7.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.0 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.0/11.0 MB[0m [31m98.6 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading safehttpx-0.1.1-py3-none-any.whl (8.4 kB)
Downloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
Downloading starlette-0.41.2-py3-none-any.whl (73 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m73.3/73.3 kB[0m [31m5.7 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading uvicorn-0.32.0-py3-none-any.whl (63 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.7/63.7 kB[0m [31m5.1 MB/s[0m eta [36m0:00:00[0m
[?25hDownloading ffmpy-0.4.0-py3-none-any.whl (5.8 kB)
Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)
Downloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.2/130.2 kB[0m [31m9.9 MB/s[0m eta [36m0:00:00[0m
[?25hInstalling collected packages: pydub, websockets, uvicorn, tomlkit, semantic-version, ruff, python-multipart, markupsafe, ffmpy, aiofiles, starlette, huggingface-hub, safehttpx, gradio-client, fastapi, gradio
  Attempting uninstall: markupsafe
    Found existing installation: MarkupSafe 3.0.2
    Uninstalling MarkupSafe-3.0.2:
      Successfully uninstalled MarkupSafe-3.0.2
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface-hub 0.24.7
    Uninstalling huggingface-hub-0.24.7:
      Successfully uninstalled huggingface-hub-0.24.7
Successfully installed aiofiles-23.2.1 fastapi-0.115.4 ffmpy-0.4.0 gradio-5.4.0 gradio-client-1.4.2 huggingface-hub-0.26.2 markupsafe-2.1.5 pydub-0.25.1 python-multipart-0.0.12 ruff-0.7.1 safehttpx-0.1.1 semantic-version-2.10.0 starlette-0.41.2 tomlkit-0.12.0 uvicorn-0.32.0 websockets-12.0

```



```python
def answer_user_question(question):
    # 抽取一條隨機的心靈處方籤
    chosen_prescription = np.random.choice(spiritual_prescriptions)

    # 檢索資料夾中的相關內容
    retriever_result = qa_chain.run(question)

    # 自訂 prompt，結合心靈處方籤、上下文和使用者問題
    prompt = f"""
    使用者抽到了一個心靈處方籤，它的內容是：{chosen_prescription}

    以下是我們從資料庫中檢索到的內容，這些內容來自書中的資料，並與使用者的問題相關：
    {retriever_result}

    請根據「心靈處方籤」中的訊息，結合書中的資料，用類似的語氣和觀念來回應使用者的問題：
    「{question}」
    """

    # 使用 HumanMessage 包裝 prompt 並生成回答
    final_response = llm.invoke([HumanMessage(content=prompt)])

    return chosen_prescription, final_response.content
```



```python
title = "【拍拍機器人】AI 心靈處方籤"
description = "請你注意自己的呼吸一分鐘, 寫下你心裡浮現的問題。發送之後會幫你抽出一支心靈處方籤, 還會幫你解籤 :)"
```



```python
inp = gr.Textbox(label="請寫下你的問題:")
out1 = gr.Textbox(label="你抽到的心靈處方籤")
out2 = gr.Textbox(label="拍拍想跟你說的話")
```



```python
iface = gr.Interface(answer_user_question,
                     title=title,
                     description=description,
                     inputs = inp,
                     outputs = [out1, out2])
```



```python
iface.launch(share=True, debug=True)
```

**Output:**
```
Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().
* Running on public URL: https://d5d8781d2e22b36077.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)

<IPython.core.display.HTML object>
```
