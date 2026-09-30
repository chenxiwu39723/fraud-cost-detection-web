"""gunicorn 部署設定。

啟動：
    gunicorn -c gunicorn.conf.py app:app
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
workers = 2          # 預測已於啟動載入記憶體，少量 worker 即可
threads = 2
timeout = 30
loglevel = "info"
