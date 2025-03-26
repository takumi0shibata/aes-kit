import logging
import sys

def setup_logger(name='aes_kit', level=logging.INFO):
    """基本的なロガーを設定する"""
    logger = logging.getLogger(name)
    if not logger.handlers: # ハンドラが既に追加されていないか確認
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # StreamHandler (コンソール出力)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # 必要に応じてFileHandlerも追加できる
        # fh = logging.FileHandler('aes_kit.log')
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

    return logger

logger = setup_logger()