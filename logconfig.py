import logging

logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,format="%(asctime)s %(levelname)-8s[*] %(message)s" #日志输出的格式
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )