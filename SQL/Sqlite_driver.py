# 导入:
from sqlalchemy import Column, String, create_engine, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


class Sqlite_driver():
    # 创建对象的基类:
    Base = declarative_base()
    # 定义User对象:
    class Word(Base):
        # 表的名字:
        __tablename__ = 'word'

        # 表的结构:
        sid = Column(Integer(), primary_key=True, autoincrement=True)
        char = Column(String(2))
        pid = Column(Integer())


    # 初始化数据库连接:
    engine = create_engine('sqlite:///Word.db')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine, checkfirst=True)

    def __init__(self) -> None:
        # 创建session对象:
        self.session = self.DBSession()
    
    def sid_is(self, sid):
        return {word.char: word.sid for word in self.session.query(self.Word).filter(self.Word.sid == sid).all()}

    # def char_is(self, char):
    #     return [word.char for word in self.session.query(self.Word).filter(self.Word.char == char).all()]

    def pid_is(self, pid):
        return {word.char: word.sid for word in self.session.query(self.Word).filter(self.Word.pid == pid).all()}
    
    def add(self, char, pid):
        word = self.Word(
            char = char,
            pid = pid
        )
        self.session.add(word)
        self.session.commit()
        return len(self.session.query(self.Word).all())
    
    # def __del__(self):
    #     self.session.close()
        



Sqlite_driver()