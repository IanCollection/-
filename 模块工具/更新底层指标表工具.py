import os

import pandas as pd
from dotenv import load_dotenv

from logic_folder.数据库表格 import Indicators, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pymysql

pymysql.install_as_MySQLdb()
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')


# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo)
    return engine


def new_create_db_session():
    # 创建数据库引擎
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine


##创建数据库和表格
def create_db_and_tables():
    engine = create_db_engine(True)
    Base.metadata.create_all(engine)
    return True


# class Indicators(Base):
#     __tablename__ = 'indicators'
#     id = Column(Integer, primary_key=True)
#     indicator_type = Column(String(255))
#     year = Column(Text)
#     code = Column(Text)
#     name = Column(Text)
#     explain = Column(Text)
#     weight_keywords = Column(Text)
#     similarity_keywords = Column(Text)
#     equality_question = Column(Text)
#     resource_keywords = Column(Text)
#     equation = Column(Text)
#     option = Column(Text)
#     without_table = Column(Integer)
#     direct_mission = Column(Integer)
#     mission_type = Column(Text)
#     table_only = Column(Integer)
#     allow_creation = Column(Integer)
#     min_similarity = Column(Float)
#     too_detail = Column(Integer)

def export_new_indicator_sheet():
    # # 根据旧excel的表头建一个新的df
    # data_columns = ['code', '基础层名称', '数据类型', 'equation', 'option', '指标描述', '信息是否可以直接获得',
    #                 '同义标签',
    #                 'info_tables', 'info_points', 'without_table', 'table_only', 'allow_creation', 'min_similarity',
    #                 'too_detail', 'resource', 'mission_type']
    # df = pd.DataFrame(columns=data_columns)
    session, engine = new_create_db_session()
    indicators = session.query(Indicators).all()
    df_data = []
    #做成一个list of dict ，然后转成df
    for indicator in indicators:
        # 写入row，需要修改info_tables ， info_points ，option 别的都不用改
        if indicator.similarity_keywords:
            info_points = ",".join(eval(indicator.similarity_keywords))
        else:
            info_points = ""
        if indicator.resource_keywords:
            info_tables = ",".join(eval(indicator.resource_keywords))
        else:
            info_tables = ""
        if indicator.option:
            option = ",".join(eval(indicator.option))
        else:
            option = ""

        # row = [indicator.code, indicator.name, indicator.indicator_type, indicator.equation, option, indicator.explain,
        #        indicator.direct_mission, indicator.equality_question, info_tables, info_points, indicator.without_table,
        #        indicator.table_only, indicator.allow_creation, indicator.min_similarity, indicator.too_detail,
        #        '', indicator.mission_type]

        row = { 'code': indicator.code, '基础层名称': indicator.name, '数据类型': indicator.indicator_type, 'equation': indicator.equation, 'option': option, '指标描述': indicator.explain,
               '信息是否可以直接获得': indicator.direct_mission, '同义标签': indicator.equality_question, 'info_tables': info_tables, 'info_points': info_points, 'without_table': indicator.without_table,
               'table_only': indicator.table_only, 'allow_creation': indicator.allow_creation, 'min_similarity': indicator.min_similarity, 'too_detail': indicator.too_detail,
               'resource': '', 'mission_type': indicator.mission_type}

        df_data.append(row)
    df = pd.DataFrame(df_data)
    #只要年月日
    time = pd.Timestamp.now().strftime('%Y-%m-%d')
    df.to_excel(f'{time}最新基础层指标表.xlsx', index=False)

    return True

if __name__ == '__main__':
    export_new_indicator_sheet()

