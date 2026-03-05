from sqlalchemy import create_engine, Column, String, Text, DateTime, func, Integer, Float, LargeBinary, ForeignKey, \
    UniqueConstraint, VARCHAR, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, deferred
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.types import TypeDecorator, Text

# 创建 SQLAlchemy 基类
Base = declarative_base()


class LongText(TypeDecorator):
    impl = Text

    def load_dialect_impl(self, dialect):
        if dialect.name == 'mysql':
            return dialect.type_descriptor(LONGTEXT())
        else:
            return dialect.type_descriptor(Text)


class Reports(Base):
    __tablename__ = 'reports'
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('company.id'))
    have_knowledge_graph = Column(Integer)
    in_db = Column(Integer)
    in_progress = Column(Integer)
    company_name = Column(Text)
    report_name = Column(Text)
    year = Column(Text)
    sections = relationship('Section', back_populates='report', lazy='dynamic')
    company = relationship('Company', back_populates='reports')
    person_name = Column(Text)

class Indicators(Base):
    __tablename__ = 'indicators'
    id = Column(Integer, primary_key=True)
    indicator_type = Column(String(255))
    year = Column(Text)
    code = Column(Text)
    name = Column(Text)
    explain = Column(Text)
    weight_keywords = Column(Text)
    necessary_points = Column(Text)
    similarity_keywords = Column(Text)
    equality_question = Column(Text)
    resource_keywords = Column(Text)
    missing_fill = Column(Text)
    section_keywords = Column(Text)
    equation = Column(Text)
    option = Column(Text)
    without_table = Column(Integer)
    direct_mission = Column(Integer)
    mission_type = Column(Text)
    table_only = Column(Integer)
    allow_creation = Column(Integer)
    min_similarity = Column(Float)
    too_detail = Column(Integer)
    format_type =Column(Text)
    positive_example = Column(Text)
    positive_example_reason = Column(Text)
    negative_example = Column(Text)
    negative_example_reason = Column(Text)
    pre_condition_indicator = Column(Text)
    execute_level = Column(Integer)
    execute_section = Column(Integer)
    master_indicator_id = Column(Integer, ForeignKey('indicators.id'))
    master_indicator = relationship("Indicators", back_populates="sub_indicators", remote_side=[id])
    sub_indicators = relationship('Indicators', back_populates='master_indicator')
    indicator_vectors = relationship('Vector', back_populates='indicator')


class IndicatorsResults(Base):
    __tablename__ = 'indicators_result'
    id = Column(Integer, primary_key=True)
    indicator_type = Column(String(255))
    company_id = Column(Integer, ForeignKey('company.id'))
    company_name = Column(Text)
    company_code = Column(Text)
    report_name = Column(Text)
    year = Column(Text)
    code = Column(Text)
    name = Column(Text)
    quit = Column(Integer)
    person = Column(Text)
    reference_length = Column(Text)
    reference = Column(LongText)
    table_reference = Column(LongText)
    original_answer = Column(Text)
    value = Column(Text)
    source = Column(Text)
    assumption = Column(Integer)
    tool_use = Column(Text)
    cost = Column(Text)


class Company(Base):
    __tablename__ = 'company'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    company_code = Column(Text)
    industry = Column(Text)
    reports = relationship('Reports', back_populates='company', lazy='dynamic')


class Section(Base):
    __tablename__ = 'sections'
    id = Column(Integer, primary_key=True)
    title = Column(Text)
    full_sentence = Column(Text)
    page = Column(Text)
    report_id = Column(Integer, ForeignKey('reports.id'))
    section_level = Column(Text)
    master_section_id = Column(Integer, ForeignKey('sections.id'))
    master_section = relationship("Section", back_populates="sub_sections", remote_side=[id])
    sub_sections = relationship('Section', back_populates='master_section', lazy='dynamic')
    charts = relationship('Chart', back_populates='section', lazy='dynamic')
    sentences = relationship('Sentences', back_populates='section', lazy='dynamic')
    report = relationship('Reports', back_populates='sections')


class Sentences(Base):
    __tablename__ = 'sentences'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    section_id = Column(Integer, ForeignKey('sections.id'))
    is_title = Column(Text)
    page = Column(Text)
    report_id = Column(Integer)
    section = relationship('Section', back_populates='sentences')
    word = relationship('Word', back_populates='sentence', lazy='dynamic')


class Word(Base):
    __tablename__ = 'word'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    sentence_id = Column(Integer, ForeignKey('sentences.id'))
    sentence = relationship('Sentences', back_populates='word')


class Chart(Base):
    __tablename__ = 'charts'
    id = Column(Integer, primary_key=True)
    title = Column(Text)
    unit = Column(String(255))
    full_header = Column(LongText)
    full_key_index = Column(LongText)
    full_value = Column(LongText)
    section_id = Column(Integer, ForeignKey('sections.id'))
    file_name = Column(Text)
    section = relationship('Section', back_populates='charts')
    headers = relationship('Header', back_populates='chart', lazy='dynamic')
    key_index = relationship('KeyIndex', back_populates='chart', lazy='dynamic')
    description = relationship('Description', back_populates='chart', lazy='dynamic')
    page = Column(Text)
    potential_table_titles = Column(LongText)


class Header(Base):
    __tablename__ = 'headers'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    chart_id = Column(Integer, ForeignKey('charts.id'))
    chart = relationship('Chart', back_populates='headers')


class KeyIndex(Base):
    __tablename__ = 'key_index'
    id = Column(Integer, primary_key=True)
    name = Column(Text)
    chart_id = Column(Integer, ForeignKey('charts.id'))
    chart = relationship('Chart', back_populates='key_index')
    table_value = relationship('TableValue', back_populates='key_index', lazy='dynamic')


class TableValue(Base):
    __tablename__ = 'table_value'
    id = Column(Integer, primary_key=True)
    value = Column(Text)
    key_index_id = Column(Integer, ForeignKey('key_index.id'))
    key_index = relationship('KeyIndex', back_populates='table_value')


class Description(Base):
    __tablename__ = 'description'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    location = Column(Text)
    chart_id = Column(Integer, ForeignKey('charts.id'))
    chart = relationship('Chart', back_populates='description')


class Vector(Base):
    __tablename__ = 'vector'
    __table_args__ = {'mysql_charset': 'utf8'}
    id = Column(Integer, primary_key=True)
    vector = deferred(Column(LargeBinary))  # or any binary type to store actual vector data
    type = Column(String(255))
    level = Column(Text)
    text = Column(LongText)
    link = Column(Integer)
    belongs_to_table = Column(Integer)
    is_table_title = Column(Integer)
    is_similarity_keywords_vector = Column(Integer)
    is_resource_keywords_vector = Column(Integer)
    is_section_keywords_vector = Column(Integer)
    ## linke with header, key_index, description, sentence, table, section
    # Foreign Key References
    header_id = Column(Integer, ForeignKey('headers.id'), nullable=True)
    key_index_id = Column(Integer, ForeignKey('key_index.id'), nullable=True)
    description_id = Column(Integer, ForeignKey('description.id'), nullable=True)
    sentence_id = Column(Integer, ForeignKey('sentences.id'), nullable=True)
    chart_id = Column(Integer, ForeignKey('charts.id'), nullable=True)
    section_id = Column(Text)
    company_id = Column(Integer, ForeignKey('company.id'), nullable=True)
    report_id = Column(Integer, ForeignKey('reports.id'), nullable=True)
    table_value_id = Column(Integer, ForeignKey('table_value.id'), nullable=True)
    word_id = Column(Integer, ForeignKey('word.id'), nullable=True)
    indicator_id = Column(Integer, ForeignKey('indicators.id', ondelete='CASCADE'), nullable=True)
    # Relationships
    header = relationship('Header', backref='vector')
    key_index = relationship('KeyIndex', backref='vector')
    description = relationship('Description', backref='vector')
    sentence = relationship('Sentences', backref='vector')
    chart = relationship('Chart', backref='vector')
    # section = relationship('Section', backref='vector')
    company = relationship('Company', backref='vector')
    report = relationship('Reports', backref='vector')
    table_value = relationship('TableValue', backref='vector')
    word = relationship('Word', backref='vector')
    have_embedding = Column(Integer)
    indicator = relationship('Indicators', back_populates='indicator_vectors')

    Index('idx_report_id', 'report_id')
    Index('idx_company_id', 'company_id')
    Index('idx_have_embedding', 'have_embedding')


class DataRecord(Base):
    __tablename__ = "data_record"
    id = Column(Integer, primary_key=True)
    question_type = Column(Text)
    question = Column(LongText)
    necessary_infos = Column(Text)
    indicator = Column(Text)
    answer = Column(Text)


class Statistic(Base):
    __tablename__ = "statistic"

    date = Column(String(100), primary_key=True)
    report_cost = Column(Float)
    report_num = Column(Integer)
    indicator_cost = Column(Float)
    indicator_num = Column(Integer)
    probe_type = Column(Integer)
    probe_type_cost = Column(Float)
    summary_type = Column(Integer)
    summary_type_cost = Column(Float)
    logic_type = Column(Integer)
    logic_type_cost = Column(Float)
    calculation_type = Column(Integer)
    calculation_type_cost = Column(Float)
    embedding_cost = Column(Float)
    version = Column(String(100))

    def __repr__(self):
        return f'<统计 {self.id}>'


class AbnormalCases(Base):
    __tablename__ = "abnormal_cases"
    id = Column(Integer, primary_key=True)
    message = Column(LongText)

    def __repr__(self):
        return f'<那些kimi认为不合规的case {self.id}>'


class EntityEvd(Base):
    __tablename__ = 'entity_evd'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键')
    year = Column(String(4), nullable=False, comment='年份')
    entity_code = Column(String(32), nullable=False, comment='主体编码')
    entity_name = Column(String(128), nullable=False, comment='主体名称')
    evd_code = Column(String(128), nullable=False, comment='Evidence编码')
    evd_name = Column(String(128), nullable=False, comment='Evidence名称')

    __table_args__ = (
        UniqueConstraint('year', 'entity_code', 'evd_code', name='unique_key'),
        Index('idx_entity_code', 'entity_code'),
        Index('idx_evd_code', 'evd_code'),

    )

class FailReports(Base):
    __tablename__ = 'fail_reports'
    id = Column(Integer, primary_key=True)
    message = Column(Text)
    step = Column(Text)


class Missions(Base):
    __tablename__ = 'missions'
    id = Column(Integer, primary_key=True)
    report_name = Column(Text)
    mission_json = Column(Text)
    execute_level = Column(Integer)
    pre_condition_indicator = Column(Text)
    execute_section = Column(Integer)
    person = Column(Text)
    done = Column(Integer)



