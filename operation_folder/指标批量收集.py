import json
import os
import pickle
import time

import pandas as pd
from sqlalchemy import and_, create_engine, or_
from sqlalchemy.orm import sessionmaker, undefer

from logic_folder.数据库表格 import Company, Indicators, IndicatorsResults, Reports, Vector, Missions, FailReports
from concurrent.futures import ProcessPoolExecutor, as_completed
from logic_folder.问答包 import asking
import signal
from contextlib import contextmanager

from 模块工具.API调用工具 import record_money
from 模块工具.智能体仓库 import num_extract_bot

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')


# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo)
    return engine


def create_db_session():
    # 使用已经创建的引擎来创建新的会话
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine


# 创建一个超时异常
class TimeoutException(Exception):
    pass


# 定义一个超时处理器，当接收到SIGALRM信号时抛出超时异常
def timeout_handler(signum, frame):
    raise TimeoutException


# 使用contextmanager创建一个超时上下文管理器
@contextmanager
def time_limit(seconds):
    # 设置信号处理器
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)  # 设置信号在seconds秒后发送
    try:
        yield
    finally:
        signal.alarm(0)  # 清除信号


def single_indicator_collect(code, create_db_session, company_id, company_name, company_code, year, report_name,report_id,sub_mode,info_vector_tracker = None,force_execute=0,dynamic_keywords = None):
    max_retry = 10
    retry = 0

    while retry < max_retry:
        try:
            with (time_limit(1200)):

                this_db, this_engine = create_db_session()
                indicator = this_db.query(Indicators).filter(Indicators.code == code).first()

                resource_keywords_vectors = []
                for each_vector in indicator.indicator_vectors:
                    if each_vector.is_resource_keywords_vector == 1:
                        resource_keywords_vectors.append(pickle.loads(each_vector.vector))
                similarity_keywords_vectors = []
                for each_vector in indicator.indicator_vectors:
                    if each_vector.is_similarity_keywords_vector == 1:
                        similarity_keywords_vectors.append(pickle.loads(each_vector.vector))
                sub_indicators = indicator.sub_indicators
                section_keywords_vectors = []
                for each_vector in indicator.indicator_vectors:
                    if each_vector.is_section_keywords_vector == 1:
                        section_keywords_vectors.append(pickle.loads(each_vector.vector))

                search_parameters = {}
                search_parameters["table_only"] = indicator.table_only
                search_parameters["without_table"] = indicator.without_table
                search_parameters["weight_keywords"] = indicator.weight_keywords
                search_parameters["similarity_keywords"] = indicator.similarity_keywords
                search_parameters["resource_keywords"] = indicator.resource_keywords
                search_parameters["resource_keywords_vectors"] = resource_keywords_vectors
                search_parameters["similarity_keywords_vectors"] = similarity_keywords_vectors
                search_parameters["min_similarity"] = indicator.min_similarity
                search_parameters["too_detail"] = indicator.too_detail
                search_parameters['report_id'] = report_id
                search_parameters['info_vector_tracker'] = info_vector_tracker
                search_parameters['section_keywords'] = indicator.section_keywords
                search_parameters['section_keywords_vectors'] = section_keywords_vectors
                search_parameters['dynamic_keywords'] = dynamic_keywords

                ask_parameters = {}
                ask_parameters["indicator_type"] = indicator.indicator_type
                ask_parameters["name"] = indicator.name
                ask_parameters["year"] = year
                ask_parameters["equation"] = indicator.equation
                ask_parameters["explain"] = indicator.explain
                ask_parameters["equality_question"] = indicator.equality_question
                ask_parameters["option"] = indicator.option
                ask_parameters["company_id"] = company_id
                ask_parameters["report_name"] = report_name
                ask_parameters["company_name"] = company_name
                ask_parameters["allow_creation"] = indicator.allow_creation
                ask_parameters["mission_type"] = indicator.mission_type
                ask_parameters["positive_example"] = indicator.positive_example
                ask_parameters["positive_example_reason"] = indicator.positive_example_reason
                ask_parameters["negative_example"] = indicator.negative_example
                ask_parameters["negative_example_reason"] = indicator.negative_example_reason
                ask_parameters["necessary_points"] = indicator.necessary_points
                ask_parameters['indicator_code'] = indicator.code

                if dynamic_keywords:
                    ask_parameters['dynamic_keywords'] = dynamic_keywords


                if len(sub_indicators) > 0 and sub_mode == 1:
                    sub_indicators_name_collection = []
                    sub_indicators_result = []
                    sub_indicators_to_find = []
                    for each_sub_indicator in sub_indicators:
                        sub_indicators_name_collection.append(each_sub_indicator.name)
                        each_sub_indicator_result = this_db.query(IndicatorsResults).filter(
                            and_(IndicatorsResults.code == each_sub_indicator.code,
                                 IndicatorsResults.company_code == company_code,
                                 IndicatorsResults.year == year)).first()
                        if each_sub_indicator_result:
                            if each_sub_indicator_result.value == "-0":
                                print(f"子指标{each_sub_indicator.code}收集失败，主指标{code}无法收集")
                                this_db.close()
                                this_engine.dispose()
                                return indicator, "-001"

                            sub_indicators_result.append(each_sub_indicator_result)
                        else:
                            print(f"子指标{each_sub_indicator.code}未收集")
                            sub_indicators_to_find.append(each_sub_indicator.code)
                    if len(sub_indicators_to_find) > 0:
                        print(f"开始收集子指标{sub_indicators_to_find}")
                        data_collection(company_code, sub_indicators_to_find, year,
                                        report_name, force_execute=force_execute)
                        for each_sub_indicator in sub_indicators_to_find:
                            each_sub_indicator_result = this_db.query(IndicatorsResults).filter(
                                and_(IndicatorsResults.code == each_sub_indicator,
                                     IndicatorsResults.company_code == company_code,
                                     IndicatorsResults.year == year)).first()
                            if each_sub_indicator_result:
                                if each_sub_indicator_result.value == "-0":
                                    print(f"子指标{each_sub_indicator.code}收集失败，主指标{code}无法收集")
                                    this_db.close()
                                    this_engine.dispose()
                                    return indicator, "-001"
                                sub_indicators_result.append(each_sub_indicator_result)
                            else:
                                print(f"子指标{each_sub_indicator}收集失败，主指标{code}无法收集")
                                this_db.close()
                                this_engine.dispose()
                                return indicator, "-001"

                    sub_indicators_info_list = []
                    for each_sub_indicator_result in sub_indicators_result:
                        sub_indicators_info_list.append(
                            f"年份:{each_sub_indicator_result.year},指标:{each_sub_indicator_result.name},值:{each_sub_indicator_result.value}")
                    sub_indicators_info = "。".join(sub_indicators_info_list)
                else:
                    sub_indicators_info = None

                record = asking(this_db, sub_indicators_info, ask_parameters, search_parameters, sub_mode)

                if record == False:
                    return "-0", "-0"
                this_db.close()
                this_engine.dispose()
                return indicator, record

        except Exception as e:
            this_db.close()
            this_engine.dispose()
            print(e)
            print(company_name)
            print(code)
            time.sleep(1)
            print(f"重试第{retry}次")
            retry += 1
            if retry == 10:
                raise e

        finally:
            this_db.close()
            this_engine.dispose()


def data_collection(company_code, indicator_code_list, year, report_name=None, force_execute=0,):
    db, engine = create_db_session()
    company = db.query(Company).filter(Company.company_code == company_code).first()
    report = db.query(Reports).filter(Reports.report_name == report_name).first()
    company_name = company.name
    if company is None:
        print("公司不存在")
        return None
    if report is None:
        print("报告不存在")
        return None
    company_id = company.id

    total_indicators = len(indicator_code_list)
    tasks_done = 0
    # 收集每一个指标

    with ProcessPoolExecutor(max_workers=6) as executor:

        # 再收集五级指标
        futures = [
            executor.submit(single_indicator_collect, indicator, create_db_session, company_id, company_name,
                            company_code, year,
                            report_name, report.id,
                            sub_mode=1, force_execute=force_execute) for indicator in
            indicator_code_list]

        for future in as_completed(futures):

            try:
                og_indicator, record = future.result()
                if og_indicator == record:
                    tasks_done += 1
                    print(f"\r指标收集Progress: {tasks_done}/{total_indicators} processed.", end='')
                    continue

                if record == "-001":
                    original_answer = ""
                    value = "信息不足"
                    type = str(og_indicator.indicator_type)
                    year = str(year)
                    code = str(og_indicator.code)
                    name = str(og_indicator.name)
                    cost = "0"
                    reference = ""
                    reference_length = "0"
                    company_id = company.id
                    company_name = company.name
                    tool_use = "False"
                    table_reference = ""
                    assumption = '0'

                else:
                    original_answer = str(record['all_answer_to_return'])
                    if len(original_answer) > 20000:
                        original_answer = original_answer[:20000]
                    true_value = record['result']
                    if og_indicator.indicator_type == "文本" and (
                            true_value == "-1" or true_value == -1) or true_value == "-0":
                        true_value = "信息不足"
                    type = str(og_indicator.indicator_type)
                    year = str(year)
                    code = str(og_indicator.code)
                    name = str(og_indicator.name)
                    tool_use = str(record['used_tool'])
                    value = str(true_value)
                    table_reference = str(record['table_reference'])
                    cost = str(record['cost'])
                    reference = str(record['reference'])
                    reference_length = str(record['reference_length'])
                    company_id = company.id
                    company_name = company.name
                    assumption = record['assumption']
                    mission_type = og_indicator.mission_type

                    # record_money(float(cost), db, is_report=False, mission_type=mission_type)

                indicator_result = IndicatorsResults(indicator_type=type, year=year, code=code, name=name, cost=cost,
                                                     value=value, table_reference=table_reference,
                                                     company_code=company_code,
                                                     report_name=report_name,
                                                     reference=reference, original_answer=original_answer,
                                                     reference_length=reference_length, assumption=assumption,
                                                     company_id=company_id, company_name=company_name,
                                                     tool_use=tool_use)
                db.add(indicator_result)
                db.commit()
                tasks_done += 1
                print(f"\r指标收集Progress: {tasks_done}/{total_indicators} processed.", end='')

            except Exception as e:
                print(name)
                print(record)
                print(f"\nError processing: {e}")
    db.close()
    engine.dispose()
    return True


def single_data_collection(indicator_info_dict,force_execute=0,dynamic_question_word = None,mission_id = None,info_vector_tracker = None):

    db, engine = create_db_session()
    company_code = indicator_info_dict['company_code']
    indicator_code = indicator_info_dict['indicator_code']
    year = indicator_info_dict['year']
    report_name = indicator_info_dict['report_name']
    if 'info_vector_tracker' in indicator_info_dict.keys():
        info_vector_tracker = indicator_info_dict['info_vector_tracker']
    else:
        info_vector_tracker = None
    try:
        company = db.query(Company).filter(Company.company_code == company_code).first()
        report = db.query(Reports).filter(Reports.report_name == report_name).first()

        if company is None:
            print("公司不存在")
            print(f"Error 1: Company does not exist. 公司code:{company_code}")
            fail_message = f'Error 1: Company does not exist. 公司code:{company_code}'
            new_fail_report = FailReports(message=fail_message, step=3)
            db.add(new_fail_report)
            db.commit()
            return False, fail_message

        company_name = company.name
        if report is None or report.in_db != 1:
            print("报告不存在")
            print(f"Error 2: Report does not exist. 报告code:{report_name}")
            fail_message = f"Error 2: Report does not exist. 报告code:{report_name}"
            new_fail_report = FailReports(message=fail_message, step=3)
            db.add(new_fail_report)
            db.commit()
            return True, fail_message
        indicator = db.query(Indicators).filter(Indicators.code == indicator_code).first()
        if not indicator:
            print(f"不存在指标{indicator_code}")
            print("非目标指标")
            success_message = 'Non-target indicator.'
            return True, success_message



        company_id = company.id

        #检查是否已经收集
        # check = db.query(IndicatorsResults).filter(and_(IndicatorsResults.code == indicator_code,IndicatorsResults.company_code == company_code,IndicatorsResults.person == dynamic_question_word,IndicatorsResults.year == year)).first()
        # have_assumption_answer = False
        # if check and check.value != "信息不足" and check.assumption == 1:
        #     have_assumption_answer = True
        #
        # if check == 1 and check.value != "信息不足":
        #     print("已存在答案")
        #     success_message = 'Answer already exists.'
        #     return True, success_message
        #
        # if check and check.value == "信息不足":
        #     db.delete(check)

        og_indicator, record = single_indicator_collect(indicator_code, create_db_session,
                                                        company_id, company_name,
                                                        company_code, year,
                                                        report_name, report.id,info_vector_tracker = info_vector_tracker,
                                                        sub_mode=1, force_execute=force_execute,dynamic_keywords = dynamic_question_word)


        if record == "-001":
            original_answer = ""
            value = "信息不足"
            type = str(og_indicator.indicator_type)
            year = str(year)
            code = str(og_indicator.code)
            name = str(og_indicator.name)
            cost = "0"
            reference = ""
            reference_length = "0"
            # company_id = company.id
            # company_name = company.name
            tool_use = "False"
            table_reference = ""
            assumption = '0'

        else:
            original_answer = str(record['all_answer_to_return'])
            if len(original_answer) > 20000:
                original_answer = original_answer[:20000]
            true_value = str(record['result'])
            if og_indicator.indicator_type == "文本" and (true_value == "-1" or true_value == -1) or true_value == "-0":
                true_value = "信息不足"

            format_type = og_indicator.format_type
            format_cost = 0
            if true_value == "信息不足":
                missing_fill = indicator.missing_fill
                if missing_fill:
                    true_value = missing_fill
            elif format_type:
                if format_type == 'num':
                    if true_value != '未披露' and not str(true_value).isdigit():
                        # format_cost,formatted_true_value= num_extract_bot(indicator.name, str(true_value))
                        # true_value = formatted_true_value
                        true_value = true_value
                elif format_type == 'percentage':
                    if true_value != '未披露':
                        if true_value == "1":
                            true_value = "100%"
                        elif '%' not in str(true_value):
                            true_value = true_value + '%'
            if indicator_code == "12.1":
                if true_value == "涉及":
                    true_value = "是"
                else:
                    true_value = "否"

            type = str(og_indicator.indicator_type)
            year = str(year)
            code = str(og_indicator.code)
            name = str(og_indicator.name)
            tool_use = str(record['used_tool'])
            value = str(true_value)
            table_reference = str(record['table_reference'])
            cost = str(record['cost'])
            reference = str(record['reference'])
            reference_length = str(record['reference_length'])
            company_id = company.id
            company_name = company.name
            assumption = record['assumption']
            mission_type = og_indicator.mission_type
            cost = float(cost) + float(format_cost)

        indicator_result = IndicatorsResults(indicator_type=type, year=year, code=code, name=name, cost=cost,
                                             value=value, table_reference=table_reference,
                                             company_code=company_code,
                                             report_name=report_name,
                                             reference=reference, original_answer=original_answer,
                                             reference_length=reference_length, assumption=assumption,
                                             company_id=company_id, company_name=company_name,
                                             tool_use=tool_use, quit=0,person = dynamic_question_word)

        # if have_assumption_answer:
        #     if indicator_result.value != "信息不足" and indicator_result.assumption == 0:
        #         db.delete(check)
        #         db.add(indicator_result)
        #
        # else:
        #     db.add(indicator_result)
        db.add(indicator_result)

        if mission_id:
            mission = db.query(Missions).filter(Missions.id == mission_id).first()
            if mission:
                mission.done = 1

        # 统一commit，减少IO次数
        db.commit()
        success_message = {}
        return True, success_message

    except Exception as e:
        print(indicator_code)
        print(f"\nError processing: {e}")
        print("Error 4: 系统性失误")
        fail_message = f'Systemic failure.{e},报告名:{report_name},指标代码:{indicator_code}'

        return False, fail_message
    finally:
        db.close()
        engine.dispose()



