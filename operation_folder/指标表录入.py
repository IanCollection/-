import json
import os
import pickle

import pandas as pd

from logic_folder.数据库表格 import Indicators, Company, Vector
from 模块工具.openai相关工具 import get_cluster_embeddings

from dotenv import load_dotenv

from logic_folder.语义增强包 import get_key_words_no_thread
from 模块工具.智能体仓库 import get_similarity_words_bot, check_sub_indicators
from concurrent.futures import ProcessPoolExecutor, as_completed

load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    # 创建数据库引擎
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine


def company_file_to_db(file=None):
    db, engine = create_db_session()
    if file:
        company_sheet_to_db(file, db)


def company_to_db(company_name=None, company_code=None):
    db, engine = create_db_session()
    company = single_company_to_db(company_name, company_code, db)
    return company


def formula_contain(text):
    if "公式" in text or "=" in text or "/" in text:
        return True
    else:
        return False



def single_indicator_to_db(row, create_db_session, year):
    db, engine = create_db_session()
    # 创建一个Indicators实例

    indicator_type = row.get('指标类型')
    code = row.get('序号')
    master_indicator_name = row.get('指标名')

    if master_indicator_name and master_indicator_name[-2:] == "_W":
        master_indicator_name = master_indicator_name[:-2]

    if pd.isnull(master_indicator_name):
        print(f"指标:{master_indicator_name}指标名为空，跳过")
        db.close()
        engine.dispose()
        return master_indicator_name

    retry = 0
    total_retry = 10
    ai_weight_keywords_for_db = "{}" # Variable to store JSON string for DB
    ai_weight_keywords_json_data = None # Variable to store Python dict for logic
    while retry < total_retry:
        need_retry = False
        try:
            # Directly get the dictionary (or None) from get_key_words_no_thread
            raw_keywords_data = get_key_words_no_thread(master_indicator_name)

            if isinstance(raw_keywords_data, dict):
                ai_weight_keywords_json_data = raw_keywords_data
                # Check for nested dicts which might indicate an issue with the keyword generation
                for each_value in ai_weight_keywords_json_data.values():
                    if isinstance(each_value, dict):
                        print(f"Warning: Nested dictionary found in keywords for {master_indicator_name}. Retrying.")
                        need_retry = True
                        break
                if not need_retry: # If no nested dicts, convert to JSON string for DB
                    ai_weight_keywords_for_db = json.dumps(ai_weight_keywords_json_data, ensure_ascii=False)

            elif raw_keywords_data is None:
                print(f"Warning: get_key_words_no_thread returned None for {master_indicator_name}")
                # Keep ai_weight_keywords_for_db as "{}" (empty JSON object)
                # ai_weight_keywords_json_data remains None
                # Depending on desired behavior, you might want to set need_retry = True here
                # For now, we assume None means no keywords, not an error requiring retry.
                pass # Explicitly do nothing, will use default "{}" for DB and None for logic

            else: # Should not happen if get_key_words_no_thread behaves as expected (returns dict or None)
                print(f"Unexpected data type from get_key_words_no_thread for {master_indicator_name}: {type(raw_keywords_data)}. Retrying.")
                need_retry = True


            if need_retry:
                retry += 1
                print(f"retrying {master_indicator_name} (attempt {retry}/{total_retry})")
                ai_weight_keywords_for_db = "{}" # Reset before retry
                ai_weight_keywords_json_data = None # Reset before retry
                continue
            break
        except Exception as e:
            print(f"Error during keyword processing for {master_indicator_name} - {e}")
            print(f"retrying {master_indicator_name} (attempt {retry}/{total_retry})")
            retry += 1
            ai_weight_keywords_for_db = "{}" # Reset on exception
            ai_weight_keywords_json_data = None # Reset on exception

    resource_keywords = row.get('info_tables')

    if pd.isnull(row.get('info_points')):
        ai_similarity_keywords = []
    else:
        ai_similarity_keywords = str(row.get('info_points').replace("，", ",")).split(",")

    equation = row.get('equation')
    equality_question = row.get('同义标签')
    explain = row.get('指标解释')
    option = row.get('option')
    without_table = row.get('without_table')
    table_only = row.get("table_only")
    allow_creation = row.get("allow_creation")
    mission_type = row.get("mission_type")
    missing_fill = row.get("缺失值填补")
    direct_mission = row.get("信息是否可以直接获得")
    min_similarity = row.get("min_similarity")
    too_detail = row.get("too_detail")
    pre_condition_indicator = row.get("precondition")
    section_keywords = row.get("info_sections")
    positive_example = row.get('正面Example')
    positive_example_reason = row.get('正面判断理由')
    negative_example = row.get('反面Example')
    negative_example_reason = row.get('反面判断理由')
    necessary_points = row.get('核心关注点')
    execute_level = row.get('execute_level')
    execute_section = row.get('execute_section')
    format_type = row.get('format_type')

    if pd.isnull(format_type):
        format_type = None
    if pd.isnull(missing_fill):
        missing_fill = None
    else:
        if str(missing_fill) == "1":
            missing_fill = '100%'
    if pd.isnull(pre_condition_indicator):
        pre_condition_indicator = None
    if pd.isnull(execute_level):
        execute_level = None
    if pd.isnull(execute_section):
        execute_section = None
    if pd.isnull(positive_example):
        positive_example = None
    if pd.isnull(positive_example_reason):
        positive_example_reason = None
    if pd.isnull(negative_example):
        negative_example = None
    if pd.isnull(negative_example_reason):
        negative_example_reason = None
    if pd.isnull(necessary_points):
        necessary_points = None
    if pd.isnull(direct_mission):
        direct_mission = None
    if pd.isnull(mission_type):
        mission_type = "逻辑型"
    if pd.isnull(section_keywords):
        section_keywords = None
    else:
        section_keywords = str(str(section_keywords).replace("，", ",").split(","))
    if pd.isnull(resource_keywords):
        resource_keywords = None
    else:
        resource_keywords = str(str(resource_keywords).replace("，", ",").split(","))
    if pd.isnull(equation):
        equation = None
    if pd.isnull(equality_question):
        equality_question = None
    if pd.isnull(explain):
        explain = None
    if pd.isnull(option):
        option = None
    else:
        option = str(str(option).replace("，", ",").split(","))
    if pd.isnull(without_table):
        without_table = None
    if pd.isnull(table_only):
        table_only = None
    if pd.isnull(allow_creation):
        allow_creation = None
    if pd.isnull(min_similarity):
        min_similarity = None
    if pd.isnull(too_detail):
        too_detail = None

    master_indicator = Indicators(
        indicator_type=indicator_type,
        code=code,
        name=master_indicator_name,
        format_type=format_type,
        explain=explain,
        equality_question=equality_question,
        weight_keywords=ai_weight_keywords_for_db, # Use the JSON string here
        pre_condition_indicator=pre_condition_indicator,
        similarity_keywords=str(ai_similarity_keywords),
        resource_keywords=resource_keywords,
        section_keywords=section_keywords,
        direct_mission=direct_mission,
        without_table=without_table,
        table_only=table_only,
        mission_type=mission_type,
        min_similarity=min_similarity,
        missing_fill=missing_fill,
        allow_creation=allow_creation,
        too_detail=too_detail,
        equation=equation,
        option=option,
        master_indicator_id=None,
        master_indicator=None,
        sub_indicators=[],
        positive_example=positive_example,
        positive_example_reason=positive_example_reason,
        negative_example=negative_example,
        negative_example_reason=negative_example_reason,
        necessary_points=necessary_points,
        execute_level=execute_level,
        execute_section=execute_section

    )
    db.add(master_indicator)
    db.commit()

    resource_keywords_vectors = []
    if not pd.isnull(resource_keywords):
        resource_keywords = str(resource_keywords).replace("，", ",").split(",")
        if len(resource_keywords) > 0:
            pure_vectors = []
            retry = 0
            while retry < 10:
                try:
                    pure_vectors = get_cluster_embeddings(resource_keywords, model="text-embedding-v1")
                    break
                except Exception as e:
                    retry += 1
                    print(e)
                    print(f"{master_indicator_name} resource_keywords embedding错误，重试第{retry}次")
            for each_vector in pure_vectors:
                new_vector = Vector(vector=pickle.dumps(each_vector), link=1, indicator_id=master_indicator.id,
                                    is_resource_keywords_vector=1)

                resource_keywords_vectors.append(new_vector)
            db.add_all(resource_keywords_vectors)
            db.commit()

    section_keywords_vectors = []
    if not pd.isnull(section_keywords):
        section_keywords = str(section_keywords).replace("，", ",").split(",")
        if len(section_keywords) > 0:
            pure_vectors = []
            retry = 0
            while retry < 10:
                try:
                    pure_vectors = get_cluster_embeddings(section_keywords, model="text-embedding-v1")
                    break
                except Exception as e:
                    retry += 1
                    print(e)
                    print(f"{master_indicator_name} section_keywords embedding错误，重试第{retry}次")
            for each_vector in pure_vectors:
                new_vector = Vector(vector=pickle.dumps(each_vector), link=1, indicator_id=master_indicator.id,
                                    is_section_keywords_vector=1)
                section_keywords_vectors.append(new_vector)
            db.add_all(section_keywords_vectors)
            db.commit()
    similarity_keywords_vectors = []
    if len(ai_similarity_keywords) > 0:

        pure_vectors = []
        retry = 0
        while retry < 10:
            try:
                pure_vectors = get_cluster_embeddings(ai_similarity_keywords, model="text-embedding-v1")
                break
            except Exception as e:
                retry += 1
                print(e)
                print(f"{master_indicator_name} similarity_keywords embedding错误，重试第{retry}次")
        for each_vector in pure_vectors:
            new_vector = Vector(vector=pickle.dumps(each_vector), link=1, indicator_id=master_indicator.id,
                                is_similarity_keywords_vector=1)
            similarity_keywords_vectors.append(new_vector)
        db.add_all(similarity_keywords_vectors)
        db.commit()

    if indicator_type == "数值" and not pd.isnull(equation):
        usage, answer = check_sub_indicators(master_indicator_name, year)
        sub_indicators = json.loads(answer)["indicators"]
        for sub_index, each_sub_indicator in enumerate(sub_indicators):
            name = each_sub_indicator['name']
            # check exist
            exist = db.query(Indicators).filter(Indicators.name == name).first()
            if name == master_indicator_name or exist:
                continue

            retry = 0
            total_retry = 10
            sub_indicator_weight_keywords_for_db = "{}" # Variable to store JSON string for DB
            sub_indicator_weight_keywords_json_data = None # Variable to store Python dict for logic
            while retry < total_retry:
                need_retry = False
                try:
                    # Directly get the dictionary (or None) from get_key_words_no_thread
                    raw_sub_keywords_data = get_key_words_no_thread(name)

                    if isinstance(raw_sub_keywords_data, dict):
                        sub_indicator_weight_keywords_json_data = raw_sub_keywords_data
                        # Check for nested dicts
                        for each_value in sub_indicator_weight_keywords_json_data.values():
                            if isinstance(each_value, dict):
                                print(f"Warning: Nested dictionary found in keywords for sub-indicator {name}. Retrying.")
                                need_retry = True
                                break
                        if not need_retry: # If no nested dicts, convert to JSON string for DB
                            sub_indicator_weight_keywords_for_db = json.dumps(sub_indicator_weight_keywords_json_data, ensure_ascii=False)
                    
                    elif raw_sub_keywords_data is None:
                        print(f"Warning: get_key_words_no_thread returned None for sub_indicator {name}")
                        # Keep sub_indicator_weight_keywords_for_db as "{}"
                        # sub_indicator_weight_keywords_json_data remains None
                        pass
                    
                    else: # Should not happen
                        print(f"Unexpected data type from get_key_words_no_thread for sub-indicator {name}: {type(raw_sub_keywords_data)}. Retrying.")
                        need_retry = True

                    if need_retry:
                        retry += 1
                        print(f"retrying sub-indicator {name} (attempt {retry}/{total_retry})")
                        sub_indicator_weight_keywords_for_db = "{}" # Reset
                        sub_indicator_weight_keywords_json_data = None # Reset
                        continue
                    break

                except Exception as e:
                    print(f"Error during keyword processing for sub-indicator {name} - {e}")
                    print(f"retrying sub-indicator {name} (attempt {retry}/{total_retry})")
                    retry += 1
                    sub_indicator_weight_keywords_for_db = "{}" # Reset
                    sub_indicator_weight_keywords_json_data = None # Reset

            sub_indicator_similarity_keywords = get_similarity_words_bot(name)

            sub_indicator = Indicators(
                indicator_type=indicator_type,
                code=code + "_" + str(sub_index),
                name=each_sub_indicator['name'],
                weight_keywords=sub_indicator_weight_keywords_for_db, # Use the JSON string here
                similarity_keywords=str(sub_indicator_similarity_keywords),
                resource_keywords=None,
                equation=None,
                master_indicator=master_indicator,
                sub_indicators=[]
            )
            db.add(sub_indicator)
            db.commit()

            sub_indicator_similarity_keywords_vectors = []
            for each_vector in get_cluster_embeddings(sub_indicator_similarity_keywords,
                                                      model="text-embedding-v1"):
                new_vector = Vector(vector=pickle.dumps(each_vector), link=1, indicator_id=sub_indicator.id)
                sub_indicator_similarity_keywords_vectors.append(new_vector)
            db.add_all(sub_indicator_similarity_keywords_vectors)
            db.commit()

    db.close()
    engine.dispose()
    return master_indicator_name


def indicators_sheet_to_db(file, year):
    base_dir = 'uploads'
    file = os.path.join(base_dir, file)

    # 读取excel
    db, engine = create_db_session()
    try:
        db.query(Indicators).delete()
        db.commit()
        df = pd.read_excel(file)
        # 遍历DataFrame中的每一行
        total_indicators = len(df)
        tasks_done = 0
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(single_indicator_to_db, row[1], create_db_session, year) for row in
                df.iterrows()]
            for future in as_completed(futures):

                try:
                    indicator_name = future.result()
                    tasks_done += 1
                    print(
                        f"\r指标:{indicator_name}已入库,指标入库Progress: {tasks_done}/{total_indicators} 入库." + " " * 30,
                        end='')
                except Exception as e:
                    print(f"\n指标入库 Error processing: {e}")
        return True, {}
    except Exception as e:
        print(f"指标入库 Error: {e}")
        return False, e
    finally:
        db.close()
        engine.dispose()


def indicators_sheet_to_db_v2(file, year):
    # 读取excel
    df = pd.read_excel(file)
    # 遍历DataFrame中的每一行
    total_indicators = len(df)
    tasks_done = 0
    db, engine = create_db_session()
    db.query(Indicators).delete()
    db.commit()

    for row in df.iterrows():
        try:
            indicator_name = single_indicator_to_db(row[1], create_db_session, year)
            print(f"开始处理{indicator_name}")
            tasks_done += 1
            print(f"\r指标:{indicator_name}已入库,指标入库Progress: {tasks_done}/{total_indicators} 入库.", end='')

        except Exception as e:
            print(e)
    db.close()
    engine.dispose()


def company_sheet_to_db(file, db):
    # 读取excel
    df = pd.read_excel(file)
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        print(f"\r正在录入第{index}个公司，名称：{row['公司名称']}", end="")
        # 创建一个Indicators实例
        company = Company(
            name=row['公司名称'],
            industry=row['行业'],
            in_db_reports="[]"
        )
        db.add(company)
        db.commit()


def single_company_to_db(company_name, company_code, db):
    company = Company(
        name=company_name,
        company_code=company_code,
        industry="无",
    )
    db.add(company)
    db.commit()
    return company
