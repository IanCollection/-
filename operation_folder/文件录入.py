import json
import os
import random
import time

from logic_folder.数据库表格 import Company, Vector, Reports, Base
from logic_folder.表格处理包 import pdf_material_process, road_map_to_structure_v2, road_map_text_v6_flat, \
    pre_road_map_embedding_convert, road_map_to_db_v6, judge_is_on_edge, standard_json_switch_v2, \
    judge_is_on_edge_v2, road_map_to_db_v7
from 模块工具.智能体仓库 import get_mainbody_v2
from operation_folder.指标表录入 import company_to_db
from dotenv import load_dotenv
from 模块工具.API调用工具 import record_money
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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


def create_db_session():
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


def report_to_db(file, person_name=None, actual_company_code=None, limited_list=None):
    db, engine = create_db_session()

    try:

        report_name_base = file.split("/")[-1][:file.split("/")[-1].rfind(".")] # Use this for report naming
        # report_name = # This was commented out

        year = '2024' # Ensure this year is correct, maybe get it from the Excel row or filename
        company_code_for_db = actual_company_code # Use the passed-in actual_company_code

        if limited_list and company_code_for_db not in limited_list:
            print(f"{report_name_base}的报告不在目标清单，跳过")

        merged = 0

        company = db.query(Company).filter(Company.company_code == company_code_for_db).first()
        if not company:
            print(f"录入新公司:{company_code_for_db}")
            company_name_argument = report_name_base # company_name can be the base report name
            company = company_to_db(company_name=company_name_argument, company_code=company_code_for_db)
            db.add(company)
            db.commit()

        # report_name_for_db will be the filename without .json, e.g., "600359_新农开发2024年度独立董事述职报告--李伟_XX_独立董事报告"
        report_name_for_db = report_name_base

        # Pass report_name_base as report_name to pre_process.
        # pre_process will save the KG JSON as "downloads/{report_name_base}.json"
        # pre_process will also use report_name_base for the Reports table entry.
        processed_report_name_from_pre_process = pre_process(file, report_name_base, company_code_for_db, create_db_session, report_name_base, year, {},
                           merged, person_name)

        if not processed_report_name_from_pre_process:
            print("Error: step1 知识图谱预处理系统性错误")
            fail_message = f'step1 知识图谱预处理系统性错误,报告名:{file}'
            return False, fail_message

        # Ensure the report name from pre_process matches our expectation
        # (It should return report_name_base if successful, or False if not)
        if processed_report_name_from_pre_process != report_name_base:
             print(f"Error: Mismatch in report name from pre_process. Expected {report_name_base}, got {processed_report_name_from_pre_process}")
             fail_message = f'step1 知识图谱预处理报告名不匹配,报告名:{file}'
             return False, fail_message


        company = db.query(Company).filter(Company.company_code == company_code_for_db).first()

        if limited_list and company_code_for_db not in limited_list:
            print(f"{report_name_base}的报告不在目标清单，跳过")

        # Check if report exists using the new report_name_for_db
        report = db.query(Reports).filter(Reports.report_name == report_name_for_db).first()

        if not report:
             retry = 0
             while retry < 5:
                 try:
                     company = db.query(Company).filter(Company.company_code == company_code_for_db).first()
                     new_report = Reports(
                         company_id=company.id,
                         company_name=report_name_base, # company_name in DB
                         report_name=report_name_for_db, # report_name in DB
                         year=year,
                         have_knowledge_graph=1,
                         person_name=person_name
                     )
                     new_report.company = company
                     db.add(new_report)
                     db.commit()
                     break
                 except Exception as e:
                     print(f"Error adding new report: {e}")
                     db.rollback()
                     time.sleep(random.randint(1, 2))
                     retry += 1
        else:
             if not report.person_name:
                 report.person_name = person_name
                 try:
                     db.commit()
                 except Exception as e:
                      print(f"Error updating report with person_name: {e}")
                      db.rollback()

        # to_db_process needs the report_name (which is report_name_base)
        # and it will load the KG JSON from "downloads/{report_name_base}.json"
        to_db_process(file,report_name_base, create_db_session, report_name_for_db, company.id)

    except Exception as e:
        print(f"报告录入发生错误:{e}")
        print("Error: step2 知识图谱预处理系统性错误")
        fail_message = f'step2 知识图谱预处理系统性错误,报告名:{report_name_base}' # Use report_name_base in error
        return False, fail_message
    finally:
        db.commit() # Ensure commit is called even if there's an exception before the explicit commits
        db.close()
        engine.dispose()
    success_message = '报告录入成功'

    return True, success_message


def pre_process(file, company_name, company_code, create_db_session, report_name, year, pic_charts_json_file, merged, person_name=None):
    db, engine = create_db_session()
    type = file.rsplit('_', 1)[-1].rsplit('.', 1)[0] # Type detection not strictly needed if saving to 'downloads'

    usage = 0
    try:
        # Standardize the folder for processed KG JSONs to 'downloads'
        # folder = 'downloads'
        # folder = 'downloads'
        # folder = 'downloads_0630'
        folder = 'downloads_0630_下午_年报'
        # Ensure both the main folder and the subfolder for the type exist
        if type == '年报':
            type = '年报'
        else:
            type = '独立董事报告'

        subfolder = os.path.join(folder, type)
        os.makedirs(subfolder, exist_ok=True)

        file_path = f"{folder}/{type}/知识图谱_{report_name}.json"
        # KG JSON will be saved as "downloads/{report_name}.json"
        # where report_name is report_name_base from report_to_db
        #跑年报的时候改成年报

        if not os.path.exists(file_path):

            # 读取json
            with open(file, "r") as f:
                ocr_json = json.load(f)
            pre_true_ocr_result = get_mainbody_v2(ocr_json)
            # pre_true_ocr_result = pre_true_ocr_result # Redundant line
            true_ocr_result = judge_is_on_edge_v2(pre_true_ocr_result)

            road_map, usage = pdf_material_process(true_ocr_result, report_name, merged)

            if len(pic_charts_json_file) > 0:
                pic_charts = []
                text_content = []
                for key, value in pic_charts_json_file.items():
                    if key == "text":
                        text_content = value
                    elif key == "chart":
                        pic_charts = value
                if len(pic_charts) > 0:
                    road_map["extra_table"] = {"table": pic_charts, "level": 1, "page": -1, "content": []}
                if len(text_content) > 0:
                    road_map["extra_text"] = {"content": text_content, "level": 1, "page": -1, "table": []}

                print("成功加入图片信息")
                print("处理完成，开始入库")

            structure_road_map = road_map_to_structure_v2(road_map)
            standard_json_result = standard_json_switch_v2(structure_road_map, starter=1)

            with open(file_path, "w") as f:
                json.dump(standard_json_result, f, ensure_ascii=False, indent=4)
            print("\n")
        print(f"{company_name} - {report_name}知识图谱已保存") # This message still uses "知识图谱"

        # The report_name used for DB operations is already report_name_base (passed as 'report_name' parameter)
        # No need to prepend "知识图谱_"

        # 检查是否已经入库report
        report = db.query(Reports).filter(Reports.report_name == report_name).first() # report_name is report_name_base
        if not report:
            retry = 0
            while retry < 5:
                try:
                    company = db.query(Company).filter(Company.company_code == company_code).first()
                    # company_name for Reports table can be the actual company name if available, or report_name (base)
                    # report_name for Reports table is report_name (base)
                    new_report = Reports(company_id=company.id, company_name=company_name, report_name=report_name,
                                         year=year, have_knowledge_graph=1, person_name=person_name)
                    new_report.company = company
                    db.add(new_report)
                    db.commit()
                    break
                except Exception as e:
                    print(e)
                    db.rollback()
                    # 等随机1-3秒
                    time.sleep(random.randint(1, 2))
                    retry += 1

            # 记录费用
            record_money(usage, db, is_report=True)
        return report_name # Return the report_name (which is report_name_base)
    except Exception as e:
        print("\n")
        # Ensure report_name in error message is the base name
        print(f"{report_name} 知识图谱转换发生bug,bug:{e}")
        return False
    finally:
        db.close()
        engine.dispose()


def to_db_process(file,company_name, create_db_session, report_name, company_id):
    db, engine = create_db_session()

    # report_name here is report_name_base
    report = db.query(Reports).filter(Reports.report_name == report_name).with_for_update().first()

    if report and report.in_db != 1 and report.in_progress != 1:

        retry = 0
        while retry < 5:
            try:
                report.in_progress = 1
                db.commit()
                break
            except Exception as e:
                db.rollback()
                # 等随机1-3秒
                time.sleep(random.randint(1, 2))
                retry += 1

        # 删除所有vector
        db.query(Vector).filter(Vector.report_id == report.id).delete(synchronize_session=False)
        db.commit()

        # Load the KG JSON from "downloads/{report_name}.json"
        # report_name is report_name_base

        type = file.rsplit('_', 1)[-1].rsplit('.', 1)[0]  # Type detection not strictly needed if saving to 'downloads'

        if type != '年报':
            type = '独立董事报告'
        kg_json_path = f"downloads_0630_下午_年报/{type}/知识图谱_{report_name}.json"

        # kg_json_path = f"downloads/{type}/知识图谱_{report_name}.json"
        with open(kg_json_path, "r") as f:
            standard_json_result = json.load(f)

        try:

            road_map_to_db_v7(standard_json_result, company_id, report.id, report_name=report_name,level=1, master_section_id_list=[])
        except Exception as e:
            print(f"{company_name} - {report_name}知识图谱入库失败")
            print(e)
            db.close()
            engine.dispose()
            return False

        retry = 0
        while retry < 5:
            try:
                report = db.query(Reports).filter(Reports.report_name == report_name).first()
                report.in_db = 1
                report.in_progress = 0
                db.commit()
                break
            except Exception as e:
                db.rollback()
                # 等随机1-3秒
                time.sleep(random.randint(1, 2))
                retry += 1
    db.close()
    engine.dispose()

    return True
