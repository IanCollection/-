import pandas as pd


def ensure_columns_exist(file_path):
    required_columns = [
        'equation', '同义标签', '指标描述', 'option', 'without_table',
        'table_only', 'allow_creation', 'mission_type', '信息是否可以直接获得',
        'min_similarity', 'too_detail', 'info_sections', '正面Example',
        '正面判断理由', '反面Example', '反面判断理由', '核心关注点','info_tables'
    ]

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Check and add missing columns
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # or you can use pd.NA or np.nan

    # Save the modified DataFrame back to Excel
    new_path = file_path.replace('.xlsx', '_new.xlsx')
    df.to_excel(new_path, index=False)

if __name__ == '__main__':
    # 使用示例
    file_path = '/Users/zinozhang/Desktop/副本数据参数准备v6.xlsx'
    ensure_columns_exist(file_path)