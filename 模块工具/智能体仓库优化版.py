from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


class IndicatorType(Enum):
    """指标类型枚举"""
    NUMERICAL = "数值"
    SINGLE_CHOICE = "单选"
    MULTIPLE_CHOICE = "多选"
    TEXT = "文本"


class MissionType(Enum):
    """任务类型枚举"""
    CALCULATION = "计算型"


@dataclass
class PromptParameters:
    """prompt参数数据类"""
    name: str
    year: Optional[str] = None
    equation: Optional[str] = None
    explain: Optional[str] = None
    equality_question: Optional[str] = None
    option: Optional[str] = None
    company_name: Optional[str] = None
    allow_creation: Optional[bool] = None
    indicator_type: str = IndicatorType.NUMERICAL.value
    mission_type: Optional[str] = None
    positive_example: Optional[str] = None
    positive_example_reason: Optional[str] = None
    negative_example: Optional[str] = None
    negative_example_reason: Optional[str] = None
    necessary_points: Optional[str] = None
    missing_fill: str = ''
    dynamic_keywords: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptParameters':
        """从字典创建PromptParameters实例"""
        return cls(
            name=data["name"],
            year=data.get("year"),
            equation=data.get("equation"),
            explain=data.get("explain"),
            equality_question=data.get("equality_question"),
            option=data.get("option"),
            company_name=data.get("company_name"),
            allow_creation=data.get("allow_creation"),
            indicator_type=data.get("indicator_type", IndicatorType.NUMERICAL.value),
            mission_type=data.get("mission_type"),
            positive_example=data.get("positive_example"),
            positive_example_reason=data.get("positive_example_reason"),
            negative_example=data.get("negative_example"),
            negative_example_reason=data.get("negative_example_reason"),
            necessary_points=data.get("necessary_points"),
            missing_fill=data.get('missing_fill', ''),
            dynamic_keywords=data.get("dynamic_keywords")
        )


class PromptBuilder:
    """负责构建不同类型的prompt"""
    
    def __init__(self, params: PromptParameters):
        self.params = params
        self.note = self._build_note()
    
    def _build_note(self) -> str:
        """构建备注信息"""
        if self.params.equality_question:
            return f",注意在这个问题中，你可以认为{self.params.equality_question}和{self.params.name}是等价的"
        return "无"
    
    def _add_examples(self, question_line: str) -> str:
        """添加正例和反例"""
        if self.params.positive_example:
            question_line += f"为了能让你明白判断的尺度我为你提供了正例。正例:{self.params.positive_example},原因:{self.params.positive_example_reason}。"
        
        if self.params.negative_example:
            question_line += f"为了能让你明白判断的尺度我为你提供了反例:{self.params.negative_example},原因:{self.params.negative_example_reason}。"
        
        return question_line
    
    def _build_strictness(self, is_choice_type: bool = False) -> str:
        """构建严格性要求"""
        if is_choice_type:
            strictness = ("(需要注意的是,材料往往并不能满足所有的问题需求,但是如果是有多种情况下的一种得到满足"
                         "(除非特别说明需要严格满足要求)，都可以视为1。但是你需要说出你的判断理由。")
        else:
            strictness = ""
        
        if self.params.necessary_points:
            if is_choice_type:
                strictness += f"但是注意!以下内容是必须要有的:{self.params.necessary_points}。)"
            else:
                strictness = f"(注意!以下内容是必须要有的:{self.params.necessary_points}。)"
        elif is_choice_type:
            strictness += ")"
        
        return strictness
    
    def build_numerical_question(self) -> str:
        """构建数值类型问题"""
        if self.params.equation:
            question_line = (f"请帮我回答关于{self.params.dynamic_keywords}的问题:{self.params.name},"
                           f"问题解释:{self.params.explain},备注信息:{self.note}。"
                           f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords}")
        else:
            question_line = (f"请帮我回答关于{self.params.dynamic_keywords}的问题:{self.params.name},"
                           f"问题解释:{self.params.explain},备注信息:{self.note}。"
                           f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords}。")
        
        return self._add_examples(question_line)
    
    def build_choice_question(self, is_single_choice: bool = True) -> str:
        """构建选择类型问题"""
        choice_type = "一个最佳选项" if is_single_choice else "一个或多个最佳选项"
        question_line = (f"请帮我回答关于{self.params.dynamic_keywords}的问题:{self.params.name},"
                        f"问题解释:{self.params.explain},你需要从option中选出{choice_type},"
                        f"option:{self.params.option},注意要保证你的答案和选项的用词一致性。"
                        f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords}")
        
        return self._add_examples(question_line)
    
    def build_text_question(self) -> str:
        """构建文本类型问题"""
        question_line = (f"请帮我回答关于{self.params.dynamic_keywords}的问题:{self.params.name},"
                        f"问题解释:{self.params.explain},备注信息:{self.note}。"
                        f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords}。"
                        f"注意!回答需要中文")
        
        return self._add_examples(question_line)


class MessageBuilder:
    """负责构建不同AI提供商的消息格式"""
    
    def __init__(self, params: PromptParameters, sub_reference: str):
        self.params = params
        self.sub_reference = sub_reference
        self.prompt_builder = PromptBuilder(params)
    
    def _build_base_instruction(self) -> str:
        """构建基础指令"""
        return (f"你是一个金融分析师，你需要基于我给你提供的参考材料回答问题，但是并不是所有参考信息都是有用信息。"
                f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords}。"
                f"回答的时候需要说出根据哪一个或多个参考得到的答案(你只需要reference参考+序号，不用reference里面的内容),"
                f"比如'根据参考4,人口总数是 123,321。")
    
    def _build_choice_instruction(self) -> str:
        """构建选择题指令"""
        return (f"你是一个金融分析师，你需要基于我给你提供的参考材料做出判定。"
                f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords},"
                f"但是并不是所有参考信息都是有用信息。回答的时候需要展示原文reference(在你的回答中引用原文原句)。")
    
    def build_numerical_message(self) -> Tuple[List[Dict[str, str]], str]:
        """构建数值类型消息"""
        question_line = self.prompt_builder.build_numerical_question()
        strictness = self.prompt_builder._build_strictness()
        
        content = (f"{self._build_base_instruction()}"
                  f"{question_line}"
                  f"{strictness}"
                  f"如果参考材料完全没有所需信息，则回复'信息不足'。")
        
        # 添加计算型任务的特殊指令
        if self.params.mission_type == MissionType.CALCULATION.value:
            content += "当涉及数学计算时，you can think step by step，你必须展示计算公式和相关数据，编写并且运行代码来回答问题。"
        
        message = [
            {"role": "user", "content": self.sub_reference},
            {"role": "user", "content": content}
        ]
        
        return message, question_line
    
    def build_choice_message(self, is_single_choice: bool = True) -> Tuple[List[Dict[str, str]], str]:
        """构建选择类型消息"""
        question_line = self.prompt_builder.build_choice_question(is_single_choice)
        strictness = self.prompt_builder._build_strictness(is_choice_type=True)
        
        content = (f"{self._build_choice_instruction()}"
                  f"{question_line}"
                  f"{strictness}"
                  f"如果参考材料完全没有所需信息，则回复'信息不足'。")
        
        message = [
            {"role": "user", "content": self.sub_reference},
            {"role": "user", "content": content}
        ]
        
        return message, question_line
    
    def build_text_message(self) -> Tuple[List[Dict[str, str]], str]:
        """构建文本类型消息"""
        question_line = self.prompt_builder.build_text_question()
        strictness = self.prompt_builder._build_strictness()
        
        content = (f"{self._build_base_instruction()}"
                  f"{question_line}"
                  f"{strictness}"
                  f"如果参考材料完全没有所需信息，则回复'信息不足'。")
        
        message = [
            {"role": "user", "content": self.sub_reference},
            {"role": "user", "content": content}
        ]
        
        return message, question_line
    
    def add_system_message_for_non_openai(self, message: List[Dict[str, str]], question_line: str) -> List[Dict[str, str]]:
        """为非OpenAI提供商添加系统消息"""
        system_content = (f"你是一个金融分析师，你需要基于我给你提供的参考材料回答问题，但是并不是所有参考信息都是有用信息。"
                         f"注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{self.params.dynamic_keywords}"
                         f"回答的时候需要说出根据哪一个或多个参考得到的答案(你只需要reference参考+序号，不用reference里面的内容),"
                         f"比如'根据参考4,人口总数是 123,321。"
                         f"{question_line}"
                         f"如果参考材料完全没有所需信息，则回复'信息不足'。")
        
        return [{"role": "system", "content": system_content}] + message


def interior_data_collection_messages_optimized(
    sub_mode: int,
    sub_reference: str,
    sub_indicators_info: Optional[Any],
    ask_parameters: Dict[str, Any],
    ai: str = "qwen"
) -> Tuple[List[Dict[str, str]], str]:
    """
    优化版本的interior_data_collection_messages函数
    
    Args:
        sub_mode: 子模式
        sub_reference: 子参考资料
        sub_indicators_info: 子指标信息
        ask_parameters: 询问参数字典
        ai: AI提供商类型
    
    Returns:
        Tuple[消息列表, 基础问题行]
    
    Raises:
        ValueError: 当指标类型不支持时
    """
    # 参数验证和转换
    try:
        params = PromptParameters.from_dict(ask_parameters)
    except KeyError as e:
        raise ValueError(f"缺少必要参数: {e}")
    
    if not params.dynamic_keywords:
        raise ValueError("dynamic_keywords参数不能为空")
    
    # 创建消息构建器
    message_builder = MessageBuilder(params, sub_reference)
    
    # 根据指标类型构建消息
    if params.indicator_type == IndicatorType.NUMERICAL.value:
        if not sub_indicators_info or sub_mode == 0:
            message, base_question_line = message_builder.build_numerical_message()
        else:
            # 处理有子指标信息的情况
            # 这里可以添加更复杂的逻辑
            message, base_question_line = message_builder.build_numerical_message()
            
    elif params.indicator_type == IndicatorType.SINGLE_CHOICE.value:
        message, base_question_line = message_builder.build_choice_message(is_single_choice=True)
        
    elif params.indicator_type == IndicatorType.MULTIPLE_CHOICE.value:
        message, base_question_line = message_builder.build_choice_message(is_single_choice=False)
        
    elif params.indicator_type == IndicatorType.TEXT.value:
        message, base_question_line = message_builder.build_text_message()
        
    else:
        raise ValueError(f"不支持的指标类型: {params.indicator_type}")
    
    # 为非OpenAI提供商添加系统消息
    if ai != "openai":
        message = message_builder.add_system_message_for_non_openai(message, base_question_line)
    
    return message, base_question_line


# 保持向后兼容的原函数名
def interior_data_collection_messages(sub_mode, sub_reference, sub_indicators_info, ask_parameters, ai="qwen"):
    """原函数的向后兼容版本"""
    return interior_data_collection_messages_optimized(sub_mode, sub_reference, sub_indicators_info, ask_parameters, ai) 