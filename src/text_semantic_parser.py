"""
文本提示词结构化解析模块
Structured Text Prompt Semantic Parser

将文本提示词分解为四类语义要素：
1. 主体对象 (Subject): 名词实体
2. 属性描述 (Attributes): 形容词/副词
3. 场景关系 (Scene): 空间位置关系
4. 风格指令 (Style): 艺术类型
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SemanticCategory(Enum):
    """语义要素分类"""
    SUBJECT = "subject"
    ATTRIBUTES = "attributes"
    SCENE = "scene"
    STYLE = "style"


@dataclass
class SemanticElement:
    """语义要素数据结构"""
    category: SemanticCategory
    tokens: List[str]
    confidence: float


class TextSemanticParser:
    """
    文本语义解析器
    
    功能：
    - 将提示词分解为语义要素
    - 提取关键词和短语
    - 计算语义相关性
    """
    
    # 常见的主体对象名词
    SUBJECT_KEYWORDS = {
        # 人物
        'person', 'man', 'woman', 'girl', 'boy', 'child', 'people',
        'portrait', 'face', 'figure',
        # 动物
        'dog', 'cat', 'bird', 'animal', 'horse', 'lion', 'tiger',
        # 场景元素
        'tree', 'forest', 'mountain', 'lake', 'ocean', 'sea', 'beach',
        'building', 'house', 'city', 'street', 'road', 'bridge',
        'flower', 'rose', 'sunflower', 'garden',
        # 物体
        'car', 'bicycle', 'boat', 'airplane', 'table', 'chair', 'book'
    }
    
    # 常见的属性描述词
    ATTRIBUTE_KEYWORDS = {
        # 颜色
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'white', 'black', 'gray', 'golden', 'silver', 'vibrant',
        # 质感
        'smooth', 'rough', 'shiny', 'matte', 'glossy', 'metallic', 'wooden',
        # 质量
        'beautiful', 'gorgeous', 'stunning', 'amazing', 'excellent',
        'high quality', 'detailed', 'sharp', 'clear', 'crisp',
        'blurry', 'soft', 'bright', 'dark', 'light',
        # 大小
        'large', 'small', 'big', 'tiny', 'massive', 'huge', 'miniature',
        # 其他
        'detailed', 'realistic', 'abstract', 'minimalist', 'intricate'
    }
    
    # 场景关系和空间位置
    SCENE_KEYWORDS = {
        # 空间位置
        'foreground', 'background', 'center', 'left', 'right', 'top', 'bottom',
        'indoor', 'outdoor', 'inside', 'outside', 'in front of', 'behind',
        'above', 'below', 'beside', 'surrounded by',
        # 时间
        'daytime', 'nighttime', 'sunset', 'sunrise', 'golden hour',
        'morning', 'evening', 'night', 'dawn', 'dusk',
        # 天气
        'sunny', 'rainy', 'cloudy', 'foggy', 'snowy', 'clear', 'stormy'
    }
    
    # 艺术风格指令
    STYLE_KEYWORDS = {
        # 艺术风格
        'oil painting', 'watercolor', 'sketch', 'pencil', 'charcoal',
        'abstract', 'surreal', 'cubism', 'impressionism', 'expressionism',
        'photorealistic', 'realistic', 'stylized', 'cartoon', 'anime',
        'digital art', 'illustration', 'concept art', '3d render',
        # 光影和色彩
        'cinematic', 'dramatic lighting', 'soft lighting', 'rim light',
        'neon', 'dark', 'vibrant', 'desaturated', 'high contrast',
        # 相机和视角
        'wide shot', 'close up', 'macro', 'telephoto', 'fisheye',
        'rule of thirds', 'leading lines', 'bokeh'
    }
    
    def __init__(self):
        """初始化解析器"""
        self.subject_keywords = self.SUBJECT_KEYWORDS
        self.attribute_keywords = self.ATTRIBUTE_KEYWORDS
        self.scene_keywords = self.SCENE_KEYWORDS
        self.style_keywords = self.STYLE_KEYWORDS
    
    def parse(self, prompt: str) -> Dict[str, SemanticElement]:
        """
        解析文本提示词
        
        Args:
            prompt: 输入文本提示词
            
        Returns:
            包含四类语义要素的字典
        """
        prompt_lower = prompt.lower()
        tokens = self._tokenize(prompt_lower)
        
        result = {
            'subject': self._extract_subject(tokens, prompt_lower),
            'attributes': self._extract_attributes(tokens, prompt_lower),
            'scene': self._extract_scene(tokens, prompt_lower),
            'style': self._extract_style(tokens, prompt_lower),
        }
        
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            词语列表
        """
        # 简单分词：按空格和标点符号分割
        tokens = re.findall(r'\b[\w]+\b', text)
        return tokens
    
    def _extract_phrases(self, text: str, min_length: int = 2) -> List[str]:
        """
        提取多词短语
        
        Args:
            text: 输入文本
            min_length: 最小词语数量
            
        Returns:
            短语列表
        """
        # 简单的短语提取（多词连续）
        phrases = re.findall(r'\b[\w\s]{' + str(min_length*2) + r',}\b', text)
        return [p.strip() for p in phrases]
    
    def _extract_subject(self, tokens: List[str], text: str) -> SemanticElement:
        """
        提取主体对象（名词实体）
        
        Args:
            tokens: 分词结果
            text: 原始文本
            
        Returns:
            SemanticElement对象
        """
        matched_subjects = []
        for token in tokens:
            if token in self.subject_keywords:
                matched_subjects.append(token)
        
        # 添加多词主体
        phrases = self._extract_phrases(text)
        for phrase in phrases:
            phrase_tokens = phrase.split()
            if any(t in self.subject_keywords for t in phrase_tokens):
                matched_subjects.append(phrase)
        
        confidence = min(len(matched_subjects) / max(len(tokens), 1), 1.0)
        
        return SemanticElement(
            category=SemanticCategory.SUBJECT,
            tokens=list(set(matched_subjects)),
            confidence=confidence
        )
    
    def _extract_attributes(self, tokens: List[str], text: str) -> SemanticElement:
        """
        提取属性描述（形容词/副词）
        
        Args:
            tokens: 分词结果
            text: 原始文本
            
        Returns:
            SemanticElement对象
        """
        matched_attributes = []
        for token in tokens:
            if token in self.attribute_keywords:
                matched_attributes.append(token)
        
        phrases = self._extract_phrases(text)
        for phrase in phrases:
            phrase_tokens = phrase.split()
            if any(t in self.attribute_keywords for t in phrase_tokens):
                matched_attributes.append(phrase)
        
        confidence = min(len(matched_attributes) / max(len(tokens), 1), 1.0)
        
        return SemanticElement(
            category=SemanticCategory.ATTRIBUTES,
            tokens=list(set(matched_attributes)),
            confidence=confidence
        )
    
    def _extract_scene(self, tokens: List[str], text: str) -> SemanticElement:
        """
        提取场景关系（空间位置）
        
        Args:
            tokens: 分词结果
            text: 原始文本
            
        Returns:
            SemanticElement对象
        """
        matched_scenes = []
        for token in tokens:
            if token in self.scene_keywords:
                matched_scenes.append(token)
        
        phrases = self._extract_phrases(text)
        for phrase in phrases:
            phrase_tokens = phrase.split()
            if any(t in self.scene_keywords for t in phrase_tokens):
                matched_scenes.append(phrase)
        
        confidence = min(len(matched_scenes) / max(len(tokens), 1), 1.0)
        
        return SemanticElement(
            category=SemanticCategory.SCENE,
            tokens=list(set(matched_scenes)),
            confidence=confidence
        )
    
    def _extract_style(self, tokens: List[str], text: str) -> SemanticElement:
        """
        提取风格指令（艺术类型）
        
        Args:
            tokens: 分词结果
            text: 原始文本
            
        Returns:
            SemanticElement对象
        """
        matched_styles = []
        for token in tokens:
            if token in self.style_keywords:
                matched_styles.append(token)
        
        phrases = self._extract_phrases(text)
        for phrase in phrases:
            phrase_tokens = phrase.split()
            if any(t in self.style_keywords for t in phrase_tokens):
                matched_styles.append(phrase)
        
        confidence = min(len(matched_styles) / max(len(tokens), 1), 1.0)
        
        return SemanticElement(
            category=SemanticCategory.STYLE,
            tokens=list(set(matched_styles)),
            confidence=confidence
        )
    
    def get_summary(self, semantic_result: Dict[str, SemanticElement]) -> str:
        """
        获取语义解析摘要
        
        Args:
            semantic_result: 解析结果
            
        Returns:
            摘要字符串
        """
        summary = "语义解析结果:\n"
        for key, element in semantic_result.items():
            summary += f"  {key}: {element.tokens} (confidence: {element.confidence:.2f})\n"
        return summary
